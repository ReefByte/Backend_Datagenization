from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from pyspark import SparkConf
from pyspark.sql import SparkSession
from typing import List, Dict
from pyspark.sql import functions as F
from pyspark.sql.functions import lit
from pydantic import BaseModel

from scripts.homogenize import homogenize_columns
from scripts.homogenize import homogenize_across_files

from scripts.similarity import find_similar_columns

import shutil
import os

app = FastAPI(
    title="Spark API",
    description="API para ejecutar trabajos en Spark y leer archivos CSV.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "../csv_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class FileColumnsMap(BaseModel):
    file_columns: Dict[str, List[str]]


def create_spark_session():
    SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.195.34.24:7077")
    configura = SparkConf()
    configura.setMaster(SPARK_MASTER_URL)
    configura.set('spark.local.dir', '../spark_files')
    configura.setAppName("Datagenization")

    spark = SparkSession.builder.config(conf=configura).getOrCreate()

    return spark


def read_and_select_columns(filename: str, columns: List[str]):
    spark = create_spark_session()
    filepath = f"file:///{UPLOAD_DIR}/{filename}"

    if not os.path.exists(os.path.join(UPLOAD_DIR, filename)):
        raise HTTPException(status_code=404, detail=f"Archivo {filename} no encontrado.")

    df = spark.read.format("csv").option("header", "true").load(filepath)

    for column in columns:
        if column not in df.columns:
            df = df.withColumn(column, lit(None))

    return df.select(columns)


def read_csv_to_spark_df(filenames: List[str]):
    spark = create_spark_session()

    all_dataframes = []
    all_columns = set()

    for filename in filenames:
        filepath = f"file:///{UPLOAD_DIR}/{filename}"
        if not os.path.exists(os.path.join(UPLOAD_DIR, filename)):
            return {"error": f"Archivo {filename} no encontrado."}

        df = spark.read.format("csv").option("header", "true").load(filepath)
        all_columns.update(df.columns)
        all_dataframes.append(df)

    all_columns = list(all_columns)

    aligned_dataframes = []
    for df in all_dataframes:
        for column in all_columns:
            if column not in df.columns:
                df = df.withColumn(column, lit(None))

        df = df.select(all_columns)
        aligned_dataframes.append(df)

    if aligned_dataframes:
        combined_df = aligned_dataframes[0]
        for df in aligned_dataframes[1:]:
            combined_df = combined_df.union(df)
        return combined_df
    else:
        return {"error": "No se encontraron archivos."}


@app.post("/upload_csv", summary="Upload CSV File",
          description="Uploads a CSV file to the server and saves it to the specified directory.")
async def upload_csv(files: List[UploadFile] = File(...), session_id: str = Form(...)):
    try:
        os.makedirs(UPLOAD_DIR + "/" + session_id, exist_ok=True)
        uploaded_files = []
        for file in files:
            if not file.filename.endswith(".csv"):
                raise HTTPException(status_code=400, detail=f"El archivo {file.filename} debe ser un CSV.")

            file_path = os.path.join(UPLOAD_DIR + "/" + session_id, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            uploaded_files.append({"filename": file.filename, "file_path": file_path})

        return {"uploaded_files": uploaded_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read_csv", summary="Get Column Names from CSV Files",
         description="Reads multiple CSV files and returns their column names.")
def read_csv_columns(filenames: List[str] = Query(...)):
    try:
        spark = create_spark_session()

        columns_by_file = {}

        for filename in filenames:
            filepath = f"file:///{UPLOAD_DIR}/{filename}"
            if not os.path.exists(os.path.join(UPLOAD_DIR, filename)):
                raise HTTPException(status_code=404, detail=f"Archivo {filename} no encontrado.")

            df = spark.read.format("csv").option("header", "true").load(filepath)

            columns = df.columns
            columns_by_file[filename] = columns

        return columns_by_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/homogenize", summary="Homogenize Specific Columns from Multiple CSV Files")
def homogenize_files(file_columns_map: FileColumnsMap):
    try:
        combined_df = homogenize_across_files(file_columns_map.file_columns)

        if combined_df is None:
            raise HTTPException(status_code=500, detail="No se pudo combinar los archivos.")

        pandas_df = combined_df.toPandas()
        result = pandas_df.to_dict(orient="records")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
