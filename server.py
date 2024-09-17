from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pyspark import SparkConf
from pyspark.sql import SparkSession
from typing import List
from pyspark.sql import functions as F
from pyspark.sql.functions import lit

from scripts.homogenize import homogenize_columns
from scripts.similarity import find_similar_columns

import shutil
import os

app = FastAPI(
    title="Spark API",
    description="API para ejecutar trabajos en Spark y leer archivos CSV.",
    version="1.0.0",
)

UPLOAD_DIR = "/almacenNFS/Spark/Datagenization/csv_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def create_spark_session():
    SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.195.34.24:7077")
    configura = SparkConf()
    configura.setMaster(SPARK_MASTER_URL)
    configura.set('spark.local.dir', '/almacenNFS/Spark/Datagenization/spark_files')
    configura.setAppName("Datagenization")

    spark = SparkSession.builder.config(conf=configura).getOrCreate()

    return spark


def read_csv_to_spark_df(filenames: List[str]):
    spark = create_spark_session()

    all_dataframes = []
    all_columns = set()

    # Primero, leer los archivos y acumular todas las columnas únicas
    for filename in filenames:
        filepath = f"file:///{UPLOAD_DIR}/{filename}"
        if not os.path.exists(os.path.join(UPLOAD_DIR, filename)):
            return {"error": f"Archivo {filename} no encontrado."}

        # Leer el archivo CSV
        df = spark.read.format("csv").option("header", "true").load(filepath)
        all_columns.update(df.columns)
        all_dataframes.append(df)

    # Crear un esquema común con todas las columnas
    all_columns = list(all_columns)

    aligned_dataframes = []
    for df in all_dataframes:
        # Agregar columnas faltantes con valores nulos
        for column in all_columns:
            if column not in df.columns:
                df = df.withColumn(column, lit(None))

        # Asegurarse de que las columnas estén en el orden correcto
        df = df.select(all_columns)
        aligned_dataframes.append(df)

    # Unir todos los DataFrames
    if aligned_dataframes:
        combined_df = aligned_dataframes[0]
        for df in aligned_dataframes[1:]:
            combined_df = combined_df.union(df)
        return combined_df
    else:
        return {"error": "No se encontraron archivos."}


@app.post("/upload_csv", summary="Upload CSV File",
          description="Uploads a CSV file to the server and saves it to the specified directory.")
async def upload_csv(files: list[UploadFile] = File(...)):
    try:
        uploaded_files = []
        for file in files:
            if not file.filename.endswith(".csv"):
                raise HTTPException(status_code=400, detail=f"El archivo {file.filename} debe ser un CSV.")

            file_path = os.path.join(UPLOAD_DIR, file.filename)
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


@app.get("/homogenize", summary="Homogenize CSV Files",
         description="Homogenizes columns in multiple CSV files and returns the result.")
def homogenize(filenames: List[str] = Query(...)):
    try:
        df = read_csv_to_spark_df(filenames)

        if isinstance(df, dict) and "error" in df:
            raise HTTPException(status_code=404, detail=df["error"])

        similar_columns = find_similar_columns(df.columns, 40)

        homogenized_df = homogenize_columns(df, similar_columns)

        pandas_df = homogenized_df.toPandas()
        result = pandas_df.to_dict(orient="records")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
