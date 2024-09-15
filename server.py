from fastapi import FastAPI, File, UploadFile, HTTPException
from pyspark import SparkConf
from pyspark.sql import SparkSession

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

def crear_sesion_spark():
    SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.195.34.24:7077")
    configura = SparkConf()
    configura.setMaster(SPARK_MASTER_URL)
    configura.set('spark.local.dir', '/almacenNFS/Spark/Datagenization/spark_files')
    configura.setAppName("Datagenization")

    spark = SparkSession.builder.config(conf=configura).getOrCreate()

    return spark


def read_csv_to_spark_df(filename: str):
    try:
        spark = crear_sesion_spark()

        filepath = f"file:///{UPLOAD_DIR}/{filename}"
        df = spark.read.format("csv").option("header", "true").load(filepath)

        return df
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload_csv", summary="Upload CSV File", description="Uploads a CSV file to the server and saves it to the specified directory.")
async def upload_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="El archivo debe ser un CSV.")

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read_csv/{filename}", summary="Read CSV File", description="Reads a CSV file and returns its contents as a JSON array of records.")
def read_csv(filename: str):
    try:
        spark = crear_sesion_spark()

        filepath = f"file:///{UPLOAD_DIR}/{filename}"
        df = spark.read.format("csv").option("header", "true").load(filepath)

        result = df.toPandas().to_dict(orient="records")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/homogenize/{filename}")
def homogenize(filename: str):
    
    try:
        df = read_csv_to_spark_df(filename)

        if isinstance(df, dict) and "error" in df:
            return df
    
        columns = df.columns
        
        similar_columns = find_similar_columns(columns, 85)
    
        homogenized_df = homogenize_columns(df, similar_columns)
        
        pandas_df = homogenized_df.toPandas()
    
        return pandas_df.to_dict(orient="records")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
