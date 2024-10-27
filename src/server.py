from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pyspark import SparkConf
from pyspark.sql import SparkSession
from typing import List, Dict
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, DataFrame
from pydantic import BaseModel, conlist, constr, RootModel
from scripts.homogenize import homogenize_across_files
from scripts.similarity import find_similar_columns
from scripts.find_recomendations import find_recomendations
from pyarrow.fs import FileType, FileSelector

import logging
import shutil
import os
import pyarrow.fs as fs

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


hdfs = fs.HadoopFileSystem(host='10.195.34.24', port=9000)
UPLOAD_DIR = "hdfs:///csv_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class FileColumnsMap(BaseModel):
    file_columns: Dict[str, List[str]]


class FileMapColumnsResult(BaseModel):  # Eliminar despues
    file_columns: Dict[str, Dict[str, List[str]]]


def create_spark_session():
    SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.195.34.24:7077")
    configura = SparkConf()
    configura.setMaster(SPARK_MASTER_URL)
    configura.set('spark.local.dir', '../spark_files')
    configura.setAppName("Datagenization")
    configura.set("spark.executor.memory", "50g")

    spark = SparkSession.builder.config(conf=configura).getOrCreate()

    return spark


def read_and_select_columns(filename: str, columns: List[str], session_id: str) -> DataFrame:
    spark = create_spark_session()
    filepath = f"file:///{UPLOAD_DIR}/{filename}"


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
        # os.makedirs(UPLOAD_DIR + "/" + session_id, exist_ok=True)
        hdfs_session_dir = os.path.join(UPLOAD_DIR, session_id)

        try:
            hdfs.create_dir(hdfs_session_dir)
        except Exception as e:
            print(f"El directorio {hdfs_session_dir} ya existe o hubo un error creándolo: {e}")
        uploaded_files = []
        for file in files:
            if not file.filename.endswith(".csv"):
                raise HTTPException(status_code=400, detail=f"El archivo {file.filename} debe ser un CSV.")
            file_path = os.path.join(hdfs_session_dir, file.filename)
            # with open(file_path, "wb") as buffer:
            #    shutil.copyfileobj(file.file, buffer)

            with hdfs.open_output_stream(file_path) as hdfs_file:
                hdfs_file.write(file.file.read())
            uploaded_files.append({"filename": file.filename, "file_path": file_path})

        return {"uploaded_files": uploaded_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read_csv", summary="Get Column Names from All CSV Files",
         description="Reads all CSV files from a session directory in HDFS and returns their column names.")
def read_all_csv_columns(session_id: str = Query(...)):
    try:
        spark = create_spark_session()

        # Definir el directorio de la sesión en HDFS basado en el session_id (sin 'hdfs://')
        session_dir = f"/csv_storage/{session_id}"  # Remover el prefijo hdfs://

        # Listar los archivos en el directorio de HDFS
        hdfs_files = hdfs.get_file_info(FileSelector(session_dir))

        if not hdfs_files:
            raise HTTPException(status_code=404, detail=f"No se encontraron archivos en el directorio {session_id}.")

        columns_by_file = {}

        # Filtrar y procesar solo los archivos CSV
        for file_info in hdfs_files:
            if file_info.type == FileType.File and file_info.path.endswith('.csv'):
                filepath = f"hdfs://{file_info.path}"  # Agregar prefijo hdfs:// para Spark

                # Leer el archivo CSV con Spark
                df = spark.read.format("csv").option("header", "true").load(filepath)

                # Guardar las columnas del archivo
                columns = df.columns
                columns_by_file[file_info.base_name] = columns  # Usar el nombre del archivo como clave

        # Verificar si no se encontraron archivos CSV
        if not columns_by_file:
            raise HTTPException(status_code=404,
                                detail=f"No se encontraron archivos CSV en el directorio {session_id}.")

        return columns_by_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/homogenize", summary="Homogenize Specific Columns from Multiple CSV Files")
def homogenize_files(session_id: str, file_columns_map: Dict[str, Dict[str, List[str]]]):
    try:
        # Llama a la función homogenize_across_files con el mapa de columnas del input
        csv_file_path = homogenize_across_files(file_columns_map, session_id)

        # Verifica si la ruta del archivo CSV fue retornada correctamente
        if not csv_file_path:
            raise HTTPException(status_code=500, detail="No se pudo generar el archivo CSV.")

        # Retorna la ruta del archivo CSV generado
        return {"message": "Archivo CSV generado exitosamente.", "csv_file_path": csv_file_path}
    except HTTPException as http_ex:
        raise http_ex  # Re-lanzar la excepción HTTP
    except Exception as e:
        logging.error(f"Session_id: {session_id}")
        logging.error(f"Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error: {str(e)}")


@app.get("/recomendations", summary="Makes recommendations to homogenize",
         description="Makes recommendations to homogenize based on the columns.")
def recomendations(session_id: str = Query(...)):
    try:
        similarity = find_recomendations(session_id)
        return {"similarity": similarity}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download_csv", summary="Download CSV File by Session ID",
         description="Downloads the CSV file that matches the session_id from the HDFS directory.")
def download_csv(session_id: str = Query(...)):
    try:

        session_dir = f"/csv_storage/{session_id}"

        hdfs_files = hdfs.get_file_info(FileSelector(session_dir))

        if not hdfs_files:
            raise HTTPException(status_code=404, detail=f"No se encontraron archivos en el directorio {session_id}.")

        csv_file_path = None
        for file_info in hdfs_files:
            if file_info.type == fs.FileType.File and file_info.path.endswith('.csv'):

                if f"{session_id}_combined.csv" in file_info.path:
                    csv_file_path = file_info.path
                    break

        if csv_file_path is None:
            raise HTTPException(status_code=404, detail=f"Archivo CSV '{session_id}_combined.csv' no encontrado.")

        local_file_path = f"/tmp/{session_id}_combined.csv"

        with hdfs.open_input_file(csv_file_path) as hdfs_file, open(local_file_path, "wb") as local_file:
            local_file.write(hdfs_file.read())

        return FileResponse(local_file_path, media_type='text/csv', filename=f"{session_id}_combined.csv")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preview_csv", summary="Preview CSV File by Session ID",
         description="Previews the CSV file that matches the session_id from the HDFS directory.")
def preview_csv(session_id: str = Query(...), limit: int = 10):
    try:
        spark = create_spark_session()

        session_dir = f"/csv_storage/{session_id}"

        hdfs_files = hdfs.get_file_info(FileSelector(session_dir))

        if not hdfs_files:
            raise HTTPException(status_code=404, detail=f"No se encontraron archivos en el directorio {session_id}.")

        csv_file_path = None
        for file_info in hdfs_files:
            if file_info.type == FileType.File and file_info.path.endswith('.csv'):
                if f"{session_id}_combined.csv" in file_info.path:
                    csv_file_path = f"hdfs://{file_info.path}"
                    break

        if csv_file_path is None:
            raise HTTPException(status_code=404,
                                detail=f"Archivo CSV con el nombre '{session_id}_combined.csv' no encontrado.")

        df = spark.read.format("csv").option("header", "true").load(csv_file_path)
        preview_data = df.limit(limit).toPandas().to_dict(orient="records")

        return preview_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/delete", summary="Deletes the session Id file.",
         description="Deletes the session Id file with all the CSV.")
def deleteFiles(session_id: str = Query(...)):
    try:
        session_dir = f"/csv_storage/{session_id}"

        if not hdfs.get_file_info(FileSelector(session_dir), strict=False):
            raise HTTPException(status_code=404, detail=f"No se encontró el directorio {session_id}.")

        hdfs.delete(session_dir, recursive=True)

        return {"detail": f"Directorio {session_id} eliminado con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



##