from fastapi import HTTPException

from pyspark.sql.functions import col, coalesce, lit
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from IPython.display import FileLink

import pyarrow.fs as fs
import pyarrow as pa
import pyarrow.csv as csv

from typing import Dict, List
import os
import time

hdfs = fs.HadoopFileSystem(host='10.195.34.24', port=9000)

UPLOAD_DIR = "hdfs:///csv_storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def create_spark_session():
    SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.195.34.24:7077")
    configura = SparkConf()
    configura.setMaster(SPARK_MASTER_URL)
    configura.set('spark.local.dir', '../../spark_files')
    configura.setAppName("Datagenization")

    spark = SparkSession.builder.config(conf=configura).getOrCreate()

    return spark


def read_and_select_columns(filename: str, columns: List[str], session_id: str) -> DataFrame:
    try:
        # Construir la ruta completa del archivo en HDFS
        filepath = f"/csv_storage/{session_id}/{filename}"

        # Verificar si el archivo existe en HDFS
        file_info = hdfs.get_file_info(filepath)
        if file_info is None or file_info.type != fs.FileType.File:
            raise HTTPException(status_code=404, detail=f"Archivo {filename} no encontrado en la ruta {filepath}")

        spark = create_spark_session()

        # Leer el archivo CSV desde HDFS
        df = spark.read.csv(filepath, header=True)

        # Verificar que todas las columnas necesarias existan
        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400,
                                detail=f"Faltan las columnas: {', '.join(missing_columns)} en el archivo {filename}")

        # Seleccionar las columnas deseadas
        selected_df = df.select(*columns)

        return selected_df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")


def homogenize_across_files(file_columns: Dict[str, Dict[str, List[str]]], session_id: str) -> str:
    try:
        all_dataframes = {}

        for main_col, file_data in file_columns.items():
            for filename, related_cols in file_data.items():
                try:

                    df = read_and_select_columns(filename, related_cols, session_id)
                    all_dataframes[filename] = df
                    print(f"Archivo {filename} leído con columnas: {df.columns}")
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")

        combined_df = None

        for main_col, file_data in file_columns.items():
            for filename, related_cols in file_data.items():
                df = all_dataframes[filename]

                if df is None or df.count() == 0:
                    print(f"El DataFrame para {filename} está vacío o es None. Se omitirá.")
                    continue

                print(f"Combinando {main_col} desde {filename} con columnas {related_cols}")

                existing_cols = [col for col in related_cols if col in df.columns]
                if existing_cols:
                    combined_df = df.select(*existing_cols)

                    if combined_df is not None:
                        combined_df = combined_df.withColumn(main_col, F.coalesce(*[df[col] for col in existing_cols]))

                else:
                    print(f"Ninguna de las columnas {related_cols} está presente en {filename}.")

                combined_df = combined_df.unionByName(df.select(*related_cols), allowMissingColumns=True)

        pandas_df = combined_df.toPandas()

        arrow_table = pa.Table.from_pandas(pandas_df)

        output_dir = f"/csv_storage/{session_id}/"
        output_path = f"{output_dir}{session_id}_combined.csv"  # Nombre del archivo CSV

        with hdfs.open_output_stream(output_path) as out_stream:
            csv.write_csv(arrow_table, out_stream)

        return output_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")


def download_file(filepath):
    return FileLink(filepath)
