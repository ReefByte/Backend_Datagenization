from fastapi import HTTPException

from pyspark.sql.functions import col, coalesce, lit, monotonically_increasing_id
from pyspark import SparkConf
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from IPython.display import FileLink
from pyspark.sql.window import Window

import pyarrow.fs as fs
from pyarrow.fs import FileSelector
import pyarrow as pa
import pyarrow.csv as csv
import pandas as pd

from typing import Dict, List
import os
import numpy as np
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

        print(selected_df)

        return selected_df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")


def homogenize_across_files(file_columns: Dict[str, Dict[str, List[str]]], session_id: str) -> str:
    try:  # Version - 3.4.0
        combined_df = None

        for main_col, file_data in file_columns.items():
            temp_dataframes = []

            for filename, related_cols in file_data.items():
                try:
                    # Leer y seleccionar columnas del archivo
                    df = read_and_select_columns(filename, related_cols, session_id)
                    if df is None or df.count() == 0:
                        print(f"El DataFrame para {filename} está vacío o es None. Se omitirá.")
                        continue

                    # Añadir una columna de identificador único para mantener el alineamiento de las filas
                    df = df.withColumn("unique_id", monotonically_increasing_id())

                    # Renombrar columnas para evitar ambigüedades (añadir contexto del archivo)
                    for col in related_cols:
                        if col in df.columns:
                            new_col_name = f"{main_col}_{filename.replace('.csv', '')}"
                            df = df.withColumnRenamed(col, new_col_name)

                    # Seleccionar columnas renombradas y el identificador único
                    df = df.select(
                        [f"{main_col}_{filename.replace('.csv', '')}" for col in related_cols] + ["unique_id"])
                    temp_dataframes.append(df)

                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")

            # Combinar todos los DataFrames temporales para el mismo main_col
            if temp_dataframes:
                combined_temp_df = temp_dataframes[0]
                for temp_df in temp_dataframes[1:]:
                    combined_temp_df = combined_temp_df.join(temp_df, on="unique_id", how="outer")

                # Consolidar todas las columnas relacionadas en una sola columna principal
                consolidated_col = None
                for col in combined_temp_df.columns:
                    if col != "unique_id":
                        if consolidated_col is None:
                            consolidated_col = combined_temp_df[col]
                        else:
                            consolidated_col = coalesce(consolidated_col, combined_temp_df[col])
                combined_temp_df = combined_temp_df.withColumn(main_col, consolidated_col)
                combined_temp_df = combined_temp_df.select(main_col, "unique_id")

                # Añadir al DataFrame combinado
                if combined_df is None:
                    combined_df = combined_temp_df
                else:
                    combined_df = combined_df.join(combined_temp_df, on="unique_id", how="outer")

        if combined_df is None:
            raise HTTPException(status_code=500, detail="No se encontraron datos para combinar.")

        # Convertir a Pandas y guardar en HDFS
        pandas_df = combined_df.toPandas()

        # Reemplazar NaN por un string vacío si es necesario
        pandas_df = pandas_df.fillna('')

        # Asegurarse de que todas las columnas originales estén presentes
        for main_col in file_columns.keys():
            if main_col not in pandas_df.columns:
                pandas_df[main_col] = ''

        # Debug: Mostrar información sobre las columnas y filas
        print(f"Columnas finales: {pandas_df.columns.tolist()}")
        print(f"Número de filas en el DataFrame combinado: {len(pandas_df)}")

        # Eliminar la columna "unique_id" antes de guardar
        pandas_df = pandas_df.drop(columns=["unique_id"], errors='ignore')

        arrow_table = pa.Table.from_pandas(pandas_df)

        output_dir = f"/csv_storage/{session_id}/"
        output_path = f"{output_dir}{session_id}_combined.csv"

        with hdfs.open_output_stream(output_path) as out_stream:
            csv.write_csv(arrow_table, out_stream)

        return output_path

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")