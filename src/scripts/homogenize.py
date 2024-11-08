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
from IPython.display import display

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

        file_info = hdfs.get_file_info(filepath)
        if file_info is None or file_info.type != fs.FileType.File:
            raise HTTPException(status_code=404, detail=f"Archivo {filename} no encontrado en la ruta {filepath}")

        spark = create_spark_session()

        df = spark.read.csv(filepath, header=True)

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400,
                                detail=f"Faltan las columnas: {', '.join(missing_columns)} en el archivo {filename}")

        selected_df = df.select(*columns)

        return selected_df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")


def read_and_select_homogenize(filename: str, columns: List[str], session_id: str) -> DataFrame:
    try:
        filepath = f"/csv_storage/{session_id}/{filename}"

        file_info = hdfs.get_file_info(filepath)
        if file_info is None or file_info.type != fs.FileType.File:
            raise HTTPException(status_code=404, detail=f"Archivo {filename} no encontrado en la ruta {filepath}")

        spark = create_spark_session()

        df = spark.read.csv(filepath, header=True).select(*columns)

        print(df)

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400,
                                detail=f"Faltan las columnas: {', '.join(missing_columns)} en el archivo {filename}")

        return df

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")


def homogenize_across_files(files_column_map: Dict[str, Dict[str, List[str]]], session_id: str) -> str:
    try:
        combined_df = None
        collectedColumns = {}

        # Obtenemos los nombres de las columnas principales desde las claves del JSON
        main_columns = list(files_column_map.keys())

        # Recolectamos todas las columnas necesarias por archivo
        for main_col, file_data in files_column_map.items():
            for filename, related_cols in file_data.items():
                if filename not in collectedColumns:
                    collectedColumns[filename] = []
                collectedColumns[filename].extend(related_cols)

        # Procesamos los archivos con todas las columnas recolectadas
        for filename, columns in collectedColumns.items():
            try:
                # Cargar el archivo con todas las columnas seleccionadas
                df = read_and_select_homogenize(filename, columns, session_id)

                # Crear una columna única para hacer los joins
                df = df.withColumn("unique_id", F.monotonically_increasing_id())

                # Crear un DataFrame temporal para manejar renombres específicos por archivo
                temp_df = df

                # Renombrar y combinar las columnas según los nombres en el JSON
                for main_col, file_data in files_column_map.items():
                    if filename in file_data:
                        relevant_cols = [col for col in file_data[filename] if col in df.columns]
                        if len(relevant_cols) > 1:
                            # Concatenar columnas múltiples en una sola si es necesario
                            temp_df = temp_df.withColumn(main_col,
                                                         F.concat_ws(" ", *[F.col(col) for col in relevant_cols]))
                            for col in relevant_cols:
                                temp_df = temp_df.drop(col)
                        elif relevant_cols:
                            # Renombrar la columna si solo hay una relacionada con `main_col`
                            temp_df = temp_df.withColumnRenamed(relevant_cols[0], main_col)

                # Seleccionar solo las columnas relevantes y unique_id, verificando su existencia en temp_df
                final_cols = [col for col in main_columns if col in temp_df.columns] + ["unique_id"]
                temp_df = temp_df.select(*final_cols)

                # Consolidar el DataFrame actual con el DataFrame combinado
                if combined_df is None:
                    combined_df = temp_df
                else:
                    combined_df = combined_df.unionByName(temp_df, allowMissingColumns=True)

            except Exception as e:
                print(f"Error al procesar el archivo {filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error al leer el archivo {filename}: {str(e)}")

        if combined_df is None:
            print("No se encontraron datos para combinar después de procesar todos los archivos.")
            raise HTTPException(status_code=500, detail="No se encontraron datos para combinar.")

        # Convertir a pandas y llenar valores ausentes
        pandas_df = combined_df.toPandas()
        pandas_df = pandas_df.fillna('')

        # Eliminar cualquier columna adicional que no sea las columnas principales
        pandas_df = pandas_df[main_columns]

        # Convertir a Arrow y escribir en HDFS
        arrow_table = pa.Table.from_pandas(pandas_df)
        output_dir = f"/csv_storage/{session_id}/"
        output_path = f"{output_dir}{session_id}_combined.csv"

        with hdfs.open_output_stream(output_path) as out_stream:
            csv.write_csv(arrow_table, out_stream)

        return output_path

    except Exception as e:
        print(f"Error inesperado durante la homogenización: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")