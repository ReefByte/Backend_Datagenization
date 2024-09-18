from pyspark.sql.functions import col, coalesce
from pyspark import SparkConf
from pyspark.sql import SparkSession

from typing import Dict, List
import os

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

def homogenize_columns(df, similar_columns):
    selected_cols = set()
    
    for target_col, similar_cols in similar_columns.items():
        cols_to_coalesce = [col(similar_col) for similar_col in similar_cols if similar_col in df.columns]
        
        if cols_to_coalesce:
            df = df.withColumn(target_col, coalesce(*cols_to_coalesce))
            selected_cols.add(target_col)
    
    return df.select(*selected_cols)


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


def homogenize_across_files(file_columns: Dict[str, List[str]]):
    all_dataframes = []
    for filename, columns in file_columns.items():
        df = read_and_select_columns(filename, columns)
        all_dataframes.append(df)

    if all_dataframes:
        combined_df = all_dataframes[0]
        for df in all_dataframes[1:]:
            combined_df = combined_df.unionByName(df, allowMissingColumns=True)
        return combined_df
    else:
        return None