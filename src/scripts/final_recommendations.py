import unicodedata
from pyspark.sql.functions import col, coalesce, lit, isnan, when, count
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.fs as fs
from pyarrow.fs import FileType, FileSelector
from multiprocessing import Pool, cpu_count
from fuzzywuzzy import process, fuzz

hdfs = fs.HadoopFileSystem(host='10.195.34.24', port=9000)
dfs = {}
result_body = {}
UPLOAD_DIR = "hdfs:///csv_storage"


def create_spark_session():
    SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.195.34.24:7077")
    configura = SparkConf()
    configura.setMaster(SPARK_MASTER_URL)
    configura.set('spark.local.dir', '../spark_files')
    configura.setAppName("Datagenization")

    spark = SparkSession.builder.config(conf=configura).getOrCreate()

    return spark


def preprocess_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = text.lower()
    text = text.strip()
    text = ' '.join(text.split())
    return text


def final_recommendations(session_id: str):
    spark = create_spark_session()
    session_dir = f"/csv_storage/{session_id}"  # Remover el prefijo hdfs://

    # Listar los archivos en el directorio de HDFS
    hdfs_files = hdfs.get_file_info(FileSelector(session_dir))

    if not hdfs_files:
        raise HTTPException(status_code=404, detail=f"No se encontraron archivos en el directorio {session_id}.")

        # Filtrar y procesar solo los archivos CSV
    for file_info in hdfs_files:
        if file_info.type == FileType.File and file_info.path.endswith('.csv'):
            filepath = f"hdfs://{file_info.path}"  # Agregar prefijo hdfs:// para Spark
            dfs[filepath] = spark.read.format("csv").option("header", "true").load(filepath)
    check_null_values()
    return result_body


def check_null_values():
    result_body["nulls_info"] = {}
    for df in dfs:
        nulls_df = dfs[df].select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in dfs[df].columns])
        nulls_info = nulls_df.collect()[0].asDict()
        for column, null_count in nulls_info.items():
            print(null_count)
            if null_count != 0:
                result_body["nulls_info"][column] = null_count


def check_outliers():
    pass
