import unicodedata
from pyspark.sql.functions import col, coalesce, lit, isnan, when, count
from pyspark import SparkConf
from pyspark.sql import SparkSession
import pandas as pd
import os
from pyspark.sql.types import DecimalType, DoubleType, StringType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.fs as fs
from pyarrow.fs import FileType, FileSelector
from multiprocessing import Pool, cpu_count
from fuzzywuzzy import process, fuzz
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

hdfs = fs.HadoopFileSystem(host='10.195.34.24', port=9000)
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
    df = None
    spark = create_spark_session()
    session_dir = f"/csv_storage/{session_id}"
    hdfs_files = hdfs.get_file_info(FileSelector(session_dir))

    if not hdfs_files:
        raise HTTPException(status_code=404, detail=f"No se encontraron archivos en el directorio {session_id}.")

    for file_info in hdfs_files:
        if file_info.type == FileType.File and file_info.path.endswith(f'{session_id}_combined.csv'):
            filepath = f"hdfs://{file_info.path}"  # Agregar prefijo hdfs:// para Spark
            df = spark.read.format("csv").option("header", "true").load(filepath)
    recommendations_body = check_null_values(df)
    recommendations_body = check_outliers(df, recommendations_body)
    # result_body["outlier_img"] = check_outliers(df)
    # pca_analysis()
    return recommendations_body


def check_null_values(df):
    result_body = {}
    nulls_df = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    nulls_info = nulls_df.collect()[0].asDict()
    for column, null_count in nulls_info.items():
        print(null_count)
        if null_count != 0:
            result_body[column] = {}
            result_body[column]["nulls"] = null_count
    return result_body


def get_numeric_columns(df):
    columnas_convertidas = []
    for column in df.columns:
        try:
            df = df.withColumn(column, col(column).cast(DoubleType()))
            if df.select(column).filter(col(column).isNotNull()).count() > 0:
                columnas_convertidas.append(column)
            else:
                df = df.withColumn(column, col(column).cast("string"))
        except Exception as e:
            df = df.withColumn(column, col(column).cast("string"))
            print(f"No se pudo convertir la columna '{column}' a DoubleType: {e}")

    df_convertido = df.select(columnas_convertidas)
    return df_convertido


def check_outliers(df, recommendations_body):
    df_numeric = get_numeric_columns(df)
    for column in df_numeric.columns:
        try:
            quantiles = df_numeric.approxQuantile(str(column), [0.25, 0.75], 0.05)
            print("QUANTILES ", quantiles)
            Q1, Q3 = quantiles[0], quantiles[1]
            IQR = Q3 - Q1

            # Definir límites para detectar outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Filtrar los outliers
            outliers = df_numeric.filter((col(column) < lower_bound) | (col(column) > upper_bound))
            outliers.show()

            df_pandas = df_numeric.select(column).toPandas()
            outliers_pandas = outliers.select(column).toPandas()

            plt.figure(figsize=(10, 6))
            plt.hist(df_pandas[column], bins=30, alpha=0.5, label='Datos')
            plt.hist(outliers_pandas[column], bins=30, alpha=0.5, label='Outliers', color='red')
            plt.xlabel(column)
            plt.ylabel('Frecuencia')
            plt.title(f'Distribución de {column} con Outliers resaltados')
            plt.legend()

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)

            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            buffer.close()
            plt.close()

            print(f"Imagen en base64 para la columna {column}: {img_base64}")
            recommendations_body[column]["outliers"] = img_base64
        except  Exception as e:
            print(f"ERROR CON COLUMNA {column} {e}")
    return recommendations_body


def get_categorical_columns(df):
    columnas_convertidas = []
    for column in df.columns:
        try:
            df = df.withColumn(column, col(column).cast(DoubleType()))
            if df.select(column).filter(col(column).isNotNull()).count() > 0:
                columnas_convertidas.append(column)
            else:
                df = df.withColumn(column, col(column).cast("string"))
        except Exception as e:
            df = df.withColumn(column, col(column).cast("string"))
            print(f"No se pudo convertir la columna '{column}' a DoubleType: {e}")
    df_convertido = df.select(columnas_convertidas)
    return df_convertido


def pca_analysis(df):
    df_numeric = get_numeric_columns(df)
    df_numeric = get_categorical_columns(df)
    assembler = VectorAssembler(inputCols=df.columns, outputCol='features')
    df_ = assembler.transform(df)
    num_features = len(df.select('features').first()[0])

    varianza_acumulada = []

    for k in range(1, num_features + 1):
        pca = PCA(k=k, inputCol='features', outputCol='pca_features')
        model = pca.fit(df_)
        explained_variance = model.explainedVariance.toArray()
        varianza_acumulada.append((k, explained_variance.sum()))

        for k, varianza in varianza_acumulada:
            print(f"Componentes: {k}, Varianza explicada acumulada: {varianza:.4f}")










