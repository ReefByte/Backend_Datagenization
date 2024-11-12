import unicodedata
from pyspark.sql.functions import col, coalesce, lit
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


def find_recommendations(session_id: str):
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
    sample_dfs()

    # file_relation()
    return column_name_analysis()


def renombrar_columnas(df):
    nuevos_nombres = {columna: columna.strip().replace("...", "_%&%_") for columna in df.columns}
    for columna_original, columna_nueva in nuevos_nombres.items():
        df = df.withColumnRenamed(columna_original, columna_nueva)
    return df


def column_name_analysis(threshold=90):
    recommendation_num = 0
    recommendations = {}
    pairs = []

    n = len(dfs)
    i = -1
    for k in range(n - 1, i, -1):
        print("K ", k)
        for j in range(k, i, -1):
            print(j, k)
            for col_name in list(dfs.values())[k].columns:
                similar_columns = process.extractBests(col_name, list(dfs.values())[j].columns,
                                                       scorer=fuzz.token_sort_ratio, limit=None)
                similar_cols = [col[0] for col in similar_columns if col[1] >= threshold and col[0] != col_name]

                if similar_cols:
                    if len(similar_cols) == 1:
                        if (similar_cols[0], col_name) not in pairs:
                            pairs.append((col_name, similar_cols[0]))
                            recommendations[f"Recommendation{recommendation_num}"] = {}
                            if list(dfs.keys())[k] == list(dfs.keys())[j]:
                                recommendations[f"Recommendation{recommendation_num}"][
                                    (str(list(dfs.keys())[k]) + "_").split("/")[-1]] = col_name
                                recommendations[f"Recommendation{recommendation_num}"][
                                    str(list(dfs.keys())[j]).split("/")[-1]] = similar_cols
                                recommendation_num += 1
                            else:
                                recommendations[f"Recommendation{recommendation_num}"][
                                    str(list(dfs.keys())[k]).split("/")[-1]] = col_name
                                recommendations[f"Recommendation{recommendation_num}"][
                                    str(list(dfs.keys())[j]).split("/")[-1]] = similar_cols
                                recommendation_num += 1
                    else:
                        recommendations[f"Recommendation{recommendation_num}"] = {}
                        if list(dfs.keys())[k] == list(dfs.keys())[j]:
                            recommendations[f"Recommendation{recommendation_num}"][
                                (str(list(dfs.keys())[k]) + "_").split("/")[-1]] = col_name
                            recommendations[f"Recommendation{recommendation_num}"][
                                str(list(dfs.keys())[j]).split("/")[-1]] = similar_cols
                            recommendation_num += 1
                        else:
                            recommendations[f"Recommendation{recommendation_num}"][
                                str(list(dfs.keys())[k]).split("/")[-1]] = col_name
                            recommendations[f"Recommendation{recommendation_num}"][
                                str(list(dfs.keys())[j]).split("/")[-1]] = similar_cols
                            recommendation_num += 1

                similar_cols2 = [col[0] for col in similar_columns if col[1] >= threshold and col[0] == col_name]

                if similar_cols2:
                    if str(list(dfs.keys())[k]) != str(list(dfs.keys())[j]):
                        recommendations[f"Recommendation{recommendation_num}"] = {}
                        recommendations[f"Recommendation{recommendation_num}"][
                            (str(list(dfs.keys())[k]) + "_").split("/")[-1]] = col_name
                        recommendations[f"Recommendation{recommendation_num}"][
                            str(list(dfs.keys())[j]).split("/")[-1]] = similar_cols2
                        recommendation_num += 1

    return recommendations


def tfidf_analysis(df1, df2, df1_name, df2_name):
    df1 = renombrar_columnas(df1)
    df2 = renombrar_columnas(df2)
    for columndf1 in df1.columns:
        for columndf2 in df2.columns:
            try:
                # Seleccionar las columnas utilizando 'col'
                df1_pandas = df1.select(col(f"{columndf1}")).toPandas()
                df2_pandas = df2.select(col(f"{columndf2}")).toPandas()
            except Exception as e:
                print(f"ERROR al seleccionar las columnas: {columndf1}, {columndf2}")
                print("Detalle del error:", e)

            # Aplicar el preprocesamiento a cada fila de las columnas seleccionadas
            df1_pandas['columna1'] = df1_pandas[columndf1].fillna('').astype(str).apply(preprocess_text)
            df2_pandas['columna2'] = df2_pandas[columndf2].fillna('').astype(str).apply(preprocess_text)

            # Concatenar todas las filas de cada columna en un solo texto (un gran string por columna)
            text1 = ' '.join(df1_pandas['columna1'].tolist())
            text2 = ' '.join(df2_pandas['columna2'].tolist())

            # Crear un DataFrame de Pandas con estos dos textos
            texts = [text1, text2]
            try:
                # Aplicar TF-IDF
                tfidf_vectorizer = TfidfVectorizer()
                tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            except:
                print("ERROR", columndf1, columndf2)

            # Calcular la similitud de coseno entre los dos textos
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

            # Mostrar el valor de la similitud
            if similarity[0][0] > 0.8:
                print(
                    f"Similitud de coseno entre {columndf1} de {df1_name} y {columndf2} de {df2_name} tratadas como textos completos: {similarity[0][0]}")
            # print("I")


def sample_dfs():
    for df_key in dfs.keys():
        dfs[df_key] = dfs[df_key].sample(False, 0.05, seed=274)
    return dfs


def file_relation():
    n = len(dfs) - 1
    i = 0
    comparaciones = []
    while (n != 0):
        j = i
        for j in range(j, n):
            # tfidf_analysis(list(dfs.values())[j],list(dfs.values())[n], str(list(dfs.keys())[j]).split("/")[-1], str(list(dfs.keys())[n]).split("/")[-1])
            # tfidf_analysis(list(dfs.values())[j],list(dfs.values())[n], list(dfs.keys())[j], list(dfs.keys())[n])
            # sparktfidf(list(dfs.values())[j],list(dfs.values())[n], str(list(dfs.keys())[j]).split("/")[-1], str(list(dfs.keys())[n]).split("/")[-1])
            print("FLAGGG", (j, n))
        n -= 1


def sparktfidf(df1, df2, df1_name, df2_name):
    vectorizer = TfidfVectorizer()
    results = []

    for columndf1 in df1.columns:
        text1 = ' '.join([row[columndf1] for row in df1.select(columndf1).collect()])
        for columndf2 in df2.columns:
            text2 = ' '.join([row[columndf2] for row in df2.select(columndf2).collect()])
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0][0]
            if cosine_similarities > 0.8:
                # Imprimir similitud en la consola
                print(
                    f"Similitud de coseno entre {columndf1} de {df1_name} y {columndf2} de {df2_name}: {cosine_similarities}")

                results.append({
                    "column_df1": columndf1,
                    "column_df2": columndf2,
                    "similarity": cosine_similarities,
                    "df1_name": df1_name,
                    "df2_name": df2_name
                })
    return results