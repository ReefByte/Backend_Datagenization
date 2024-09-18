from pyspark.sql.functions import col, coalesce
from fuzzywuzzy import process, fuzz

def find_similar_columns(columns, threshold=80):
    
    column_mapping = {}
    used_columns = set()

    for col_name in columns:
        if col_name in used_columns:
            continue

        similar_cols = process.extractBests(col_name, columns, scorer=fuzz.token_sort_ratio, limit=None)
        similar_cols = [col[0] for col in similar_cols if col[1] >= threshold and col[0] != col_name]

        if similar_cols:
            similar_cols.append(col_name)
            column_mapping[col_name] = similar_cols
            used_columns.update(similar_cols)

    return column_mapping