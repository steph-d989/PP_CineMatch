import json
from pathlib import Path

import pandas as pd


### FUNCION QUE VERIFICA SI UN ARCHIVO EXISTE

def validate_file_exists(path: Path) -> bool:

    return path.exists() and path.is_file()


### FUNCION QUE VERIFICA QUE UNA CARPETA EXISTE

def validate_directory_exists(path: Path) -> bool:

    return path.exists() and path.is_dir()


### FUNCION QUE VERIFICA QUE UN ARCHIVO JSON EXISTE Y SE PUEDE LEER

def validate_json_file(path: Path) -> dict:

    if not validate_file_exists(path):
        return {
            "valid": False,
            "error": "file_not_found"
        }

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            "valid": True,
            "data_type": type(data).__name__
        }

    except (json.JSONDecodeError, OSError) as error:
        return {
            "valid": False,
            "error": str(error)
        }


### FUNCION QUE VERIFICA QUE UN CSV EXISTE Y PUEDA CARGARSE
 
def validate_csv_file(path: Path) -> dict:

    if not validate_file_exists(path):
        return {
            "valid": False,
            "error": "file_not_found"
        }

    try:
        df = pd.read_csv(path)

        return {
            "valid": True,
            "rows": len(df),
            "columns": len(df.columns)
        }

    except Exception as error:
        return {
            "valid": False,
            "error": str(error)
        }


### FUNCION QUE CONFIRMA QUE UN DATAFRAME TENGA LAS COLUMNAS ESPERADAS

def validate_expected_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> dict:

    missing_columns = [
        column
        for column in expected_columns
        if column not in df.columns
    ]

    return {
        "valid": len(missing_columns) == 0,
        "missing_columns": missing_columns
    }