"""Validaciones básicas y eficientes para las fuentes Bronze de CineMatch.

    Las funciones mantienen los nombres y campos de retorno utilizados por los
    módulos de ingesta existentes. Las comprobaciones de esquema son opcionales.
"""

import json
from pathlib import Path

import pandas as pd


MOVIELENS_EXPECTED_COLUMNS = {
    "movies.csv": ["movieId", "title", "genres"],
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
    "genome-scores.csv": ["movieId", "tagId", "relevance"],
    "genome-tags.csv": ["tagId", "tag"],
}


def validate_file_exists(path: Path) -> bool:
    """Comprueba que la ruta es un archivo regular."""
    path = Path(path)
    return path.exists() and path.is_file()


def validate_directory_exists(path: Path) -> bool:
    """Comprueba que la ruta es un directorio."""
    path = Path(path)
    return path.exists() and path.is_dir()


def validate_json_file(
    path: Path,
    expected_type: type | None = None,
) -> dict:
    """Valida la lectura JSON y, opcionalmente, su tipo superior.

    El esquema específico de TMDb/Watchmode debe comprobarse en su módulo.
    """
    path = Path(path)
    if not validate_file_exists(path):
        return {"valid": False, "error": "file_not_found"}

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, UnicodeError, OSError) as error:
        return {"valid": False, "error": str(error)}

    if expected_type is not None and not isinstance(data, expected_type):
        return {
            "valid": False,
            "error": f"expected_{expected_type.__name__}_got_{type(data).__name__}",
            "data_type": type(data).__name__,
        }

    return {"valid": True, "data_type": type(data).__name__}


def validate_csv_file(
    path: Path,
    expected_columns: list[str] | None = None,
    chunksize: int = 100_000,
) -> dict:
    """Lee un CSV en bloques para evitar cargarlo entero en memoria.

    Por defecto comprueba automáticamente el esquema de los seis CSV de
    MovieLens conocidos, y para otros CSV solo confirma que sean legibles.
    No valida integridad semántica de todas las filas ni checksums.
    """
    path = Path(path)
    if not validate_file_exists(path):
        return {"valid": False, "error": "file_not_found"}
    if chunksize < 1:
        raise ValueError("chunksize debe ser un entero positivo")

    if expected_columns is None:
        expected_columns = MOVIELENS_EXPECTED_COLUMNS.get(path.name)

    try:
        # Leer exclusivamente la cabecera para revisar el esquema antes de iterar.
        header = pd.read_csv(path, nrows=0)
        columns = list(header.columns)
        if not columns:
            return {"valid": False, "error": "empty_header"}

        missing = [column for column in (expected_columns or []) if column not in columns]
        if missing:
            return {
                "valid": False,
                "error": "missing_columns",
                "missing_columns": missing,
                "columns": len(columns),
            }

        rows = 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            rows += len(chunk)

        if rows == 0:
            return {"valid": False, "error": "empty_csv", "rows": 0, "columns": len(columns)}

        return {"valid": True, "rows": rows, "columns": len(columns)}
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError, OSError, ValueError) as error:
        return {"valid": False, "error": str(error)}


def validate_expected_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> dict:
    """Confirma que un DataFrame incluya todas las columnas indicadas."""
    missing_columns = [column for column in expected_columns if column not in df.columns]
    return {"valid": len(missing_columns) == 0, "missing_columns": missing_columns}
