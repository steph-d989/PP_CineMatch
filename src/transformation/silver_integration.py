import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_TMDB_DIR,
    SILVER_WATCHMODE_DIR,
    SILVER_INTEGRATED_DIR,
)


### FUNCION QUE CARGA LOS DATASETS NECESARIOS PARA INTEGRAR LAS APIS

def load_silver_datasets() -> dict[str, pd.DataFrame]:

    rutas = {
        "movies": (
            SILVER_MOVIELENS_DIR
            / "movies.parquet"
        ),
        "links": (
            SILVER_MOVIELENS_DIR
            / "links.parquet"
        ),
        "tmdb": (
            SILVER_TMDB_DIR
            / "movies_tmdb.parquet"
        ),
        "watchmode_status": (
            SILVER_WATCHMODE_DIR
            / "availability_status.parquet"
        ),
    }

    datasets = {}

    for nombre, ruta in rutas.items():

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró: {ruta}"
            )

        datasets[nombre] = pd.read_parquet(
            ruta
        )

    return datasets


### FUNCION QUE INTEGRA CATALOGO DE MOVIELENS CON LOS IDENTIFICADORES

def integrate_movielens(
    movies: pd.DataFrame,
    links: pd.DataFrame,
) -> pd.DataFrame:

    df = movies.merge(
        links,
        on="movie_id",
        how="left",
        validate="one_to_one",
    )

    return df


### FUNCION QUE ENRIQUECE MOVIELENS CON METADATA TMDB

def integrate_tmdb(
    movies: pd.DataFrame,
    tmdb: pd.DataFrame,
) -> pd.DataFrame:

    tmdb = tmdb.rename(
        columns={
            "title": "tmdb_title",
            "imdb_id": "tmdb_imdb_id",
            "genres": "tmdb_genres",
        }
    )

    duplicados_tmdb = tmdb[
        tmdb["tmdb_id"].duplicated(
            keep=False
        )
    ]

    if not duplicados_tmdb.empty:
        raise ValueError(
            "TMDb Silver contiene tmdb_id duplicados."
        )

    df = movies.merge(
        tmdb,
        on="tmdb_id",
        how="left",
        validate="many_to_one",
    )

    return df


### FUNCION QUE CREA UN TITULO CANONICO

def create_canonical_fields(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df["canonical_title"] = (
        df["tmdb_title"]
        .combine_first(df["title"])
    )

    return df


### FUNCION QUE AÑADE INFORMACION SOBRE DISPONIBILIDAD CONOCIDA EN WATCHMODE

def integrate_watchmode_status(
    movies: pd.DataFrame,
    status: pd.DataFrame,
) -> pd.DataFrame:

    status = status.drop_duplicates(
        subset=["tmdb_id"]
    )


    duplicados_watchmode = status[
        status["tmdb_id"].duplicated(
            keep=False
        )
    ]

    if not duplicados_watchmode.empty:
        raise ValueError(
            "Watchmode Silver contiene tmdb_id duplicados."
        )

    df = movies.merge(
        status,
        on="tmdb_id",
        how="left",
        validate="many_to_one",
    )

    df["has_availability"] = (
        df["has_availability"]
        .fillna(False)
        .astype(bool)
    )

    df["sources_count"] = (
        df["sources_count"]
        .fillna(0)
        .astype("int64")
    )

    return df


### FUNCION QUE INTEGRA EL ESTADO DE WATCHMODE

def add_watchmode_check_status(
    df: pd.DataFrame,
    watchmode_ids: set
) -> pd.DataFrame:

    df = df.copy()

    df["watchmode_checked"] = (
        df["tmdb_id"].isin(
            watchmode_ids
        )
    )

    return df


### FUNCION PRINCIPAL DE INTEGRACION CONSTRUYE UN CATALOGO INTEGRADO

def build_integrated_movies(
    datasets: dict[str, pd.DataFrame]
) -> pd.DataFrame:

    df = integrate_movielens(
        datasets["movies"],
        datasets["links"],
    )

    df = integrate_tmdb(
        df,
        datasets["tmdb"],
    )

    watchmode_status = (
        datasets["watchmode_status"]
    )

    watchmode_ids = set(
        watchmode_status[
            "tmdb_id"
        ].dropna().tolist()
    )

    df = integrate_watchmode_status(
        df,
        watchmode_status,
    )

    df = add_watchmode_check_status(
        df,
        watchmode_ids,
    )

    df = create_canonical_fields(
        df
    )

    return df


### FUNCION QUE GUARDA UN CATALOGO INTEGRADO SILVER

def save_integrated_movies(
    df: pd.DataFrame
) -> Path:

    SILVER_INTEGRATED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = (
        SILVER_INTEGRATED_DIR
        / "movies_integrated.parquet"
    )

    df.to_parquet(
        ruta,
        index=False,
    )

    return ruta


### FUNCION QIE GUARDA LOS METADATOS

def save_integrated_metadata(
    df: pd.DataFrame
) -> Path:

    metadata = {
        "dataset": "movies_integrated",
        "capa": "silver",
        "fecha_transformacion": (
            datetime.now().isoformat()
        ),
        "formato": "parquet",
        "filas": len(df),
        "columnas": len(df.columns),
        "con_tmdb": int(
            df["tmdb_title"]
            .notna()
            .sum()
        ),
        "watchmode_consultadas": int(
            df["watchmode_checked"]
            .sum()
        ),
        "con_disponibilidad": int(
            df["has_availability"]
            .sum()
        ),
        "estado": "success",
    }

    ruta = (
        SILVER_INTEGRATED_DIR
        / "_metadata.json"
    )

    with open(
        ruta,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    return ruta


### FUNCION QUE ACTUA DE ORQUESTADOR 

def run_silver_integration():

    print("=" * 60)
    print("INTEGRACIÓN SILVER CINEMATCH")
    print("=" * 60)

    datasets = load_silver_datasets()

    df = build_integrated_movies(
        datasets
    )

    ruta = save_integrated_movies(
        df
    )

    metadata = save_integrated_metadata(
        df
    )

    print(
        "Películas integradas:",
        len(df)
    )

    print(
        "Con TMDb:",
        df["tmdb_title"]
        .notna()
        .sum()
    )

    print(
        "Consultadas en Watchmode:",
        df["watchmode_checked"]
        .sum()
    )

    print(
        "Con disponibilidad:",
        df["has_availability"]
        .sum()
    )

    print(
        "Dataset:",
        ruta
    )

    print(
        "Metadata:",
        metadata
    )

    print("=" * 60)
    print("SILVER INTEGRATED COMPLETADO")
    print("=" * 60)

    return df


if __name__ == "__main__":
    run_silver_integration()