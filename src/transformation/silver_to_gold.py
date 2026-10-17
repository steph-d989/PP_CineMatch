import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_INTEGRATED_DIR,
    GOLD_DIMENSIONS_DIR,
    GOLD_METRICS_DIR,
    GOLD_ML_CONTENT_DIR,
    GOLD_ML_COLLABORATIVE_DIR,
)


#FUNCION QUE CARGA LOS DATOS DE SILVER

def load_silver_for_gold() -> dict[str, pd.DataFrame]:

    movies_path = (
        SILVER_INTEGRATED_DIR
        / "movies_integrated.parquet"
    )

    ratings_path = (
        SILVER_MOVIELENS_DIR
        / "ratings"
    )

    tags_path = (
        SILVER_MOVIELENS_DIR
        / "tags.parquet"
    )

    if not movies_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {movies_path}"
        )

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {ratings_path}"
        )

    if not tags_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {tags_path}"
        )

    movies = pd.read_parquet(
        movies_path
    )

    # ratings está particionado en varios parquet
    ratings = pd.read_parquet(
        ratings_path
    )

    tags = pd.read_parquet(
        tags_path
    )

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
    }


### FUNCION QUE CONSTRUYE DIM_MOVIE

def build_dim_movie(
    movies: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "canonical_title",
        "original_title",
        "overview",
        "release_date",
        "runtime",
        "original_language",
        "popularity",
        "vote_average",
        "vote_count",
        "poster_path",
        "backdrop_path",
        "genres",
        "tmdb_genres",
        "director",
        "actors",
        "keywords",
    ]

    columnas_existentes = [
        columna
        for columna in columnas
        if columna in movies.columns
    ]

    df = movies[
        columnas_existentes
    ].copy()

    # Una película por movie_id
    df = df.drop_duplicates(
        subset=["movie_id"]
    )

    # Nombre final más claro
    if "canonical_title" in df.columns:
        df = df.rename(
            columns={
                "canonical_title": "title"
            }
        )

    # Eliminar filas sin movie_id
    df = df.dropna(
        subset=["movie_id"]
    )

    return df


### FUNCION QUE CONSTRUYE LAS METRICAS POR PELICULA

def build_movie_metrics(
    ratings: pd.DataFrame
) -> pd.DataFrame:

    metrics = (
        ratings
        .groupby("movie_id")
        .agg(
            rating_count=(
                "rating",
                "count"
            ),
            rating_mean=(
                "rating",
                "mean"
            ),
            rating_std=(
                "rating",
                "std"
            ),
        )
        .reset_index()
    )

    return metrics

### FUNCION QUE CONSTRUYE METRICAS POR USUARIO

def build_user_metrics(
    ratings: pd.DataFrame
) -> pd.DataFrame:

    metrics = (
        ratings
        .groupby("user_id")
        .agg(
            ratings_count=(
                "rating",
                "count"
            ),
            rating_mean=(
                "rating",
                "mean"
            ),
            rating_std=(
                "rating",
                "std"
            ),
        )
        .reset_index()
    )

    return metrics


### FUNCION QUE CONSTRUYE DATASET GOLD PARA SER UTILIZADO POREL MODELO

def build_content_based_features(
    movies: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "movie_id",
        "tmdb_id",
        "canonical_title",
        "genres",
        "tmdb_genres",
        "keywords",
        "director",
        "actors",
        "overview",
        "original_language",
        "release_date",
    ]

    columnas_existentes = [
        columna
        for columna in columnas
        if columna in movies.columns
    ]

    df = movies[
        columnas_existentes
    ].copy()

    df = df.drop_duplicates(
        subset=["movie_id"]
    )

    df = df.dropna(
        subset=["movie_id"]
    )

    return df

### FUNCION QUE CONSTRUYE EL DATASET GOLD TILIZADO POR FILTRADO COLABORATIVO

def build_collaborative_interactions(
    ratings: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "user_id",
        "movie_id",
        "rating",
        "rated_at",
    ]

    df = ratings[
        columnas
    ].copy()

    # Eliminar registros completamente duplicados
    df = df.drop_duplicates(
        subset=[
            "user_id",
            "movie_id",
            "rated_at",
        ]
    )

    # Variables esenciales
    df = df.dropna(
        subset=[
            "user_id",
            "movie_id",
            "rating",
        ]
    )

    return df


### FUNCION QUE GUARDA UN DATAFRAME GOLD EN PARQUET

def save_gold_dataset(
    df: pd.DataFrame,
    directory: Path,
    filename: str,
) -> Path:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = (
        directory
        / filename
    )

    df.to_parquet(
        ruta,
        index=False,
    )

    return ruta


### FUNCION QUE GUARDA LA METADATA GENERAL DE LA CONSTRUCCION GOLD

def save_gold_metadata(
    rutas: dict[str, Path],
    datasets: dict[str, pd.DataFrame],
) -> Path:

    metadata = {
        "capa": "gold",
        "fecha_transformacion": (
            datetime.now().isoformat()
        ),
        "estado": "success",
        "datasets": {},
    }

    for nombre, df in datasets.items():

        metadata["datasets"][nombre] = {
            "filas": len(df),
            "columnas": len(df.columns),
            "ruta": str(
                rutas[nombre]
            ),
        }

    # Guardamos metadata en metrics
    GOLD_METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_metadata = (
        GOLD_METRICS_DIR
        / "_metadata.json"
    )

    with open(
        ruta_metadata,
        "w",
        encoding="utf-8",
    ) as archivo:

        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    return ruta_metadata

### FUNCION QUE EJECUTA ORQUESTADOR DE SILVER A GOLD

def run_silver_to_gold() -> dict:


    print("=" * 60)
    print("SILVER -> GOLD CINEMATCH")
    print("=" * 60)

    # Cargar Silver

    silver = load_silver_for_gold()

    print(
        "Películas Silver:",
        len(silver["movies"])
    )

    print(
        "Ratings Silver:",
        len(silver["ratings"])
    )

    # DIMENSIONES

    print("\nConstruyendo dimensiones...")

    dim_movie = build_dim_movie(
        silver["movies"]
    )

    # MÉTRICAS

    print(
        "Construyendo métricas analíticas..."
    )

    movie_metrics = build_movie_metrics(
        silver["ratings"]
    )

    user_metrics = build_user_metrics(
        silver["ratings"]
    )

    # MACHINE LEARNING

    print(
        "Construyendo datasets ML..."
    )

    content_based_features = (
        build_content_based_features(
            silver["movies"]
        )
    )

    collaborative_interactions = (
        build_collaborative_interactions(
            silver["ratings"]
        )
    )

    # GUARDAR DATASETS

    rutas = {}

    datasets = {}

    # DIMENSIONS

    rutas["dim_movie"] = (
        save_gold_dataset(
            dim_movie,
            GOLD_DIMENSIONS_DIR,
            "dim_movie.parquet",
        )
    )

    datasets["dim_movie"] = (
        dim_movie
    )

    # METRICS

    rutas["movie_metrics"] = (
        save_gold_dataset(
            movie_metrics,
            GOLD_METRICS_DIR,
            "movie_metrics.parquet",
        )
    )

    datasets["movie_metrics"] = (
        movie_metrics
    )

    rutas["user_metrics"] = (
        save_gold_dataset(
            user_metrics,
            GOLD_METRICS_DIR,
            "user_metrics.parquet",
        )
    )

    datasets["user_metrics"] = (
        user_metrics
    )

    # ML - CONTENT BASED

    rutas["content_based_features"] = (
        save_gold_dataset(
            content_based_features,
            GOLD_ML_CONTENT_DIR,
            "content_based_features.parquet",
        )
    )

    datasets[
        "content_based_features"
    ] = content_based_features

    # ML - COLLABORATIVE

    rutas[
        "collaborative_interactions"
    ] = (
        save_gold_dataset(
            collaborative_interactions,
            GOLD_ML_COLLABORATIVE_DIR,
            "collaborative_interactions.parquet",
        )
    )

    datasets[
        "collaborative_interactions"
    ] = collaborative_interactions

    # METADATA

    metadata = save_gold_metadata(
        rutas,
        datasets,
    )

    # RESUMEN

    print(
        "\nDatasets Gold generados:"
    )

    for nombre, ruta in rutas.items():

        print(
            nombre,
            "->",
            ruta
        )

    print(
        "\nMetadata Gold:",
        metadata
    )

    print("\n" + "=" * 60)
    print("GOLD COMPLETADO")
    print("=" * 60)

    return {
        "rutas": rutas,
        "metadata": metadata,
    }

### FUNCION QUE EJECUTA 

if __name__ == "__main__":
    run_silver_to_gold()