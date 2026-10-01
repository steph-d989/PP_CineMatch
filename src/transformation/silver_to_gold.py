import json
from datetime import datetime

import pandas as pd

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_INTEGRATED_DIR,
    GOLD_ANALYTICS_DIR,
    GOLD_ML_DIR,
)


### FUNCION QUE CARGA DATASETS SILVER NECESARIOS PARA CONSTRUIR GOLD

def load_silver_for_gold() -> dict[str, pd.DataFrame]:

    movies = pd.read_parquet(
        SILVER_INTEGRATED_DIR
        / "movies_integrated.parquet"
    )

    ratings = pd.read_parquet(
        SILVER_MOVIELENS_DIR
        / "ratings"
    )

    tags = pd.read_parquet(
        SILVER_MOVIELENS_DIR
        / "tags.parquet"
    )

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
    }
    

### FUNCION GOLD ANALITICO DE PELICULAS

def build_movies_analytics(
    movies: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "movie_id",
        "tmdb_id",
        "imdb_id",
        "canonical_title",
        "genres",
        "tmdb_genres",
        "release_date",
        "runtime",
        "original_language",
        "popularity",
        "vote_average",
        "vote_count",
        "director",
        "actors",
        "keywords",
        "has_availability",
        "watchmode_checked",
        "sources_count",
    ]

    columnas_existentes = [
        c
        for c in columnas
        if c in movies.columns
    ]

    df = movies[
        columnas_existentes
    ].copy()

    return df


## FUNCION DEL GOLD ANALITICO DE INTERACCIONES 

def build_user_movie_interactions(
    ratings: pd.DataFrame
) -> pd.DataFrame:

    df = ratings[
        [
            "user_id",
            "movie_id",
            "rating",
            "rated_at",
        ]
    ].copy()

    return df


### FUNCION GOLD DE METRICAS POR PELICULA

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


### FUNCION GOLD DE METRICAS POR USUARIO

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


### FUNCION QUE GUARDA GOLD

def save_gold_dataset(
    df: pd.DataFrame,
    directory,
    filename: str,
):
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


### FUNCION QUE CONSTRUYE EL DATASET GOLD PARA EL MODELO 

def build_content_based_features(
    movies: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el dataset Gold para
    el modelo basado en contenido.
    """

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

    return df


### FUNCION QUE CONSTRUYE DATASET PARA EL FILTRADO COLABORATIVO

def build_collaborative_interactions(
    ratings: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye el dataset Gold para
    filtrado colaborativo.
    """

    df = ratings[
        [
            "user_id",
            "movie_id",
            "rating",
            "rated_at",
        ]
    ].copy()

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "movie_id",
            "rated_at",
        ]
    )

    df = df.dropna(
        subset=[
            "user_id",
            "movie_id",
            "rating",
        ]
    )

    return df


### FUNCION PRINCIPAL 

def run_silver_to_gold():

    print("=" * 60)
    print("SILVER -> GOLD CINEMATCH")
    print("=" * 60)

    silver = load_silver_for_gold()

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
    
    movies_analytics = (
        build_movies_analytics(
            silver["movies"]
        )
    )

    interactions = (
        build_user_movie_interactions(
            silver["ratings"]
        )
    )

    movie_metrics = (
        build_movie_metrics(
            silver["ratings"]
        )
    )

    user_metrics = (
        build_user_metrics(
            silver["ratings"]
        )
    )

    rutas = {}

    rutas["movies_analytics"] = (
        save_gold_dataset(
            movies_analytics,
            GOLD_ANALYTICS_DIR,
            "movies_analytics.parquet",
        )
    )

    rutas["movie_metrics"] = (
        save_gold_dataset(
            movie_metrics,
            GOLD_ANALYTICS_DIR,
            "movie_metrics.parquet",
        )
    )

    rutas["user_metrics"] = (
        save_gold_dataset(
            user_metrics,
            GOLD_ANALYTICS_DIR,
            "user_metrics.parquet",
        )
    )

    rutas["interactions"] = (
        save_gold_dataset(
            interactions,
            GOLD_ML_DIR,
            "user_movie_interactions.parquet",
        )
    )
    
    rutas["content_based_features"] = (
    save_gold_dataset(
        content_based_features,
        GOLD_ML_DIR,
        "content_based_features.parquet",
    )
)

    rutas["collaborative_interactions"] = (
        save_gold_dataset(
            collaborative_interactions,
            GOLD_ML_DIR,
            "collaborative_interactions.parquet",
        )
    )

    print("\nDatasets Gold generados:")

    for nombre, ruta in rutas.items():
        print(
            nombre,
            "->",
            ruta
        )

    print("=" * 60)
    print("GOLD COMPLETADO")
    print("=" * 60)

    return rutas



if __name__ == "__main__":
    run_silver_to_gold()