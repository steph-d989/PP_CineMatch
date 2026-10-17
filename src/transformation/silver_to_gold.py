import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_INTEGRATED_DIR,
    SILVER_WATCHMODE_DIR,
    GOLD_DIMENSIONS_DIR,
    GOLD_FACTS_DIR,
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
    
    availability_path = (
        SILVER_WATCHMODE_DIR
        / "availability.parquet"
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
        
    if not availability_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {availability_path}"
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
    
    availability = pd.read_parquet(
        availability_path
    )

    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "availability": availability,
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

### FUNCION QUE CONSTRUYE DIM_PLATFORM

def build_dim_platform(
    availability: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "source_id",
        "provider_name",
    ]

    columnas_existentes = [
        columna
        for columna in columnas
        if columna in availability.columns
    ]

    df = availability[
        columnas_existentes
    ].copy()

    # Eliminar registros sin nombre de plataforma
    df = df.dropna(
        subset=["provider_name"]
    )

    # Eliminar duplicados de proveedor
    df = df.drop_duplicates(
        subset=[
            "source_id",
            "provider_name",
        ]
    )

    # Renombrar para mantener convención dimensional
    df = df.rename(
        columns={
            "source_id": "provider_id"
        }
    )

    # Ordenar para consistencia
    df = df.sort_values(
        by="provider_name"
    ).reset_index(drop=True)

    return df


### FUNCION QUE CONTRUYE  DIM_DATE

def build_dim_date(
    ratings: pd.DataFrame
) -> pd.DataFrame:

    fecha_min = (
        pd.to_datetime(
            ratings["rated_at"],
            utc=True,
        )
        .min()
        .normalize()
        .tz_localize(None)
    )

    fecha_actual = (
        pd.Timestamp.now()
        .normalize()
        .tz_localize(None)
    )

    fecha_max = (
        fecha_actual
        + pd.DateOffset(years=5)
    )

    fechas = pd.date_range(
        start=fecha_min,
        end=fecha_max,
        freq="D",
    )

    df = pd.DataFrame({
        "full_date": fechas
    })

    df["date_key"] = (
        df["full_date"]
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    df["day"] = (
        df["full_date"].dt.day
    )

    df["month"] = (
        df["full_date"].dt.month
    )

    df["month_name"] = (
        df["full_date"].dt.month_name()
    )

    df["quarter"] = (
        df["full_date"].dt.quarter
    )

    df["year"] = (
        df["full_date"].dt.year
    )

    df["day_of_week"] = (
        df["full_date"].dt.dayofweek
    )

    df["day_name"] = (
        df["full_date"].dt.day_name()
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    )

    return df[
        [
            "date_key",
            "full_date",
            "day",
            "month",
            "month_name",
            "quarter",
            "year",
            "day_of_week",
            "day_name",
            "is_weekend",
        ]
    ]
    
### FUNCION QUE CONSTRUYE FACT_RATING

def build_fact_rating(
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

    df = df.dropna(
        subset=[
            "user_id",
            "movie_id",
            "rating",
            "rated_at",
        ]
    )

    # Crear date_key compatible con dim_date
    df["date_key"] = (
        pd.to_datetime(
            df["rated_at"],
            utc=True,
        )
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    # Reordenar columnas
    columnas = [
        "user_id",
        "movie_id",
        "date_key",
        "rating",
        "rated_at",
    ]

    df = df[columnas]

    return df


### FUNCION QUE CONSTRUYE FACT_AVAILABILITY

def build_fact_availability(
    availability: pd.DataFrame,
    dim_movie: pd.DataFrame,
    dim_platform: pd.DataFrame,
) -> pd.DataFrame:

    df = availability.copy()

    # Eliminar registros sin claves mínimas
    df = df.dropna(
        subset=[
            "tmdb_id",
            "provider_name",
        ]
    )

    # Relacionar con dim_movie mediante tmdb_id
    movie_lookup = (
        dim_movie[
            [
                "movie_id",
                "tmdb_id",
            ]
        ]
        .dropna(subset=["tmdb_id"])
        .drop_duplicates(subset=["tmdb_id"])
    )

    df = df.merge(
        movie_lookup,
        on="tmdb_id",
        how="left",
        validate="many_to_one",
    )

    # Relacionar con dim_platform
    platform_lookup = (
        dim_platform[
            [
                "provider_id",
                "provider_name",
            ]
        ]
        .drop_duplicates(
            subset=[
                "provider_id",
                "provider_name",
            ]
        )
    )

    df = df.rename(
        columns={
            "source_id": "provider_id",
        }
    )

    df = df.merge(
        platform_lookup,
        on=[
            "provider_id",
            "provider_name",
        ],
        how="left",
        validate="many_to_one",
    )

    # Fecha de referencia:
    # por ahora usamos la fecha de ejecución del pipeline
    today = pd.Timestamp.now().normalize()

    df["date_key"] = int(
        today.strftime("%Y%m%d")
    )

    # Estandarizar nombre de tipo de acceso
    df = df.rename(
        columns={
            "type": "access_type",
        }
    )

    columnas = [
        "movie_id",
        "tmdb_id",
        "provider_id",
        "provider_name",
        "date_key",
        "region",
        "access_type",
        "format",
        "price",
        "web_url",
        "ios_url",
        "android_url",
    ]

    columnas_existentes = [
        columna
        for columna in columnas
        if columna in df.columns
    ]

    df = df[
        columnas_existentes
    ].copy()

    # Evitar duplicados exactos de disponibilidad
    columnas_clave = [
        columna
        for columna in [
            "movie_id",
            "provider_id",
            "region",
            "access_type",
        ]
        if columna in df.columns
    ]

    if columnas_clave:
        df = df.drop_duplicates(
            subset=columnas_clave
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
    
    dim_platform = build_dim_platform(
        silver["availability"]
    )
    
    dim_date = build_dim_date(
        silver["ratings"]
    )
    
    # FACTS
    
    print("Construyendo facts...")

    fact_rating = build_fact_rating(
        silver["ratings"]
    )
    
    fact_availability = build_fact_availability(
        silver["availability"],
        dim_movie,
        dim_platform,
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

    rutas["dim_platform"] = (
        save_gold_dataset(
            dim_platform,
            GOLD_DIMENSIONS_DIR,
            "dim_platform.parquet",
        )
    )

    datasets["dim_platform"] = ( 
        dim_platform
    ) 

    rutas["dim_date"] = (
        save_gold_dataset(
            dim_date,
            GOLD_DIMENSIONS_DIR,
            "dim_date.parquet",
        )
    )

    datasets["dim_date"] = (
        dim_date
    )
    
    # FACTS
    
    rutas["fact_rating"] = (
        save_gold_dataset(
            fact_rating,
            GOLD_FACTS_DIR,
            "fact_rating.parquet",
        )
    )

    datasets["fact_rating"] = (
        fact_rating
    )
    
    rutas["fact_availability"] = (
        save_gold_dataset(
            fact_availability,
            GOLD_FACTS_DIR,
            "fact_availability.parquet",
        )
    )

    datasets["fact_availability"] = (
        fact_availability
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