from pathlib import Path

import pandas as pd

from src.utils.paths import (
    GOLD_DIMENSIONS_DIR,
    GOLD_FACTS_DIR,
    GOLD_ML_CONTENT_DIR,
    GOLD_ML_COLLABORATIVE_DIR,
    GOLD_ML_HYBRID_DIR,
)

### FUNCION QUE CARGA LOS DATASETS DE GOLD ANALITICS

def load_gold_analytics() -> dict[str, pd.DataFrame]:

    dim_movie_path = (
        GOLD_DIMENSIONS_DIR
        / "dim_movie.parquet"
    )

    fact_rating_path = (
        GOLD_FACTS_DIR
        / "fact_rating.parquet"
    )

    if not dim_movie_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {dim_movie_path}"
        )

    if not fact_rating_path.exists():
        raise FileNotFoundError(
            f"No se encontró: {fact_rating_path}"
        )

    dim_movie = pd.read_parquet(
        dim_movie_path
    )

    fact_rating = pd.read_parquet(
        fact_rating_path
    )

    return {
        "dim_movie": dim_movie,
        "fact_rating": fact_rating,
    }

### FUNCION QUE GENERA DATASETS BASE PARA EL RECOEMNDADOR CONTENT-BASED GOLD ANALYTICS -> DIM_MOVIE

def build_content_based_features(
    dim_movie: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "movie_id",
        "tmdb_id",
        "title",
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
        if columna in dim_movie.columns
    ]

    df = dim_movie[
        columnas_existentes
    ].copy()

    df = df.drop_duplicates(
        subset=["movie_id"]
    )

    df = df.dropna(
        subset=["movie_id"]
    )

    return df

### FUNCION QUE GENERA DATASET UTILIADO POR FILTRADO COLABORATIVO

def build_collaborative_interactions(
    fact_rating: pd.DataFrame
) -> pd.DataFrame:

    columnas = [
        "user_id",
        "movie_id",
        "rating",
        "rated_at",
    ]

    df = fact_rating[
        columnas
    ].copy()

    df = df.dropna(
        subset=[
            "user_id",
            "movie_id",
            "rating",
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "user_id",
            "movie_id",
            "rated_at",
        ]
    )

    return df

### FUNCION QUE GUARDA EL DATASET PREPARADO PARA ML

def save_ml_dataset(
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

### FUNCION QUE GUARDA LOS DATOS EN MULTIPLES ARCHIVOS PARQUET

def save_partitioned_parquet(
    df: pd.DataFrame,
    directory: Path,
    folder_name: str,
    rows_per_file: int = 2_000_000,
) -> list[Path]:

    target_dir = directory / folder_name

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas = []

    total_rows = len(df)

    for i, start in enumerate(
        range(
            0,
            total_rows,
            rows_per_file,
        )
    ):
        end = min(
            start + rows_per_file,
            total_rows,
        )

        chunk = df.iloc[start:end]

        ruta = (
            target_dir
            / f"part_{i:03d}.parquet"
        )

        chunk.to_parquet(
            ruta,
            index=False,
        )

        rutas.append(ruta)

        print(
            f"{folder_name}/"
            f"{ruta.name} -> "
            f"{len(chunk):,} filas"
        )

    return rutas

### FUNCION QUE ORQUESTA

def run_gold_analytics_to_ml():

    print("=" * 60)
    print("GOLD ANALYTICS -> GOLD ML")
    print("=" * 60)

    analytics = load_gold_analytics()

    print(
        "Películas:",
        len(analytics["dim_movie"])
    )

    print(
        "Ratings:",
        len(analytics["fact_rating"])
    )

    # CONTENT BASED

    print(
        "\nConstruyendo dataset Content-Based..."
    )

    content_based_features = (
        build_content_based_features(
            analytics["dim_movie"]
        )
    )

    # COLLABORATIVE

    print(
        "Construyendo dataset Collaborative..."
    )

    collaborative_interactions = (
        build_collaborative_interactions(
            analytics["fact_rating"]
        )
    )

    # GUARDADO

    rutas = {}

    rutas["content_based_features"] = (
        save_ml_dataset(
            content_based_features,
            GOLD_ML_CONTENT_DIR,
            "content_based_features.parquet",
        )
    )

    rutas["collaborative_interactions"] = (
        save_partitioned_parquet(
            collaborative_interactions,
            GOLD_ML_COLLABORATIVE_DIR,
            "collaborative_interactions",
        )
    )

    rutas[
        "collaborative_interactions"
    ] = (
        save_ml_dataset(
            collaborative_interactions,
            GOLD_ML_COLLABORATIVE_DIR,
            "collaborative_interactions.parquet",
        )
    )

    print("\nDatasets ML generados:")

    for nombre, ruta in rutas.items():

        print(
            nombre,
            "->",
            ruta
        )

    print("\n" + "=" * 60)
    print("GOLD ML COMPLETADO")
    print("=" * 60)

    return rutas


if __name__ == "__main__":
    run_gold_analytics_to_ml()