import pandas as pd

from src.utils.paths import (
    GOLD_DIMENSIONS_DIR,
    GOLD_FACTS_DIR,
)

dim_movie = pd.read_parquet(
    GOLD_DIMENSIONS_DIR / "dim_movie.parquet"
)

dim_platform = pd.read_parquet(
    GOLD_DIMENSIONS_DIR / "dim_platform.parquet"
)

dim_date = pd.read_parquet(
    GOLD_DIMENSIONS_DIR / "dim_date.parquet"
)

fact_rating = pd.read_parquet(
    GOLD_FACTS_DIR / "fact_rating.parquet"
)

fact_availability = pd.read_parquet(
    GOLD_FACTS_DIR / "fact_availability.parquet"
)

### VERIFICAR DIM Y FACTS
print("dim_movie:", dim_movie.shape)
print("dim_platform:", dim_platform.shape)
print("dim_date:", dim_date.shape)
print("fact_rating:", fact_rating.shape)
print("fact_availability:", fact_availability.shape)

### REVISAR COLUMNAS
print("\ndim_movie")
print(dim_movie.columns.tolist())

print("\ndim_platform")
print(dim_platform.columns.tolist())

print("\ndim_date")
print(dim_date.columns.tolist())

print("\nfact_rating")
print(fact_rating.columns.tolist())

print("\nfact_availability")
print(fact_availability.columns.tolist())

### VERIFICAR RELACION FACT_RATING -> DIM_MOVIE
movies_fact = set(
    fact_rating["movie_id"]
    .dropna()
    .unique()
)

movies_dim = set(
    dim_movie["movie_id"]
    .dropna()
    .unique()
)

movies_sin_dim = (
    movies_fact - movies_dim
)

print(
    "movie_id en fact_rating sin dim_movie:",
    len(movies_sin_dim)
)

### VERIFICAR FACT_RATING -> DIM_DATE
dates_fact = set(
    fact_rating["date_key"]
    .dropna()
    .unique()
)

dates_dim = set(
    dim_date["date_key"]
    .dropna()
    .unique()
)

dates_sin_dim = (
    dates_fact - dates_dim
)

print(
    "date_key de ratings sin dim_date:",
    len(dates_sin_dim)
)

### VERIFICAR FACT_AVAILABILITY -> DIM_MOVIE
availability_movies = set(
    fact_availability["movie_id"]
    .dropna()
    .unique()
)

movies_sin_dim = (
    availability_movies
    - movies_dim
)

print(
    "Películas de disponibilidad sin dim_movie:",
    len(movies_sin_dim)
)

### VERIFICAR PLATAFORMAS
fact_platforms = (
    fact_availability[
        ["provider_id", "provider_name"]
    ]
    .drop_duplicates()
)

dim_platforms = (
    dim_platform[
        ["provider_id", "provider_name"]
    ]
    .drop_duplicates()
)

platform_check = fact_platforms.merge(
    dim_platforms,
    on=[
        "provider_id",
        "provider_name",
    ],
    how="left",
    indicator=True,
)

print(
    platform_check["_merge"]
    .value_counts()
)

### COMPROBACION IMPORTANTE: FACT_AVAILABILITY → DIM_DATE
availability_dates = set(
    fact_availability["date_key"]
    .dropna()
    .unique()
)

dates_sin_dim = (
    availability_dates
    - dates_dim
)

print(
    "date_key de disponibilidad sin dim_date:",
    dates_sin_dim
)