import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils.paths import (
    BRONZE_MOVIELENS_DIR,
    BRONZE_TMDB_DIR,
    BRONZE_WATCHMODE_DIR,
    SILVER_MOVIELENS_DIR,
    SILVER_TMDB_DIR,
    SILVER_WATCHMODE_DIR,
)


### FUNCION QUE CARGA LOS CSV DE MOVIELENS DESDE BRONZE

def load_movielens_bronze() -> dict[str, pd.DataFrame]:

    source_dir = (
        BRONZE_MOVIELENS_DIR
        / "ml-32m"
    )

    archivos = {
        "movies": "movies.csv",
        "ratings": "ratings.csv",
        "tags": "tags.csv",
        "links": "links.csv",
    }

    datasets = {}

    for nombre, archivo in archivos.items():

        ruta = source_dir / archivo

        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró: {ruta}"
            )

        datasets[nombre] = pd.read_csv(ruta)

    return datasets


### FUNCION QUE LIMPIA Y ESTANDARIZA MOVIES.CSV PARA SILVER

def transform_movies(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Eliminar duplicados
    df = df.drop_duplicates(
        subset=["movieId"]
    )

    # Tipos
    df["movieId"] = (
        pd.to_numeric(
            df["movieId"],
            errors="raise"
        )
        .astype("int64")
    )

    df["title"] = (
        df["title"]
        .astype("string")
        .str.strip()
    )

    df["genres"] = (
        df["genres"]
        .astype("string")
        .str.strip()
    )

    # Normalizar nombres
    df = df.rename(
        columns={
            "movieId": "movie_id",
        }
    )

    return df


### FUNCION QUE LIMPIA Y ESTANDARIZA RATINGS.CSV

def transform_ratings(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Eliminar duplicados exactos
    df = df.drop_duplicates()

    # Tipos
    df["userId"] = pd.to_numeric(
        df["userId"],
        errors="raise"
    ).astype("int64")

    df["movieId"] = pd.to_numeric(
        df["movieId"],
        errors="raise"
    ).astype("int64")

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Eliminar ratings inválidos
    df = df[
        df["rating"].between(
            0.5,
            5.0
        )
    ]

    # Unix timestamp → datetime UTC
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    df = df.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
            "timestamp": "rated_at",
        }
    )

    return df


### FUNCION QUE LIMPIA Y ESTADARIZA TAGS.CSV

def transform_tags(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df = df.drop_duplicates()

    df = df.dropna(
        subset=["tag"]
    )

    df["userId"] = pd.to_numeric(
        df["userId"],
        errors="raise"
    ).astype("int64")

    df["movieId"] = pd.to_numeric(
        df["movieId"],
        errors="raise"
    ).astype("int64")

    df["tag"] = (
        df["tag"]
        .astype("string")
        .str.strip()
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    df = df.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
            "timestamp": "tagged_at",
        }
    )

    return df


### FUNCION QUE LIMPIA Y ESTADARIZA LINKS.CSV

def transform_links(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    df = df.drop_duplicates(
        subset=["movieId"]
    )

    df["movieId"] = pd.to_numeric(
        df["movieId"],
        errors="raise"
    ).astype("int64")

    df["tmdbId"] = pd.to_numeric(
        df["tmdbId"],
        errors="coerce"
    ).astype("Int64")

    # IMDb se trata como identificador, no variable numérica.
    df["imdbId"] = (
        df["imdbId"]
        .astype("Int64")
        .astype("string")
    )

    df = df.rename(
        columns={
            "movieId": "movie_id",
            "imdbId": "imdb_id",
            "tmdbId": "tmdb_id",
        }
    )

    return df


### FUNCION QUE GUARDA LOS DATASETS SILVER EN FORMATO PARQUET

def save_movielens_silver(
    datasets: dict[str, pd.DataFrame]
) -> dict:

    SILVER_MOVIELENS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas = {}

    for nombre in [
        "movies",
        "tags",
        "links",
    ]:

        df = datasets[nombre]

        ruta = (
            SILVER_MOVIELENS_DIR
            / f"{nombre}.parquet"
        )

        df.to_parquet(
            ruta,
            index=False,
        )

        rutas[nombre] = ruta

    rutas["ratings"] = (
        save_ratings_partitioned(
            datasets["ratings"]
        )
    )

    return rutas


### FUNCION QUE GUARDA LA METADATA SILVER 

def save_movielens_silver_metadata(
    datasets: dict[str, pd.DataFrame]
) -> Path:

    metadata = {
        "fuente": "MovieLens",
        "capa": "silver",
        "fecha_transformacion": (
            datetime.now().isoformat()
        ),
        "formato": "parquet",
        "datasets": {
            nombre: {
                "filas": len(df),
                "columnas": len(df.columns),
            }
            for nombre, df
            in datasets.items()
        },
        "estado": "success",
    }

    ruta_metadata = (
        SILVER_MOVIELENS_DIR
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

### FUNCION QUE GUARDA LOS RATINGS PARTICIONADOS

def save_ratings_partitioned(
    df: pd.DataFrame,
    rows_per_file: int = 2_000_000,
) -> list[Path]:

    ratings_dir = (
        SILVER_MOVIELENS_DIR
        / "ratings"
    )

    ratings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas = []

    total_rows = len(df)

    for i, start in enumerate(
        range(
            0,
            total_rows,
            rows_per_file
        )
    ):

        end = min(
            start + rows_per_file,
            total_rows
        )

        chunk = df.iloc[
            start:end
        ]

        ruta = (
            ratings_dir
            / f"part_{i:03d}.parquet"
        )

        chunk.to_parquet(
            ruta,
            index=False,
        )

        rutas.append(ruta)

        print(
            f"{ruta.name} -> "
            f"{len(chunk):,} filas"
        )

    return rutas


### FUNCION QUE EJECUTA TRANSFORMACION BRONZE A SILVER

def run_movielens_bronze_to_silver():

    print("=" * 50)
    print("MOVIELENS: BRONZE -> SILVER")
    print("=" * 50)

    bronze = load_movielens_bronze()

    silver = {
        "movies": transform_movies(
            bronze["movies"]
        ),
        "ratings": transform_ratings(
            bronze["ratings"]
        ),
        "tags": transform_tags(
            bronze["tags"]
        ),
        "links": transform_links(
            bronze["links"]
        ),
    }

    rutas = save_movielens_silver(
        silver
    )

    metadata = (
        save_movielens_silver_metadata(
            silver
        )
    )

    print("\nDatasets generados:")

    for nombre, ruta in rutas.items():
        print(
            nombre,
            "->",
            ruta
        )

    print(
        "\nMetadata:",
        metadata
    )

    print("=" * 50)
    print("MOVIELENS SILVER COMPLETADO")
    print("=" * 50)

    return silver


### FUNCION QUE CARGA LOS JSON DE PELICULAS TMDB ALMACENADAS A BRONZE

def load_tmdb_bronze() -> list[dict]:

    archivos = list(
        BRONZE_TMDB_DIR.glob("movie_*.json")
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron JSON TMDb en {BRONZE_TMDB_DIR}"
        )

    registros = []

    for archivo in archivos:

        try:
            with open(
                archivo,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            if isinstance(data, dict):
                registros.append(data)

        except Exception as error:
            print(
                f"Error leyendo {archivo.name}: {error}"
            )

    return registros


### FUNCION QUE NORMALIZA LOS JSON TMDB Y GENERA UN DF LIMPIO PARA SILVER

def transform_tmdb_movies(
    registros: list[dict]
) -> pd.DataFrame:

    filas = []

    for pelicula in registros:

        genres = [
            genero.get("name")
            for genero in pelicula.get(
                "genres",
                []
            )
            if genero.get("name")
        ]

        production_companies = [
            company.get("name")
            for company in pelicula.get(
                "production_companies",
                []
            )
            if company.get("name")
        ]

        credits = pelicula.get(
            "credits",
            {}
        )

        cast = credits.get(
            "cast",
            []
        )

        crew = credits.get(
            "crew",
            []
        )

        # Tomamos principales actores
        actors = [
            actor.get("name")
            for actor in cast[:5]
            if actor.get("name")
        ]

        # Buscar director
        director = next(
            (
                person.get("name")
                for person in crew
                if person.get("job") == "Director"
            ),
            None,
        )

        keywords_data = pelicula.get(
            "keywords",
            {}
        )

        keywords_list = (
            keywords_data.get("keywords")
            or keywords_data.get("results")
            or []
        )

        keywords = [
            keyword.get("name")
            for keyword in keywords_list
            if keyword.get("name")
        ]

        filas.append({
            "tmdb_id": pelicula.get("id"),
            "imdb_id": pelicula.get("imdb_id"),
            "title": pelicula.get("title"),
            "original_title": pelicula.get(
                "original_title"
            ),
            "overview": pelicula.get("overview"),
            "release_date": pelicula.get(
                "release_date"
            ),
            "runtime": pelicula.get("runtime"),
            "original_language": pelicula.get(
                "original_language"
            ),
            "popularity": pelicula.get(
                "popularity"
            ),
            "vote_average": pelicula.get(
                "vote_average"
            ),
            "vote_count": pelicula.get(
                "vote_count"
            ),
            "poster_path": pelicula.get(
                "poster_path"
            ),
            "backdrop_path": pelicula.get(
                "backdrop_path"
            ),
            "genres": genres,
            "director": director,
            "actors": actors,
            "keywords": keywords,
            "production_companies": (
                production_companies
            ),
        })

    df = pd.DataFrame(filas)

    return df


### FUNCION QUE LIMPIA Y TIPA EL DF TMDB

def clean_tmdb_movies(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    # Duplicados
    df = df.drop_duplicates(
        subset=["tmdb_id"]
    )

    # ID
    df["tmdb_id"] = pd.to_numeric(
        df["tmdb_id"],
        errors="coerce"
    ).astype("Int64")

    # Fecha
    df["release_date"] = pd.to_datetime(
        df["release_date"],
        errors="coerce",
    )

    # Numéricos
    columnas_numericas = [
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
    ]

    for columna in columnas_numericas:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

    # Strings
    columnas_texto = [
        "imdb_id",
        "title",
        "original_title",
        "overview",
        "original_language",
        "poster_path",
        "backdrop_path",
        "director",
    ]

    for columna in columnas_texto:

        df[columna] = (
            df[columna]
            .astype("string")
            .str.strip()
        )

    # Eliminar registros sin ID
    df = df.dropna(
        subset=["tmdb_id"]
    )

    return df


### FUNCION QUE GUARDA TMDN SILVER COMO PARQUET

def save_tmdb_silver(
    df: pd.DataFrame
) -> Path:

    SILVER_TMDB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = (
        SILVER_TMDB_DIR
        / "movies_tmdb.parquet"
    )

    df.to_parquet(
        ruta,
        index=False,
    )

    return ruta


### FUNCION QUE GUARDA METADATA TMDB SILVER

def save_tmdb_silver_metadata(
    df: pd.DataFrame
) -> Path:

    metadata = {
        "fuente": "TMDb",
        "capa": "silver",
        "fecha_transformacion": (
            datetime.now().isoformat()
        ),
        "formato": "parquet",
        "filas": len(df),
        "columnas": len(df.columns),
        "estado": "success",
    }

    ruta_metadata = (
        SILVER_TMDB_DIR
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


### FUNCION PRINCIPAL DE TMDB BRONZE A SILVER

def run_tmdb_bronze_to_silver():

    print("=" * 50)
    print("TMDb: BRONZE -> SILVER")
    print("=" * 50)

    registros = load_tmdb_bronze()

    print(
        "JSON cargados:",
        len(registros)
    )

    df = transform_tmdb_movies(
        registros
    )

    df = clean_tmdb_movies(
        df
    )

    ruta = save_tmdb_silver(
        df
    )

    metadata = (
        save_tmdb_silver_metadata(
            df
        )
    )

    print(
        "Dataset generado:",
        ruta
    )

    print(
        "Filas:",
        len(df)
    )

    print(
        "Columnas:",
        len(df.columns)
    )

    print(
        "Metadata:",
        metadata
    )

    print("=" * 50)
    print("TMDb SILVER COMPLETADO")
    print("=" * 50)

    return df


### FUNCION QUE CARGA LOS JSON DE DISPONIBILIDAD ALMACENADOS EN BRONZE

def load_watchmode_bronze() -> list[dict]:

    archivos = list(
        BRONZE_WATCHMODE_DIR.glob(
            "movie_*_sources.json"
        )
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron JSON Watchmode en "
            f"{BRONZE_WATCHMODE_DIR}"
        )

    registros = []

    for archivo in archivos:

        try:
            with open(
                archivo,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            tmdb_id = (
                archivo.stem
                .replace("movie_", "")
                .replace("_sources", "")
            )

            registros.append({
                "tmdb_id": int(tmdb_id),
                "sources": data,
            })

        except Exception as error:
            print(
                f"Error leyendo {archivo.name}: {error}"
            )

    return registros


### FUNCION QUE NORMALIZA DISPONIBILIDAD A FILAS TABULARES

def transform_watchmode_sources(
    registros: list[dict]
) -> pd.DataFrame:

    filas = []

    for registro in registros:

        tmdb_id = registro["tmdb_id"]
        sources = registro["sources"]

        if not isinstance(sources, list):
            continue

        # Si está vacío, no generamos una plataforma,
        # pero luego podemos conservar el estado aparte.
        for source in sources:

            filas.append({
                "tmdb_id": tmdb_id,
                "source_id": source.get("source_id"),
                "provider_name": source.get("name"),
                "type": source.get("type"),
                "region": source.get("region"),
                "format": source.get("format"),
                "price": source.get("price"),
                "web_url": source.get("web_url"),
                "ios_url": source.get("ios_url"),
                "android_url": source.get(
                    "android_url"
                ),
            })

    return pd.DataFrame(filas)


### FUNCION QUE LIMPIA Y ESTADARIZA LA DISPONIBILIDAD

def clean_watchmode_sources(
    df: pd.DataFrame
) -> pd.DataFrame:

    df = df.copy()

    if df.empty:
        return df

    df = df.drop_duplicates()

    df["tmdb_id"] = pd.to_numeric(
        df["tmdb_id"],
        errors="coerce",
    ).astype("Int64")

    df["source_id"] = pd.to_numeric(
        df["source_id"],
        errors="coerce",
    ).astype("Int64")

    df["price"] = pd.to_numeric(
        df["price"],
        errors="coerce",
    )

    columnas_texto = [
        "provider_name",
        "type",
        "region",
        "format",
        "web_url",
        "ios_url",
        "android_url",
    ]

    for columna in columnas_texto:
        if columna in df.columns:
            df[columna] = (
                df[columna]
                .astype("string")
                .str.strip()
            )

    df = df.dropna(
        subset=["tmdb_id"]
    )

    return df


### FUNCION QUE CREA UNA TABLA DE ESTADO POR PELICULA

def transform_watchmode_status(
    registros: list[dict]
) -> pd.DataFrame:

    filas = []

    for registro in registros:

        sources = registro["sources"]

        disponible = (
            isinstance(sources, list)
            and len(sources) > 0
        )

        filas.append({
            "tmdb_id": registro["tmdb_id"],
            "has_availability": disponible,
            "sources_count": (
                len(sources)
                if isinstance(sources, list)
                else 0
            ),
        })

    return pd.DataFrame(filas)



### FUNCION QUE GUARDA WATCHMODE SILVER

def save_watchmode_silver(
    availability_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> dict[str, Path]:

    SILVER_WATCHMODE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas = {}

    ruta_availability = (
        SILVER_WATCHMODE_DIR
        / "availability.parquet"
    )

    availability_df.to_parquet(
        ruta_availability,
        index=False,
    )

    rutas["availability"] = (
        ruta_availability
    )

    ruta_status = (
        SILVER_WATCHMODE_DIR
        / "availability_status.parquet"
    )

    status_df.to_parquet(
        ruta_status,
        index=False,
    )

    rutas["availability_status"] = (
        ruta_status
    )

    return rutas


### FUNCIONQ UE GUARDA METADATA WATCHMODE SILVER

def save_watchmode_silver_metadata(
    availability_df: pd.DataFrame,
    status_df: pd.DataFrame,
) -> Path:

    metadata = {
        "fuente": "Watchmode",
        "capa": "silver",
        "fecha_transformacion": (
            datetime.now().isoformat()
        ),
        "formato": "parquet",
        "datasets": {
            "availability": {
                "filas": len(
                    availability_df
                ),
                "columnas": len(
                    availability_df.columns
                ),
            },
            "availability_status": {
                "filas": len(
                    status_df
                ),
                "columnas": len(
                    status_df.columns
                ),
            },
        },
        "estado": "success",
    }

    ruta_metadata = (
        SILVER_WATCHMODE_DIR
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


### FUNCION PRINCIPAL QUE EJECUTA WATCHMODE DE BRONZE A SILVER

def run_watchmode_bronze_to_silver():

    print("=" * 50)
    print("WATCHMODE: BRONZE -> SILVER")
    print("=" * 50)

    registros = load_watchmode_bronze()

    print(
        "JSON cargados:",
        len(registros)
    )

    availability_df = (
        transform_watchmode_sources(
            registros
        )
    )

    availability_df = (
        clean_watchmode_sources(
            availability_df
        )
    )

    status_df = (
        transform_watchmode_status(
            registros
        )
    )

    rutas = save_watchmode_silver(
        availability_df,
        status_df,
    )

    metadata = (
        save_watchmode_silver_metadata(
            availability_df,
            status_df,
        )
    )

    print("\nDatasets generados:")

    for nombre, ruta in rutas.items():
        print(
            nombre,
            "->",
            ruta
        )

    print(
        "\nMetadata:",
        metadata
    )

    print("=" * 50)
    print("WATCHMODE SILVER COMPLETADO")
    print("=" * 50)

    return {
        "availability": availability_df,
        "availability_status": status_df,
    }
    
    
if __name__ == "__main__":
    run_movielens_bronze_to_silver()
    print()
    run_tmdb_bronze_to_silver()
    print()
    run_watchmode_bronze_to_silver()