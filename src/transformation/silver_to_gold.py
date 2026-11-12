import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_INTEGRATED_DIR,
    SILVER_WATCHMODE_DIR,
    GOLD_DIR,
    GOLD_DIMENSIONS_DIR,
    GOLD_FACTS_DIR,
    GOLD_METRICS_DIR,
    GOLD_ML_CONTENT_DIR,
    GOLD_ML_COLLABORATIVE_DIR,
)

from src.utils.gold_runs import (
    validate_gold_run_directories,
    finalize_gold_run_paths,
    seal_gold_run,
)


from src.validation.gold_integrity import (
    validate_fact_rating_user_references,
    validate_fact_rating_movie_references,
    validate_fact_rating_date_references,
    validate_fact_availability_platform_references,
    validate_fact_availability_movie_references,
    validate_fact_availability_date_references,
    validate_gold_dim_fact_files,
)


#FUNCION QUE CARGA LOS DATOS DE SILVER
def load_silver_for_gold(
    run_dirs: dict[str, Path] | None = None,
) -> dict[str, pd.DataFrame]:

    
    # Seleccionar las carpetas de Silver.
    # Si no se proporciona run_dirs, conservar
    # temporalmente las rutas tradicionales.

    movielens_dir = (
        Path(run_dirs["movielens"])
        if run_dirs is not None
        else SILVER_MOVIELENS_DIR
    )

    watchmode_dir = (
        Path(run_dirs["watchmode"])
        if run_dirs is not None
        else SILVER_WATCHMODE_DIR
    )

    integrated_dir = (
        Path(run_dirs["integrated"])
        if run_dirs is not None
        else SILVER_INTEGRATED_DIR
    )



    movies_path = (
        integrated_dir / "movies_integrated.parquet"
    )

    ratings_path = (
        movielens_dir / "ratings"
    )

    tags_path = (
        movielens_dir / "tags.parquet"
    )

    availability_path = (
        watchmode_dir / "availability.parquet"
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

### FUNCION QUE CONTRUYE LA DIMENSION USER
def build_dim_user(
    ratings: pd.DataFrame,
) -> pd.DataFrame:

    if "user_id" not in ratings.columns:
        raise ValueError(
            "Ratings no contiene la columna user_id."
        )

    ids = pd.to_numeric(
        ratings["user_id"],
        errors="coerce",
    )

    if (
        ids.isna().any()
        or (ids <= 0).any()
        or (ids % 1 != 0).any()
    ):
        raise ValueError(
            "Ratings contiene user_id inválidos."
        )

    usuarios = pd.DataFrame({
        "user_id": ids.astype("int64").unique()
    })

    usuarios = usuarios.sort_values(
        "user_id"
    ).reset_index(drop=True)

    return usuarios

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

    # 1. Verificar que exista la clave principal.
    if "movie_id" not in df.columns:
        raise ValueError(
            "dim_movie no contiene movie_id."
        )

    # 2. No permitir identificadores nulos.
    if df["movie_id"].isna().any():
        raise ValueError(
            "dim_movie contiene movie_id nulos."
        )

    # 3. No eliminar duplicados silenciosamente.
    if df["movie_id"].duplicated().any():
        raise ValueError(
            "dim_movie contiene movie_id duplicados."
        )

    # 4. Validar que sean enteros positivos.
    ids = pd.to_numeric(
        df["movie_id"],
        errors="coerce",
    )

    if (
        ids.isna().any()
        or (ids <= 0).any()
        or (ids % 1 != 0).any()
    ):
        raise ValueError(
            "dim_movie contiene movie_id inválidos."
        )

    # 5. Normalizar el tipo de identificador.
    df["movie_id"] = ids.astype("int64")

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

    
    # Validar que cada plataforma tenga
    # un identificador único y válido.

    if df["source_id"].isna().any():
        raise ValueError(
            "dim_platform contiene source_id nulos."
        )

    ids = pd.to_numeric(
        df["source_id"],
        errors="coerce",
    )

    if (
        ids.isna().any()
        or (ids <= 0).any()
        or (ids % 1 != 0).any()
    ):
        raise ValueError(
            "dim_platform contiene source_id inválidos."
        )

    # Un mismo source_id no puede identificar
    # plataformas con nombres diferentes.
    if df["source_id"].duplicated().any():
        raise ValueError(
            "dim_platform contiene identificadores "
            "de plataforma duplicados o contradictorios."
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
    ratings: pd.DataFrame,
    required_date: str | None = None,
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

    
    # Incluir la fecha de procesamiento Gold
    # cuando se haya indicado explícitamente.

    if required_date is not None:

        fecha_requerida = pd.to_datetime(
            required_date,
            format="%Y-%m-%d",
            errors="raise",
        ).normalize()

        # Ampliar el calendario si la fecha
        # está fuera del intervalo existente.
        if fecha_requerida < fecha_min:
            fecha_min = fecha_requerida

        if fecha_requerida > fecha_max:
            fecha_max = fecha_requerida

    
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
    processing_date: str | None = None,
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
    # Detectar cuando un mismo tmdb_id corresponde
    # a varias películas diferentes de MovieLens.
    relaciones = (
        dim_movie[["movie_id", "tmdb_id"]]
        .dropna(subset=["tmdb_id", "movie_id"])
        .drop_duplicates()
    )

    cantidad_peliculas = (
        relaciones.groupby("tmdb_id")["movie_id"]
        .nunique()
    )

    tmdb_ambiguos = cantidad_peliculas[
        cantidad_peliculas > 1
    ]

    if not tmdb_ambiguos.empty:
        raise ValueError(
            "Gold contiene tmdb_id asociados a "
            "varios movie_id. No se puede construir "
            "fact_availability de forma segura."
        )

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


    # Fecha de procesamiento Gold.
    # No representa necesariamente la fecha
    # original de consulta de Watchmode.

    if processing_date is None:
        fecha_proceso = pd.Timestamp.now().normalize()
    else:
        fecha_proceso = pd.to_datetime(
            processing_date,
            format="%Y-%m-%d",
            errors="raise",
        ).normalize()

    df["date_key"] = int(
        fecha_proceso.strftime("%Y%m%d")
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


    # Eliminar solamente ofertas idénticas
    # en todos los atributos conservados.
    #
    # No consolidar ofertas diferentes por
    # formato, precio o enlaces de acceso.

    df = df.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)


    return df

### FUNCION QUE CLASIFICA LAS OFERTAS DE STREAMING SEGUN TENGAN O NO CORRESPONDENCIA CON MOVIELENS
def classify_availability_movie_links(
    fact_availability: pd.DataFrame,
) -> dict[str, pd.DataFrame]:

    columnas_requeridas = {
        "movie_id",
        "tmdb_id",
    }

    faltantes = (
        columnas_requeridas
        - set(fact_availability.columns)
    )

    if faltantes:
        raise ValueError(
            "fact_availability no contiene columnas: "
            + ", ".join(sorted(faltantes))
        )

    # Identificar ofertas con y sin película MovieLens.
    tiene_pelicula = (
        fact_availability["movie_id"].notna()
    )

    vinculadas = (
        fact_availability.loc[tiene_pelicula]
        .copy()
        .reset_index(drop=True)
    )

    no_vinculadas = (
        fact_availability.loc[~tiene_pelicula]
        .copy()
        .reset_index(drop=True)
    )

    return {
        "linked": vinculadas,
        "unlinked": no_vinculadas,
    }

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
    rutas: dict[str, Path | list[Path]],
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
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

        referencia = rutas[nombre]

        informacion = {
            "filas": len(df),
            "columnas": len(df.columns),
        }

        if isinstance(referencia, list):

            # Dataset almacenado en varias particiones.
            informacion["rutas"] = [
                str(ruta)
                for ruta in referencia
            ]

        else:

            # Dataset almacenado en un solo archivo.
            informacion["ruta"] = str(referencia)

        metadata["datasets"][nombre] = informacion

    # Seleccionar el destino de los metadatos.
    # Conserva la ruta anterior si no se proporciona
    # un directorio específico de ejecución.

    destino = (
        Path(output_dir)
        if output_dir is not None
        else GOLD_METRICS_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_metadata = destino / "_metadata.json"


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

### FUNCION QUE GUARDA UN DATAFRAME GRANDE EN MULTIPLES ARCHIVOS
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

### FUNCION QUE EJECUTA ORQUESTADOR DE SILVER A GOLD
def run_silver_to_gold(
    run_dirs: dict[str, Path] | None = None,
    processing_date: str | None = None,
    gold_dirs: dict[str, Path] | None = None,
    seal_run: bool = False,
) -> dict:

    if seal_run and gold_dirs is None:
        raise ValueError(
            "El sellado Gold requiere gold_dirs versionados."
        )

    # Comprobar el aislamiento de los directorios
    # antes de iniciar el procesamiento Gold.
    if gold_dirs is not None:
        validate_gold_run_directories(
            gold_dirs=gold_dirs,
        )

    
    # Impedir reutilizar una ejecución Gold existente.
    # No borrar archivos ni sobrescribir resultados
    # de intentos anteriores.

    if gold_dirs is not None:

        run_root = Path(gold_dirs["root"])

        if run_root.exists():
            raise FileExistsError(
                "La ejecución Gold ya existe. "
                "Utiliza un nuevo run_id."
            )

    
    # Comprobar que Silver y Gold pertenecen
    # a la misma ejecución cuando se proporcionan
    # ambos conjuntos de directorios.
    if run_dirs is not None and gold_dirs is not None:

        silver_run_id = str(
            UUID(
                Path(run_dirs["root"])
                .name
                .removeprefix("run_id=")
            )
        )

        gold_run_id = str(
            UUID(
                Path(gold_dirs["root"])
                .name
                .removeprefix("run_id=")
            )
        )

        if silver_run_id != gold_run_id:
            raise ValueError(
                "Silver y Gold pertenecen "
                "a diferentes run_id."
            )


    print("=" * 60)
    print("SILVER -> GOLD CINEMATCH")
    print("=" * 60)

    # Cargar Silver


    silver = load_silver_for_gold(
        run_dirs=run_dirs,
    )


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

    
    # Dimensión de usuarios MovieLens.
    dim_user = build_dim_user(
        silver["ratings"]
    )

    dim_movie = build_dim_movie(
        silver["movies"]
    )
    
    dim_platform = build_dim_platform(
        silver["availability"]
    )
    
    # Utilizar la misma fecha de procesamiento
    # en la dimensión y en los hechos.

    parametros_calendario = {}

    if processing_date is not None:
        parametros_calendario["required_date"] = processing_date

    dim_date = build_dim_date(
        silver["ratings"],
        **parametros_calendario,
    )

    # FACTS
    
    print("Construyendo facts...")


    fact_rating = build_fact_rating(
        silver["ratings"]
    )

    # 1. Validar los usuarios de las calificaciones.
    validate_fact_rating_user_references(
        fact_rating=fact_rating,
        dim_user=dim_user,
    )

    # 2. Validar las películas de las calificaciones.
    validate_fact_rating_movie_references(
        fact_rating=fact_rating,
        dim_movie=dim_movie,
    )

    # 3. Validar las fechas de las calificaciones.
    validate_fact_rating_date_references(
        fact_rating=fact_rating,
        dim_date=dim_date,
    )


    # Transmitir la fecha de procesamiento solamente
    # cuando se haya proporcionado explícitamente.
    # Esto conserva la compatibilidad con llamadas
    # anteriores y pruebas existentes.

    parametros_fecha = {}

    if processing_date is not None:
        parametros_fecha["processing_date"] = processing_date

    fact_availability = build_fact_availability(
        silver["availability"],
        dim_movie,
        dim_platform,
        **parametros_fecha,
    )

    # Clasificar las ofertas según su correspondencia
    # con el catálogo MovieLens.
    clasificacion = classify_availability_movie_links(
        fact_availability
    )

    # Solo las ofertas vinculadas formarán parte
    # de la tabla de hechos.
    fact_availability = clasificacion["linked"]

    # Conservar las ofertas no vinculadas para auditoría.
    availability_unlinked = clasificacion["unlinked"]

    
    # Validar que todas las ofertas vinculadas
    # correspondan a plataformas de dim_platform.
    validate_fact_availability_platform_references(
        fact_availability=fact_availability,
        dim_platform=dim_platform,
    )

    
    # Validar que todas las ofertas vinculadas
    # correspondan a películas de dim_movie.
    validate_fact_availability_movie_references(
        fact_availability=fact_availability,
        dim_movie=dim_movie,
    )

    
    # Comprobar que las fechas de disponibilidad
    # existan en la dimensión de calendario.
    validate_fact_availability_date_references(
        fact_availability=fact_availability,
        dim_date=dim_date,
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

    # Seleccionar el directorio de dimensiones
    # de la ejecución Gold.
    dimensions_dir = (
        Path(gold_dirs["dimensions"])
        if gold_dirs is not None
        else GOLD_DIMENSIONS_DIR
    )

    # Seleccionar el directorio de hechos
    # correspondiente a la ejecución Gold.
    facts_dir = (
        Path(gold_dirs["facts"])
        if gold_dirs is not None
        else GOLD_FACTS_DIR
    )

    
    # Directorio de métricas correspondiente
    # a la ejecución Gold seleccionada.
    metrics_dir = (
        Path(gold_dirs["metrics"])
        if gold_dirs is not None
        else GOLD_METRICS_DIR
    )
    
    # Directorio de diagnóstico de esta ejecución.
    quality_dir = (
        Path(gold_dirs["quality"])
        if gold_dirs is not None
        else GOLD_DIR / "quality"
    )

    # DIMENSIONS

    rutas["dim_user"] = save_gold_dataset(
        dim_user,
        dimensions_dir,
        "dim_user.parquet",
    )
    datasets["dim_user"] = dim_user

    rutas["dim_movie"] = save_gold_dataset(
        dim_movie,
        dimensions_dir,
        "dim_movie.parquet",
    )
    datasets["dim_movie"] = dim_movie

    rutas["dim_platform"] = save_gold_dataset(
        dim_platform,
        dimensions_dir,
        "dim_platform.parquet",
    )
    datasets["dim_platform"] = dim_platform

    rutas["dim_date"] = save_gold_dataset(
        dim_date,
        dimensions_dir,
        "dim_date.parquet",
    )
    datasets["dim_date"] = dim_date

    # FACTS
    
    # Registrar el dataset de calificaciones.
    # Su almacenamiento se realizará únicamente
    # mediante las particiones Parquet posteriores.
    datasets["fact_rating"] = fact_rating
    
    rutas["fact_availability"] = save_gold_dataset(
        fact_availability,
        facts_dir,
        "fact_availability.parquet",
    )

    datasets["fact_availability"] = (
        fact_availability
    )
    
    # Guardar separadamente los registros
    # que no tienen correspondencia con MovieLens.

    rutas["availability_unlinked"] = save_gold_dataset(
        availability_unlinked,
        quality_dir,
        "availability_unlinked.parquet",
    )

    datasets["availability_unlinked"] = availability_unlinked
    
    # METRICS


    rutas["movie_metrics"] = save_gold_dataset(
        movie_metrics,
        metrics_dir,
        "movie_metrics.parquet",
    )

    datasets["movie_metrics"] = movie_metrics

    

    rutas["fact_rating"] = save_partitioned_parquet(
        fact_rating,
        facts_dir,
        "fact_rating",
    )

    

    rutas["user_metrics"] = save_gold_dataset(
        user_metrics,
        metrics_dir,
        "user_metrics.parquet",
    )

    datasets["user_metrics"] = user_metrics


    # METADATA


    # METADATA

    # Utilizar el directorio de métricas
    # correspondiente a la ejecución Gold.
    metadata = save_gold_metadata(
        rutas=rutas,
        datasets=datasets,
        output_dir=metrics_dir,
    )

    
    # Verificar las rutas de salida cuando Gold
    # utiliza una ejecución versionada.

    resultado_rutas = None

    if gold_dirs is not None:
        resultado_rutas = finalize_gold_run_paths(
            gold_dirs=gold_dirs,
            rutas=rutas,
            metadata_path=metadata,
        )

    

    resultado_sellado = None
    integridad_dim_fact = None

    if seal_run:

        # Verificar las seis relaciones sobre
        # los archivos Parquet ya generados.
        integridad_dim_fact = (
            validate_gold_dim_fact_files(
                rutas=rutas,
            )
        )

        # Solo permitir el sellado si la
        # integridad referencial es correcta.
        resultado_sellado = seal_gold_run(
            gold_dirs=gold_dirs,
            rutas=rutas,
            metadata_path=metadata,
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
        "validacion_rutas": resultado_rutas,
        "integridad_dim_fact": integridad_dim_fact,
        "sellado": resultado_sellado,
    }



if __name__ == "__main__":
    run_silver_to_gold()