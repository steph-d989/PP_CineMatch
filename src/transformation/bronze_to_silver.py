import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.ingestion.watchmode import (
    get_local_watchmode_files,
    validate_watchmode_local_file,
    get_watchmode_provenance,
)

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

        if nombre == "links":
            # IMDb es un identificador, no una cantidad.
            # Leerlo como texto conserva los ceros iniciales.
            datasets[nombre] = pd.read_csv(
                ruta,
                dtype={"imdbId": "string"},
            )
        else:
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

    # Eliminar únicamente filas idénticas.
    df = df.drop_duplicates()

    # Comprobar que un movieId no tenga
    # identificadores contradictorios.
    duplicados = df[
        df["movieId"].duplicated(keep=False)
    ]

    if not duplicados.empty:
        raise ValueError(
            "MovieLens links contiene movieId "
            "duplicados con información diferente."
        )

    df["movieId"] = pd.to_numeric(
        df["movieId"],
        errors="raise"
    ).astype("int64")

    df["tmdbId"] = pd.to_numeric(
        df["tmdbId"],
        errors="coerce"
    ).astype("Int64")

    # Los identificadores TMDb deben ser positivos.
    # Los valores cero o negativos se consideran
    # desconocidos, sin eliminar la película.
    df["tmdbId"] = df["tmdbId"].where(
        df["tmdbId"] > 0,
        pd.NA,
    )

    # Conservar IMDb como texto.
    # No convertir primero a entero.
    df["imdbId"] = (
        df["imdbId"]
        .astype("string")
        .str.strip()
    )

    df = df.rename(
        columns={
            "movieId": "movie_id",
            "imdbId": "imdb_id",
            "tmdbId": "tmdb_id",
        }
    )

    return df

### FUNCION QUE COMPRUEBA QUE CADA RATING CORRESPONDA A UNA PELICULA DE MOVIELENS
def validate_movielens_rating_references(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> None:

    if "movie_id" not in movies.columns:
        raise ValueError(
            "MovieLens movies no contiene movie_id."
        )

    if "movie_id" not in ratings.columns:
        raise ValueError(
            "MovieLens ratings no contiene movie_id."
        )

    if movies["movie_id"].isna().any():
        raise ValueError(
            "MovieLens movies contiene movie_id nulos."
        )

    if movies["movie_id"].duplicated().any():
        raise ValueError(
            "MovieLens movies contiene movie_id duplicados."
        )

    if ratings["movie_id"].isna().any():
        raise ValueError(
            "MovieLens ratings contiene movie_id nulos."
        )

    ids_validos = pd.Index(movies["movie_id"].unique())

    referencias_invalidas = ~ratings["movie_id"].isin(
        ids_validos
    )

    if referencias_invalidas.any():
        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "MovieLens ratings contiene "
            f"{cantidad} referencias a películas inexistentes."
        )

### FUNCION QUE COMPRUEBA QUE LOS TAGS HAGAN REFERENCIA A PELICULAS EXISTENTE
def validate_movielens_tag_references(
    movies: pd.DataFrame,
    tags: pd.DataFrame,
) -> None:

    if "movie_id" not in movies.columns:
        raise ValueError(
            "MovieLens movies no contiene movie_id."
        )

    if "movie_id" not in tags.columns:
        raise ValueError(
            "MovieLens tags no contiene movie_id."
        )

    if movies["movie_id"].isna().any():
        raise ValueError(
            "MovieLens movies contiene movie_id nulos."
        )

    if movies["movie_id"].duplicated().any():
        raise ValueError(
            "MovieLens movies contiene movie_id duplicados."
        )

    if tags["movie_id"].isna().any():
        raise ValueError(
            "MovieLens tags contiene movie_id nulos."
        )

    referencias_invalidas = ~tags["movie_id"].isin(
        movies["movie_id"]
    )

    if referencias_invalidas.any():
        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "MovieLens tags contiene "
            f"{cantidad} referencias a películas inexistentes."
        )

### FUNCION QUE GUARDA LOS DATASETS SILVER EN FORMATO PARQUET
def save_movielens_silver(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> dict:

    # Seleccionar la carpeta de esta ejecución.
    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_MOVIELENS_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    rutas = {}

    # Guardar los tres datasets principales.
    for nombre in (
        "movies",
        "tags",
        "links",
    ):

        df = datasets[nombre]

        ruta = destino / f"{nombre}.parquet"

        df.to_parquet(
            ruta,
            index=False,
        )

        rutas[nombre] = ruta

    # Guardar ratings dentro de la misma ejecución.
    rutas["ratings"] = save_ratings_partitioned(
        df=datasets["ratings"],
        output_dir=destino / "ratings",
    )

    return rutas

### FUNCION QUE GUARDA LA METADATA SILVER 
def save_movielens_silver_metadata(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> Path:

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_MOVIELENS_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

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
            for nombre, df in datasets.items()
        },
        "estado": "success",
    }

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

### FUNCION QUE GUARDA LOS RATINGS PARTICIONADOS
def save_ratings_partitioned(
    df: pd.DataFrame,
    rows_per_file: int = 2_000_000,
    output_dir: Path | None = None,
) -> list[Path]:

    # Si se proporciona una carpeta, utilizarla.
    # En caso contrario, conservar la ruta anterior.
    ratings_dir = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_MOVIELENS_DIR / "ratings"
    )


    ratings_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    
    # Validar el tamaño de cada partición.
    if rows_per_file <= 0:
        raise ValueError(
            "rows_per_file debe ser mayor que cero"
        )

    # Calcular las particiones que necesita
    # la nueva ejecución.
    cantidad_esperada = len(
        range(0, len(df), rows_per_file)
    )

    nombres_esperados = {
        f"part_{i:03d}.parquet"
        for i in range(cantidad_esperada)
    }

    # Identificar particiones existentes.
    nombres_existentes = {
        archivo.name
        for archivo in ratings_dir.glob("part_*.parquet")
        if archivo.is_file()
    }

    # Identificar archivos que pertenecen
    # a una ejecución anterior.
    particiones_obsoletas = (
        nombres_existentes - nombres_esperados
    )

    if particiones_obsoletas:
        raise RuntimeError(
            "Se detectaron particiones antiguas "
            "de ratings que no pertenecen a la "
            "nueva ejecución: "
            + ", ".join(sorted(particiones_obsoletas))
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
def run_movielens_bronze_to_silver(
    output_dir: Path | None = None,
):

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

    
    # Comprobar la integridad referencial antes
    # de guardar los datasets MovieLens Silver.
    validate_movielens_rating_references(
        movies=silver["movies"],
        ratings=silver["ratings"],
    )

    
    # Validar las referencias de tags antes
    # de guardar los archivos MovieLens Silver.
    validate_movielens_tag_references(
        movies=silver["movies"],
        tags=silver["tags"],
    )


    rutas = save_movielens_silver(
        datasets=silver,
        output_dir=output_dir,
    )



    metadata = save_movielens_silver_metadata(
        datasets=silver,
        output_dir=output_dir,
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
        
    # Verificar que existan películas para procesar.
    if df.empty:
        raise ValueError(
            "TMDb Silver no contiene películas "
            "para procesar."
        )


    # 1. Convertir el identificador al tipo esperado.
    df["tmdb_id"] = pd.to_numeric(
        df["tmdb_id"],
        errors="coerce",
    ).astype("Int64")

    # 2. Eliminar registros sin identificador.
    df = df.dropna(
        subset=["tmdb_id"]
    )

    
    # Comprobar que la limpieza no eliminó
    # todos los registros de películas.
    if df.empty:
        raise ValueError(
            "TMDb Silver no contiene películas "
            "con identificadores válidos."
        )

    
    # 3. Comprobar duplicados antes de descartarlos.
    duplicados = df[
        df["tmdb_id"].duplicated(
            keep=False
        )
    ]

    if not duplicados.empty:
        raise ValueError(
            "TMDb Silver contiene tmdb_id duplicados."
        )


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
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_TMDB_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = destino / "movies_tmdb.parquet"

    df.to_parquet(
        ruta,
        index=False,
    )

    return ruta

### FUNCION QUE GUARDA METADATA TMDB SILVER
def save_tmdb_silver_metadata(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_TMDB_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "fuente": "TMDb",
        "capa": "silver",
        "fecha_transformacion": datetime.now().isoformat(),
        "formato": "parquet",
        "filas": len(df),
        "columnas": len(df.columns),
        "estado": "success",
    }

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

### FUNCION PRINCIPAL DE TMDB BRONZE A SILVER
def run_tmdb_bronze_to_silver(
    output_dir: Path | None = None,
):

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
        df=df,
        output_dir=output_dir,
    )



    metadata = save_tmdb_silver_metadata(
        df=df,
        output_dir=output_dir,
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
def load_watchmode_bronze(
    region: str = "PE",
) -> list[dict]:

    archivos = get_local_watchmode_files(
        region=region
    )

    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos Watchmode "
            f"para la región {region.upper()}"
        )

    registros = []

    for archivo in archivos:

        validacion = validate_watchmode_local_file(
            archivo
        )

        if not validacion["valid"]:
            print(
                f"{archivo.name}: archivo inválido. "
                "Se omite."
            )
            continue

        tmdb_id = validacion["tmdb_id"]

        procedencia = get_watchmode_provenance(
            tmdb_id=tmdb_id,
            region=region,
        )

        if (
            procedencia["status"] != "verified"
            or not procedencia["region_verified"]
        ):
            print(
                f"{archivo.name}: procedencia "
                "no verificada. Se omite."
            )
            continue

        try:
            data = json.loads(
                archivo.read_text(encoding="utf-8")
            )

            registros.append({
                "tmdb_id": tmdb_id,
                "sources": data,
            })

        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print(
                f"{archivo.name}: error de lectura "
                f"- {error}"
            )

    if not registros:
        raise ValueError(
            "No existen archivos Watchmode válidos "
            "y con procedencia verificada."
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


    columnas = [
        "tmdb_id",
        "source_id",
        "provider_name",
        "type",
        "region",
        "format",
        "price",
        "web_url",
        "ios_url",
        "android_url",
    ]

    return pd.DataFrame(
        filas,
        columns=columnas,
    )

### FUNCION QUE LIMPIA Y ESTADARIZA LA DISPONIBILIDAD
def clean_watchmode_sources(
    df: pd.DataFrame
) -> pd.DataFrame:


    df = df.copy()

    # Una tabla vacía también necesita tipos definidos.
    if df.empty:

        df = df.astype({
            "tmdb_id": "Int64",
            "source_id": "Int64",
            "provider_name": "string",
            "type": "string",
            "region": "string",
            "format": "string",
            "price": "Float64",
            "web_url": "string",
            "ios_url": "string",
            "android_url": "string",
        })

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
    output_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Guarda los datasets Watchmode en formato Parquet.

    Permite seleccionar una carpeta por ejecución.
    Mantiene compatibilidad con la ruta anterior.

    No consulta APIs ni Supabase.
    """

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_WATCHMODE_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Guardar las plataformas disponibles.
    ruta_availability = destino / "availability.parquet"

    availability_df.to_parquet(
        ruta_availability,
        index=False,
    )

    # 2. Guardar el estado de consulta por película.
    ruta_status = destino / "availability_status.parquet"

    status_df.to_parquet(
        ruta_status,
        index=False,
    )

    return {
        "availability": ruta_availability,
        "availability_status": ruta_status,
    }

### FUNCIONQ UE GUARDA METADATA WATCHMODE SILVER
def save_watchmode_silver_metadata(
    availability_df: pd.DataFrame,
    status_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """
    Guarda los metadatos de Watchmode Silver.

    Permite seleccionar una carpeta de ejecución.
    Mantiene compatibilidad con la ubicación anterior.

    No consulta APIs ni Supabase.
    """

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_WATCHMODE_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "fuente": "Watchmode",
        "capa": "silver",
        "fecha_transformacion": datetime.now().isoformat(),
        "formato": "parquet",
        "datasets": {
            "availability": {
                "filas": len(availability_df),
                "columnas": len(availability_df.columns),
            },
            "availability_status": {
                "filas": len(status_df),
                "columnas": len(status_df.columns),
            },
        },
        "estado": "success",
    }

    ruta_metadata = destino / "_metadata.json"

    with ruta_metadata.open(
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
def run_watchmode_bronze_to_silver(
    output_dir: Path | None = None,
):

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
        availability_df=availability_df,
        status_df=status_df,
        output_dir=output_dir,
    )

    metadata = save_watchmode_silver_metadata(
        availability_df=availability_df,
        status_df=status_df,
        output_dir=output_dir,
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