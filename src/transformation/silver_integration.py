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
def load_silver_datasets(
    run_dirs: dict[str, Path] | None = None,
    include_availability: bool = False,
):


    
    movielens_dir = (
        Path(run_dirs["movielens"])
        if run_dirs is not None
        else SILVER_MOVIELENS_DIR
    )

    tmdb_dir = (
        Path(run_dirs["tmdb"])
        if run_dirs is not None
        else SILVER_TMDB_DIR
    )

    watchmode_dir = (
        Path(run_dirs["watchmode"])
        if run_dirs is not None
        else SILVER_WATCHMODE_DIR
    )

    rutas = {
        "movies": (
            movielens_dir / "movies.parquet"
        ),
        "links": (
            movielens_dir / "links.parquet"
        ),
        "tmdb": (
            tmdb_dir / "movies_tmdb.parquet"
        ),
        "watchmode_status": (
            watchmode_dir / "availability_status.parquet"
        ),
    }

    
    # Incluir las plataformas para validar
    # las cantidades declaradas en Watchmode.
    if include_availability:
        rutas["watchmode_availability"] = (
            watchmode_dir / "availability.parquet"
        )

    
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

    
    # No permitir películas TMDb sin identificador.
    # Evita coincidencias incorrectas entre valores nulos.
    if tmdb["tmdb_id"].isna().any():
        raise ValueError(
            "TMDb Silver contiene tmdb_id nulos. "
            "No se puede realizar una integración segura."
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

    # Detectar duplicados antes de integrar.
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

    # Conservar disponibilidad desconocida.
    df["has_availability"] = (
        df["has_availability"].astype("boolean")
    )

    df["sources_count"] = (
        df["sources_count"].astype("Int64")
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

### FUNCION QUE COMPRUEBA QUE SOURCES_COUNT COINCIDA CON LAS FILAS REALES DE DISPONIBILIDAD DE WATCHMODE
def validate_watchmode_source_counts(
    availability: pd.DataFrame,
    status: pd.DataFrame,
) -> None:

    required_availability = {"tmdb_id"}
    required_status = {
        "tmdb_id",
        "sources_count",
    }

    if not required_availability.issubset(availability.columns):
        raise ValueError(
            "Watchmode availability no contiene tmdb_id."
        )

    if not required_status.issubset(status.columns):
        raise ValueError(
            "Watchmode status no contiene columnas obligatorias."
        )

    if status["tmdb_id"].isna().any():
        raise ValueError(
            "Watchmode status contiene tmdb_id nulos."
        )

    if status["tmdb_id"].duplicated().any():
        raise ValueError(
            "Watchmode status contiene tmdb_id duplicados."
        )

    conteos = availability.groupby(
        "tmdb_id"
    ).size()

    # No debe haber plataformas de películas
    # que no aparezcan en la tabla de estado.
    ids_huerfanos = set(conteos.index) - set(
        status["tmdb_id"]
    )

    if ids_huerfanos:
        raise ValueError(
            "Watchmode contiene plataformas sin estado."
        )

    esperados = (
        status["tmdb_id"]
        .map(conteos)
        .fillna(0)
        .astype("int64")
    )

    declarados = pd.to_numeric(
        status["sources_count"],
        errors="coerce",
    )

    if (
        declarados.isna().any()
        or (declarados != esperados).any()
    ):
        raise ValueError(
            "Watchmode sources_count no coincide "
            "con las plataformas registradas."
        )

### FUNCION PRINCIPAL DE INTEGRACION CONSTRUYE UN CATALOGO INTEGRADO
def build_integrated_movies(
    datasets: dict[str, pd.DataFrame]
) -> pd.DataFrame:

    
    # Validar las cantidades reales de plataformas
    # cuando se haya cargado availability.parquet.
    if "watchmode_availability" in datasets:
        validate_watchmode_source_counts(
            availability=datasets["watchmode_availability"],
            status=datasets["watchmode_status"],
        )

    
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
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """
    Guarda el catálogo integrado Silver en Parquet.

    Permite seleccionar una carpeta de ejecución.
    Mantiene compatibilidad con la ruta anterior.

    No consulta APIs ni Supabase.
    """

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_INTEGRATED_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta = destino / "movies_integrated.parquet"

    df.to_parquet(
        ruta,
        index=False,
    )

    return ruta

### FUNCION QUE VALIDA LAS CLAVES DEL CATALOGO INTEGRATED, 1 PELI APARECE SOLO 1 VEZ, CON SOLO UN MOVIE_ID
def validate_integrated_catalog(
    df: pd.DataFrame,
) -> pd.DataFrame:


    if "movie_id" not in df.columns:
        raise ValueError(
            "Integrated no contiene la columna movie_id."
        )

    if df["movie_id"].isna().any():
        raise ValueError(
            "Integrated contiene movie_id nulos."
        )

    if df["movie_id"].duplicated().any():
        raise ValueError(
            "Integrated contiene movie_id duplicados."
        )

    # Validar que movie_id sea numérico,
    # entero y estrictamente positivo.
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
            "Integrated contiene movie_id inválidos. "
            "Deben ser enteros positivos."
        )


    # Validar coherencia de la disponibilidad Watchmode.
    columnas_watchmode = {
        "watchmode_checked",
        "has_availability",
        "sources_count",
    }

    faltantes = columnas_watchmode - set(df.columns)

    if faltantes:
        raise ValueError(
            "Integrated no contiene columnas Watchmode: "
            + ", ".join(sorted(faltantes))
        )

    checked = df["watchmode_checked"].astype("boolean")
    available = df["has_availability"].astype("boolean")
    count = pd.to_numeric(
        df["sources_count"],
        errors="coerce",
    )

    # No aceptamos un estado de consulta desconocido.
    if checked.isna().any():
        raise ValueError(
            "Integrated contiene estados Watchmode incoherentes."
        )

    # Películas consultadas:
    # deben tener disponibilidad y conteo conocidos.
    consultadas = checked.fillna(False)

    invalidas_consultadas = (
        consultadas
        & (
            available.isna()
            | count.isna()
            | (count < 0)
            | (available.fillna(False) & (count <= 0))
            | (~available.fillna(False) & (count != 0))
        ).fillna(True)
    )

    # Películas no consultadas:
    # la disponibilidad y el conteo deben ser desconocidos.
    invalidas_no_consultadas = (
        ~consultadas
        & (
            available.notna()
            | count.notna()
        )
    )

    if (
        invalidas_consultadas.any()
        or invalidas_no_consultadas.any()
    ):
        raise ValueError(
            "Integrated contiene estados Watchmode incoherentes."
        )


    return df

### FUNCION QIE GUARDA LOS METADATOS
def save_integrated_metadata(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """
    Guarda los metadatos del catálogo integrado.

    Permite seleccionar una carpeta por ejecución.
    Mantiene compatibilidad con la ubicación anterior.

    No consulta APIs ni Supabase.
    """

    destino = (
        Path(output_dir)
        if output_dir is not None
        else SILVER_INTEGRATED_DIR
    )

    destino.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "dataset": "movies_integrated",
        "capa": "silver",
        "fecha_transformacion": datetime.now().isoformat(),
        "formato": "parquet",
        "filas": len(df),
        "columnas": len(df.columns),
        "con_tmdb": int(
            df["tmdb_title"].notna().sum()
        ),
        "watchmode_consultadas": int(
            df["watchmode_checked"].sum()
        ),
        "con_disponibilidad": int(
            df["has_availability"].sum()
        ),
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


### FUNCION QUE ACTUA DE ORQUESTADOR 
def run_silver_integration(
    output_dir: Path | None = None,
    run_dirs: dict[str, Path] | None = None,
):


    print("=" * 60)
    print("INTEGRACIÓN SILVER CINEMATCH")
    print("=" * 60)

    
    if run_dirs is not None:

        if output_dir is not None and (
            Path(output_dir) != Path(run_dirs["integrated"])
        ):
            raise ValueError(
                "El directorio Integrated no coincide "
                "con la ejecución seleccionada."
            )

        output_dir = Path(run_dirs["integrated"])


    datasets = load_silver_datasets(
        run_dirs=run_dirs,
        include_availability=True,
    )



    df = build_integrated_movies(
        datasets
    )

    
    # Validar las claves antes de guardar.
    df = validate_integrated_catalog(df)

    
    ruta = save_integrated_movies(
        df=df,
        output_dir=output_dir,
    )


    metadata = save_integrated_metadata(
            df=df,
            output_dir=output_dir,
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