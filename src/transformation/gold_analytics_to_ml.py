from pathlib import Path
import json
import pandas as pd

from src.utils.paths import (
    GOLD_DIMENSIONS_DIR,
    GOLD_FACTS_DIR,
    GOLD_ML_CONTENT_DIR,
    GOLD_ML_COLLABORATIVE_DIR,
    GOLD_ML_HYBRID_DIR,
)

from src.utils.gold_runs import (
    validate_gold_run_directories,
    verify_gold_manifest,
    GOLD_REQUIRED_DATASETS,
)

### FUNCION QUE CARGA LOS DATASETS DE GOLD ANALITICS
def load_gold_analytics(
    gold_dirs: dict[str, Path] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Carga los datasets Gold Analytics necesarios
    para construir los modelos de CineMatch.

    Si se proporcionan gold_dirs, utiliza
    exclusivamente esa ejecución versionada.

    Mantiene temporalmente la compatibilidad
    con los directorios antiguos.
    """

    
    if gold_dirs is not None:

        # 1. Validar los directorios del run_id.
        validate_gold_run_directories(
            gold_dirs=gold_dirs,
        )

        raiz = Path(gold_dirs["root"]).resolve()

        # 2. Localizar el manifiesto Gold.
        manifest_path = raiz / "_manifest.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Gold ML requiere _manifest.json "
                "en la ejecución seleccionada."
            )

        # 3. Leer el manifiesto.
        try:
            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as archivo:
                manifest = json.load(archivo)

        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError(
                "Gold ML encontró un manifiesto inválido."
            ) from exc

        if not isinstance(manifest, dict):
            raise ValueError(
                "Gold ML requiere un manifiesto JSON tipo objeto."
            )

        # 4. Verificar run_id, tamaños y SHA-256.
        verify_gold_manifest(
            gold_dirs=gold_dirs,
            manifest=manifest,
        )

        # 5. Comprobar los datasets obligatorios.
        entradas = manifest["archivos"]

        datasets = {
            entrada["dataset"]
            for entrada in entradas
        }

        faltantes = GOLD_REQUIRED_DATASETS - datasets

        if faltantes:
            raise ValueError(
                "Gold ML encontró un manifiesto "
                "con datasets faltantes: "
                + ", ".join(sorted(faltantes))
            )

        # 6. Verificar que dim_movie esté registrada.
        rutas_manifest = {
            entrada["path"]
            for entrada in entradas
        }

        if (
            "dimensions/dim_movie.parquet"
            not in rutas_manifest
        ):
            raise ValueError(
                "Gold ML: dim_movie no está "
                "registrado en el manifiesto."
            )

        # 7. Comprobar las particiones de ratings.
        ratings_dir = (
            raiz / "facts" / "fact_rating"
        )

        particiones = {
            ruta.relative_to(raiz).as_posix()
            for ruta in ratings_dir.glob(
                "part_*.parquet"
            )
        }

        registradas = {
            entrada["path"]
            for entrada in entradas
            if entrada["dataset"] == "fact_rating"
        }

        if not particiones or particiones != registradas:
            raise ValueError(
                "Gold ML: particiones fact_rating "
                "inconsistentes con el manifiesto."
            )

        # 8. Seleccionar los directorios versionados.
        dimensions_dir = Path(
            gold_dirs["dimensions"]
        )

        facts_dir = Path(
            gold_dirs["facts"]
        )

    else:

        # Compatibilidad con versiones anteriores.
        dimensions_dir = GOLD_DIMENSIONS_DIR
        facts_dir = GOLD_FACTS_DIR

    dim_movie_path = (
        dimensions_dir / "dim_movie.parquet"
    )

    ratings_directory = (
        facts_dir / "fact_rating"
    )

    ratings_file = (
        facts_dir / "fact_rating.parquet"
    )

    # En ejecuciones versionadas se exige
    # el dataset particionado oficial.
    if gold_dirs is not None:

        if not (
            ratings_directory.is_dir()
            and any(
                ratings_directory.glob(
                    "part_*.parquet"
                )
            )
        ):
            raise FileNotFoundError(
                "Gold ML requiere fact_rating "
                "particionado en el run_id seleccionado."
            )

        fact_rating_path = ratings_directory

    elif (
        ratings_directory.is_dir()
        and any(
            ratings_directory.glob(
                "part_*.parquet"
            )
        )
    ):
        fact_rating_path = ratings_directory

    elif ratings_file.is_file():

        # Compatibilidad temporal exclusiva
        # para el almacenamiento antiguo.
        fact_rating_path = ratings_file

    else:
        raise FileNotFoundError(
            "No se encontró fact_rating."
        )

    if not dim_movie_path.is_file():
        raise FileNotFoundError(
            f"No se encontró: {dim_movie_path}"
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

    
    # Verificar que el catálogo tenga al menos
    # alguna característica semántica aprovechable.
    columnas_semanticas = [
        "genres",
        "tmdb_genres",
        "keywords",
        "director",
        "actors",
        "overview",
    ]

    disponibles = [
        columna
        for columna in columnas_semanticas
        if columna in df.columns
    ]

    if not disponibles:
        raise ValueError(
            "Gold ML no contiene columnas "
            "semánticas para Content-Based."
        )

    def tiene_contenido(valor):

        if valor is None:
            return False

        if isinstance(valor, (list, tuple, set)):
            return any(
                isinstance(item, str) and item.strip()
                for item in valor
            )

        if isinstance(valor, str):
            return bool(valor.strip())

        return False

    hay_contenido = any(
        df[columna].map(tiene_contenido).any()
        for columna in disponibles
    )

    if not hay_contenido:
        raise ValueError(
            "Gold ML no contiene información "
            "semántica aprovechable."
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
def run_gold_analytics_to_ml(
    gold_dirs: dict[str, Path] | None = None,
):


    print("=" * 60)
    print("GOLD ANALYTICS -> GOLD ML")
    print("=" * 60)

    
    # Seleccionar la ejecución Gold de origen.
    # Mantener compatibilidad con las llamadas
    # anteriores mientras migramos el pipeline.

    if gold_dirs is not None:
        analytics = load_gold_analytics(
            gold_dirs=gold_dirs,
        )
    else:
        analytics = load_gold_analytics()


    # Seleccionar los directorios de salida ML.
    # Las ejecuciones versionadas conservan
    # sus resultados dentro del mismo run_id.

    if gold_dirs is not None:

        validate_gold_run_directories(
            gold_dirs=gold_dirs,
        )

        content_dir = Path(
            gold_dirs["ml_content"]
        )

        collaborative_dir = Path(
            gold_dirs["ml_collaborative"]
        )

    else:

        # Compatibilidad temporal con las rutas
        # utilizadas antes del versionado.
        content_dir = GOLD_ML_CONTENT_DIR
        collaborative_dir = GOLD_ML_COLLABORATIVE_DIR

    
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
            content_dir,
            "content_based_features.parquet",
        )
    )



    rutas["collaborative_interactions"] = (
        save_partitioned_parquet(
            collaborative_interactions,
            collaborative_dir,
            "collaborative_interactions",
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