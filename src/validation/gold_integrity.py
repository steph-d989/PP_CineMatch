
"""
Validaciones de integridad dimensional de CineMatch.

Estas funciones no leen ni escriben archivos.
Tampoco realizan conexiones con Supabase.
"""

from pathlib import Path
import pandas as pd
from pandas.api.types import is_integer_dtype


def validate_fact_rating_user_references(
    fact_rating: pd.DataFrame,
    dim_user: pd.DataFrame,
) -> None:
    """
    Comprueba la relación:
    fact_rating.user_id -> dim_user.user_id
    """

    if "user_id" not in fact_rating.columns:
        raise ValueError(
            "fact_rating no contiene user_id."
        )

    if "user_id" not in dim_user.columns:
        raise ValueError(
            "dim_user no contiene user_id."
        )

    # Una dimensión no puede contener claves nulas.
    if dim_user["user_id"].isna().any():
        raise ValueError(
            "dim_user contiene user_id nulos."
        )

    # La clave de la dimensión debe ser única.
    if dim_user["user_id"].duplicated().any():
        raise ValueError(
            "dim_user contiene user_id duplicados."
        )

    # Ningún hecho puede tener un usuario desconocido.
    if fact_rating["user_id"].isna().any():
        raise ValueError(
            "fact_rating contiene user_id nulos."
        )

    # Comprobar que cada usuario de los hechos
    # exista en la dimensión.
    referencias_invalidas = (
        ~fact_rating["user_id"].isin(
            dim_user["user_id"]
        )
    )

    if referencias_invalidas.any():

        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "fact_rating contiene "
            f"{cantidad} referencias a usuarios inexistentes."
        )

def validate_fact_rating_movie_references(
    fact_rating: pd.DataFrame,
    dim_movie: pd.DataFrame,
) -> None:
    """
    Comprueba la relación:
    fact_rating.movie_id -> dim_movie.movie_id

    No lee ni modifica archivos.
    """

    if "movie_id" not in fact_rating.columns:
        raise ValueError(
            "fact_rating no contiene movie_id."
        )

    if "movie_id" not in dim_movie.columns:
        raise ValueError(
            "dim_movie no contiene movie_id."
        )

    # La dimensión debe tener claves válidas y únicas.
    if dim_movie["movie_id"].isna().any():
        raise ValueError(
            "dim_movie contiene movie_id nulos."
        )

    if dim_movie["movie_id"].duplicated().any():
        raise ValueError(
            "dim_movie contiene movie_id duplicados."
        )

    # Los hechos no deben tener claves nulas.
    if fact_rating["movie_id"].isna().any():
        raise ValueError(
            "fact_rating contiene movie_id nulos."
        )

    # Comprobar correspondencia con la dimensión.
    referencias_invalidas = (
        ~fact_rating["movie_id"].isin(
            dim_movie["movie_id"]
        )
    )

    if referencias_invalidas.any():

        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "fact_rating contiene "
            f"{cantidad} referencias a películas inexistentes."
        )

def validate_fact_rating_date_references(
    fact_rating: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> None:
    """
    Comprueba la relación:
    fact_rating.date_key -> dim_date.date_key

    No lee ni escribe archivos.
    """

    if "date_key" not in fact_rating.columns:
        raise ValueError(
            "fact_rating no contiene date_key."
        )

    if "date_key" not in dim_date.columns:
        raise ValueError(
            "dim_date no contiene date_key."
        )

    if dim_date["date_key"].isna().any():
        raise ValueError(
            "dim_date contiene date_key nulos."
        )

    if dim_date["date_key"].duplicated().any():
        raise ValueError(
            "dim_date contiene date_key duplicados."
        )

    if fact_rating["date_key"].isna().any():
        raise ValueError(
            "fact_rating contiene date_key nulos."
        )

    referencias_invalidas = (
        ~fact_rating["date_key"].isin(
            dim_date["date_key"]
        )
    )

    if referencias_invalidas.any():
        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "fact_rating contiene "
            f"{cantidad} referencias a fechas inexistentes."
        )

def validate_fact_availability_platform_references(
    fact_availability: pd.DataFrame,
    dim_platform: pd.DataFrame,
) -> None:
    """
    Comprueba la relación:
    fact_availability.provider_id
        -> dim_platform.provider_id

    No lee ni escribe archivos.
    """

    if "provider_id" not in fact_availability.columns:
        raise ValueError(
            "fact_availability no contiene provider_id."
        )

    if "provider_id" not in dim_platform.columns:
        raise ValueError(
            "dim_platform no contiene provider_id."
        )

    # Las claves de la dimensión deben existir.
    if dim_platform["provider_id"].isna().any():
        raise ValueError(
            "dim_platform contiene provider_id nulos."
        )

    # Cada identificador debe aparecer una sola vez.
    if dim_platform["provider_id"].duplicated().any():
        raise ValueError(
            "dim_platform contiene provider_id duplicados."
        )

    # No se permiten hechos sin plataforma.
    if fact_availability["provider_id"].isna().any():
        raise ValueError(
            "fact_availability contiene provider_id nulos."
        )

    # Comprobar las claves foráneas.
    referencias_invalidas = (
        ~fact_availability["provider_id"].isin(
            dim_platform["provider_id"]
        )
    )

    if referencias_invalidas.any():

        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "fact_availability contiene "
            f"{cantidad} referencias a plataformas inexistentes."
        )

def validate_fact_availability_movie_references(
    fact_availability: pd.DataFrame,
    dim_movie: pd.DataFrame,
) -> None:
    """
    Comprueba la relación:
    fact_availability.movie_id -> dim_movie.movie_id

    Esta función valida únicamente las ofertas
    que deben formar parte de la tabla de hechos.
    """

    if "movie_id" not in fact_availability.columns:
        raise ValueError(
            "fact_availability no contiene movie_id."
        )

    if "movie_id" not in dim_movie.columns:
        raise ValueError(
            "dim_movie no contiene movie_id."
        )

    if dim_movie["movie_id"].isna().any():
        raise ValueError(
            "dim_movie contiene movie_id nulos."
        )

    if dim_movie["movie_id"].duplicated().any():
        raise ValueError(
            "dim_movie contiene movie_id duplicados."
        )

    if fact_availability["movie_id"].isna().any():
        raise ValueError(
            "fact_availability contiene movie_id nulos."
        )

    referencias_invalidas = (
        ~fact_availability["movie_id"].isin(
            dim_movie["movie_id"]
        )
    )

    if referencias_invalidas.any():
        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "fact_availability contiene "
            f"{cantidad} referencias a películas inexistentes."
        )


def validate_fact_availability_date_references(
    fact_availability: pd.DataFrame,
    dim_date: pd.DataFrame,
) -> None:
    """
    Valida la relación:
    fact_availability.date_key -> dim_date.date_key

    Comprueba que todas las fechas de disponibilidad
    estén presentes en la dimensión de calendario.

    No consulta APIs ni modifica archivos.
    """

    # 1. Comprobar columnas obligatorias.
    if "date_key" not in fact_availability.columns:
        raise ValueError(
            "fact_availability no contiene date_key."
        )

    if "date_key" not in dim_date.columns:
        raise ValueError(
            "dim_date no contiene date_key."
        )

    # 2. Validar las claves de la dimensión.
    if dim_date["date_key"].isna().any():
        raise ValueError(
            "dim_date contiene date_key nulos."
        )

    if dim_date["date_key"].duplicated().any():
        raise ValueError(
            "dim_date contiene date_key duplicados."
        )

    # 3. Validar las claves de los hechos.
    if fact_availability["date_key"].isna().any():
        raise ValueError(
            "fact_availability contiene date_key nulos."
        )

    # 4. Comprobar la integridad referencial.
    referencias_invalidas = (
        ~fact_availability["date_key"].isin(
            dim_date["date_key"]
        )
    )

    if referencias_invalidas.any():
        cantidad = int(referencias_invalidas.sum())

        raise ValueError(
            "fact_availability contiene "
            f"{cantidad} referencias a fechas inexistentes."
        )


def validate_gold_dim_movie_file(
    parquet_path: Path,
) -> dict:
    """
    Valida la clave principal de dim_movie
    directamente desde un archivo Parquet.

    Comprueba:
    - Existencia de movie_id.
    - Ausencia de valores nulos.
    - Tipo entero.
    - Identificadores positivos.
    - Ausencia de duplicados.

    No modifica el archivo.
    """

    ruta = Path(parquet_path)

    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe dim_movie: {ruta}"
        )

    # Leer solo la clave principal para reducir
    # el uso de memoria durante la validación.
    try:
        df = pd.read_parquet(
            ruta,
            columns=["movie_id"],
        )
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise ValueError(
            "Gold: no se pudo leer movie_id "
            "desde dim_movie."
        ) from exc

    if df.empty:
        raise ValueError(
            "Gold: dim_movie no contiene películas."
        )

    movie_ids = df["movie_id"]

    if movie_ids.isna().any():
        raise ValueError(
            "Gold: dim_movie contiene movie_id nulos."
        )

    if not is_integer_dtype(movie_ids.dtype):
        raise ValueError(
            "Gold: movie_id debe tener tipo entero."
        )

    if (movie_ids <= 0).any():
        raise ValueError(
            "Gold: dim_movie contiene movie_id "
            "no positivos."
        )

    if movie_ids.duplicated().any():
        raise ValueError(
            "Gold: dim_movie contiene movie_id duplicados."
        )

    return {
        "dataset": "dim_movie",
        "estado": "validated",
        "total_peliculas": len(df),
        "movie_ids_unicos": movie_ids.nunique(),
        "nulos": 0,
        "duplicados": 0,
    }


def validate_gold_fact_rating_users_files(
    dim_user_path: Path,
    fact_rating_paths: list[Path],
) -> dict:
    """
    Comprueba que todos los usuarios de fact_rating
    existan en dim_user, leyendo archivos Parquet.

    No modifica datos.
    """

    if not fact_rating_paths:
        raise ValueError(
            "Gold: fact_rating no tiene particiones."
        )

    usuarios = pd.read_parquet(
        dim_user_path,
        columns=["user_id"],
    )

    if usuarios["user_id"].isna().any():
        raise ValueError(
            "Gold: dim_user contiene usuarios nulos."
        )

    if usuarios["user_id"].duplicated().any():
        raise ValueError(
            "Gold: dim_user contiene usuarios duplicados."
        )

    usuarios_validos = set(
        usuarios["user_id"].tolist()
    )

    total_ratings = 0

    for ruta in fact_rating_paths:

        ratings = pd.read_parquet(
            ruta,
            columns=["user_id"],
        )

        if ratings["user_id"].isna().any():
            raise ValueError(
                "Gold: fact_rating contiene usuarios nulos."
            )

        usuarios_desconocidos = (
            set(ratings["user_id"].tolist())
            - usuarios_validos
        )

        if usuarios_desconocidos:
            raise ValueError(
                "Gold: fact_rating contiene usuarios "
                "sin correspondencia en dim_user."
            )

        total_ratings += len(ratings)

    return {
        "relacion": "fact_rating.user_id -> dim_user.user_id",
        "estado": "validated",
        "total_usuarios": len(usuarios_validos),
        "total_ratings": total_ratings,
        "usuarios_huerfanos": 0,
    }


def validate_gold_dim_fact_files(
    rutas: dict,
) -> dict:
    """
    Valida las seis relaciones DIM/FACT leyendo
    los archivos Parquet físicos de Gold.

    Comprueba:
    - Claves primarias no nulas y únicas.
    - Claves foráneas no nulas.
    - Correspondencia con las dimensiones.

    No modifica los archivos.
    """

    relaciones = {
        "fact_rating": {
            "user_id": "dim_user",
            "movie_id": "dim_movie",
            "date_key": "dim_date",
        },
        "fact_availability": {
            "movie_id": "dim_movie",
            "provider_id": "dim_platform",
            "date_key": "dim_date",
        },
    }

    dimensiones = {
        "dim_user": "user_id",
        "dim_movie": "movie_id",
        "dim_platform": "provider_id",
        "dim_date": "date_key",
    }

    claves_validas = {}

    # 1. Validar las claves de las dimensiones.
    for dimension, clave in dimensiones.items():

        df = pd.read_parquet(
            rutas[dimension],
            columns=[clave],
        )

        if df[clave].isna().any():
            raise ValueError(
                f"Gold: {dimension} contiene "
                f"{clave} nulos."
            )

        if df[clave].duplicated().any():
            raise ValueError(
                f"Gold: {dimension} contiene "
                f"{clave} duplicados."
            )

        claves_validas[dimension] = set(
            df[clave].tolist()
        )

    # 2. Validar las referencias de los hechos.
    relaciones_validadas = []

    for hecho, referencias in relaciones.items():

        origen = rutas[hecho]

        archivos = (
            origen
            if isinstance(origen, list)
            else [origen]
        )

        if not archivos:
            raise ValueError(
                f"Gold: {hecho} no tiene archivos."
            )

        for clave, dimension in referencias.items():

            valores_dimension = claves_validas[dimension]
            registros_revisados = 0

            for archivo in archivos:

                df = pd.read_parquet(
                    archivo,
                    columns=[clave],
                )

                if df[clave].isna().any():
                    raise ValueError(
                        f"Gold: {hecho}.{clave} "
                        "contiene valores nulos."
                    )

                desconocidos = (
                    set(df[clave].tolist())
                    - valores_dimension
                )

                if desconocidos:
                    raise ValueError(
                        f"Gold: {hecho}.{clave} "
                        f"contiene referencias inexistentes "
                        f"en {dimension}.{clave}."
                    )

                registros_revisados += len(df)

            relaciones_validadas.append({
                "hecho": hecho,
                "clave": clave,
                "dimension": dimension,
                "registros_revisados": registros_revisados,
                "referencias_huerfanas": 0,
            })

    return {
        "estado": "validated",
        "total_relaciones": len(relaciones_validadas),
        "relaciones": relaciones_validadas,
    }
