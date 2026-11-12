from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import BRONZE_MOVIELENS_DIR

from src.models.evaluate import (
    temporal_train_validation_test_split,
)

def sample_movielens_ratings(
    ratings_path,
    sample_users=300,
    min_ratings=5,
    seed=42,
    chunksize=250_000,
):
    """
    Selecciona usuarios de MovieLens de forma reproducible.

    Conserva todas las calificaciones de los
    usuarios seleccionados.

    Lee ratings.csv por bloques para evitar
    cargar el archivo completo en memoria.
    """

    ratings_path = Path(ratings_path)

    if not ratings_path.is_file():
        raise FileNotFoundError(ratings_path)

    for nombre, valor in {
        "sample_users": sample_users,
        "min_ratings": min_ratings,
        "chunksize": chunksize,
    }.items():
        if (
            isinstance(valor, bool)
            or not isinstance(valor, int)
            or valor <= 0
        ):
            raise ValueError(
                f"{nombre} debe ser un entero positivo."
            )

    # Primera lectura: identificar usuarios.
    usuarios = set()

    for bloque in pd.read_csv(
        ratings_path,
        usecols=["userId"],
        chunksize=chunksize,
    ):
        usuarios.update(
            bloque["userId"].unique().tolist()
        )

    if not usuarios:
        raise ValueError(
            "El archivo no contiene usuarios."
        )

    rng = np.random.default_rng(seed)

    usuarios_seleccionados = set(
        rng.choice(
            sorted(usuarios),
            size=min(sample_users, len(usuarios)),
            replace=False,
        ).tolist()
    )

    # Segunda lectura: obtener solamente
    # las interacciones de esos usuarios.
    bloques_seleccionados = []

    for bloque in pd.read_csv(
        ratings_path,
        usecols=[
            "userId",
            "movieId",
            "rating",
            "timestamp",
        ],
        chunksize=chunksize,
    ):
        seleccion = bloque.loc[
            bloque["userId"].isin(
                usuarios_seleccionados
            )
        ]

        if not seleccion.empty:
            bloques_seleccionados.append(
                seleccion.copy()
            )

    if not bloques_seleccionados:
        raise ValueError(
            "No se encontraron interacciones."
        )

    muestra = pd.concat(
        bloques_seleccionados,
        ignore_index=True,
    )

    # Descartar usuarios con pocas interacciones.
    cantidades = muestra.groupby(
        "userId"
    )["movieId"].transform("size")

    muestra = muestra.loc[
        cantidades >= min_ratings
    ].copy()

    if muestra.empty:
        raise ValueError(
            "Ningún usuario cumple el mínimo "
            "de interacciones."
        )

    muestra = muestra.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
        }
    )

    muestra["rated_at"] = pd.to_datetime(
        muestra["timestamp"],
        unit="s",
        utc=True,
    )

    muestra = muestra[
        ["user_id", "movie_id", "rating", "rated_at"]
    ]

    muestra = muestra.sort_values(
        ["user_id", "rated_at", "movie_id"],
        kind="stable",
    ).reset_index(drop=True)

    if muestra.duplicated(
        ["user_id", "movie_id"]
    ).any():
        raise ValueError(
            "Se encontraron interacciones duplicadas."
        )

    return muestra

def build_movielens_sample_catalog(
    sample_ratings: pd.DataFrame,
    movies_path,
    links_path,
) -> pd.DataFrame:
    """
    Construye un catálogo con las películas
    presentes en una muestra de ratings.

    Conserva las películas sin tmdb_id.
    Rechaza identificadores ambiguos o
    películas ausentes de movies.csv.
    """

    if (
        "movie_id" not in sample_ratings.columns
        or sample_ratings["movie_id"].isna().any()
        or sample_ratings.empty
    ):
        raise ValueError(
            "La muestra requiere movie_id válido."
        )

    movies = pd.read_csv(
        movies_path,
        usecols=["movieId", "title", "genres"],
    ).rename(columns={"movieId": "movie_id"})

    links = pd.read_csv(
        links_path,
        usecols=["movieId", "imdbId", "tmdbId"],
        dtype={"imdbId": "string"},
    ).rename(
        columns={
            "movieId": "movie_id",
            "imdbId": "imdb_id",
            "tmdbId": "tmdb_id",
        }
    )

    if movies["movie_id"].duplicated().any():
        raise ValueError(
            "movies.csv contiene movie_id duplicados."
        )

    if links["movie_id"].duplicated().any():
        raise ValueError(
            "links.csv contiene movie_id duplicados."
        )

    ids = pd.DataFrame({
        "movie_id": sorted(
            sample_ratings["movie_id"].unique()
        )
    })

    catalog = ids.merge(
        movies,
        on="movie_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if (catalog["_merge"] != "both").any():
        raise ValueError(
            "Existen películas de la muestra "
            "que no aparecen en movies.csv."
        )

    catalog = catalog.drop(columns="_merge")

    catalog = catalog.merge(
        links,
        on="movie_id",
        how="left",
        validate="one_to_one",
    )

    catalog["tmdb_id"] = pd.to_numeric(
        catalog["tmdb_id"],
        errors="coerce",
    ).astype("Int64")

    return catalog[
        [
            "movie_id",
            "title",
            "genres",
            "imdb_id",
            "tmdb_id",
        ]
    ].sort_values(
        "movie_id"
    ).reset_index(drop=True)

def prepare_real_evaluation_split(
    sample_ratings: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict:
    """
    Prepara Train/Validation/Test y diagnostica
    cobertura de películas en entrenamiento.

    No elimina películas nuevas de Validation
    ni Test: informa su cantidad para evaluar
    correctamente el cold start.
    """

    if sample_ratings.empty or catalog.empty:
        raise ValueError(
            "La muestra y el catálogo no pueden estar vacíos."
        )

    if (
        "movie_id" not in sample_ratings.columns
        or "movie_id" not in catalog.columns
    ):
        raise ValueError(
            "Ambos datasets requieren movie_id."
        )

    if catalog["movie_id"].isna().any():
        raise ValueError(
            "El catálogo contiene identificadores nulos."
        )

    if catalog["movie_id"].duplicated().any():
        raise ValueError(
            "El catálogo contiene películas duplicadas."
        )

    ids_catalogo = set(catalog["movie_id"])

    if not set(sample_ratings["movie_id"]).issubset(
        ids_catalogo
    ):
        raise ValueError(
            "Existen ratings sin película en el catálogo."
        )

    train, validation, test = (
        temporal_train_validation_test_split(
            sample_ratings
        )
    )

    peliculas_train = set(train["movie_id"])
    usuarios_train = set(train["user_id"])

    def diagnostico(df):

        conocidos = df["movie_id"].isin(
            peliculas_train
        )

        usuarios_conocidos = df["user_id"].isin(
            usuarios_train
        )

        return {
            "interacciones": len(df),
            "usuarios": df["user_id"].nunique(),
            "peliculas": df["movie_id"].nunique(),
            "peliculas_conocidas_interacciones": int(
                conocidos.sum()
            ),
            "peliculas_nuevas_interacciones": int(
                (~conocidos).sum()
            ),
            "cobertura_peliculas_train": float(
                conocidos.mean()
            ),
            "usuarios_nuevos_interacciones": int(
                (~usuarios_conocidos).sum()
            ),
            "interacciones_relevantes": int(
                (df["rating"] >= 4.0).sum()
            ),
        }

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "diagnosticos": {
            "train": diagnostico(train),
            "validation": diagnostico(validation),
            "test": diagnostico(test),
        },
    }


if __name__ == "__main__":

    origen = (
        BRONZE_MOVIELENS_DIR
        / "ml-32m"
        / "ratings.csv"
    )

    destino = (
        Path("data")
        / "experiments"
        / "movielens_32m"
        / "ratings_sample.parquet"
    )

    muestra = sample_movielens_ratings(
        ratings_path=origen,
        sample_users=300,
        min_ratings=5,
        seed=42,
    )

    destino.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    muestra.to_parquet(
        destino,
        index=False,
    )

    print("\nMUESTRA REAL DE MOVIELENS 32M")
    print("------------------------------")
    print("Usuarios:", muestra["user_id"].nunique())
    print("Películas:", muestra["movie_id"].nunique())
    print("Interacciones:", len(muestra))
    print("Rating medio:", round(muestra["rating"].mean(), 3))
    print("Archivo:", destino)


    catalog = build_movielens_sample_catalog(
        sample_ratings=muestra,
        movies_path=(
            BRONZE_MOVIELENS_DIR
            / "ml-32m"
            / "movies.csv"
        ),
        links_path=(
            BRONZE_MOVIELENS_DIR
            / "ml-32m"
            / "links.csv"
        ),
    )

    catalog_path = (
        destino.parent / "catalog_sample.parquet"
    )

    catalog.to_parquet(
        catalog_path,
        index=False,
    )

    
    experimento = prepare_real_evaluation_split(
        sample_ratings=muestra,
        catalog=catalog,
    )

    print("\nDIVISIÓN TEMPORAL REAL")
    print("------------------------------")

    for nombre in ("train", "validation", "test"):

        datos = experimento[nombre]
        diagnostico = experimento[
            "diagnosticos"
        ][nombre]

        archivo = destino.parent / f"{nombre}.parquet"

        datos.to_parquet(
            archivo,
            index=False,
        )

        print(f"\n{nombre.upper()}")
        print("Interacciones:", len(datos))
        print("Usuarios:", diagnostico["usuarios"])
        print("Películas:", diagnostico["peliculas"])
        print(
            "Películas conocidas en Train:",
            f"{diagnostico['cobertura_peliculas_train']:.2%}"
        )
        print(
            "Interacciones relevantes:",
            diagnostico["interacciones_relevantes"]
        )
        print("Archivo:", archivo)

    
    total = len(catalog)
    con_tmdb = catalog["tmdb_id"].notna().sum()
    sin_tmdb = total - con_tmdb

    print("\nCATÁLOGO INTEGRADO")
    print("------------------------------")
    print("Películas:", total)
    print("Con TMDb ID:", con_tmdb)
    print("Sin TMDb ID:", sin_tmdb)
    print(
        "Cobertura TMDb:",
        f"{100 * con_tmdb / total:.2f}%"
    )
    print("Archivo:", catalog_path)

