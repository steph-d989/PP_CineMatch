
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import BRONZE_MOVIELENS_DIR


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
