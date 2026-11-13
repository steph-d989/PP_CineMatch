
from pathlib import Path

import pandas as pd

from src.models.content_based import (
    load_content_based_model,
)
from src.models.collaborative import (
    load_collaborative_model,
)
from src.models.prepare_mvp_models import (
    EXPERIMENT_DIR,
    MODEL_RUN_ID,
)


def load_mvp_context(
    experiment_dir=EXPERIMENT_DIR,
    run_id=MODEL_RUN_ID,
) -> dict:
    """
    Carga el contexto necesario para recomendar.

    No entrena modelos.
    No consulta APIs.
    No utiliza Validation ni Test.

    Los modelos deben proceder de artefactos
    locales confiables y verificables.
    """

    base = Path(experiment_dir)

    train = pd.read_parquet(
        base / "train.parquet"
    )

    catalog = pd.read_parquet(
        base / "catalog_sample.parquet"
    )

    models_dir = base / "models" / run_id

    content_model = load_content_based_model(
        models_dir / "content_based",
        trusted_source=True,
        expected_run_id=run_id,
    )

    collaborative_model = load_collaborative_model(
        models_dir / "collaborative",
        expected_run_id=run_id,
    )

    if train.empty or catalog.empty:
        raise ValueError(
            "Train y catálogo deben contener datos."
        )

    if catalog["movie_id"].duplicated().any():
        raise ValueError(
            "El catálogo contiene películas duplicadas."
        )

    if not set(train["movie_id"]).issubset(
        set(catalog["movie_id"])
    ):
        raise ValueError(
            "Train contiene películas fuera del catálogo."
        )

    # Comprobar que el artefacto colaborativo
    # corresponde a los usuarios de Train.
    if set(train["user_id"]) != set(
        collaborative_model["user_ids"]
    ):
        raise ValueError(
            "Los usuarios del modelo no coinciden "
            "con Train."
        )

    if set(train["movie_id"]) != set(
        collaborative_model["movie_ids"]
    ):
        raise ValueError(
            "Las películas del modelo no coinciden "
            "con Train."
        )

    return {
        "train": train,
        "catalog": catalog,
        "content_model": content_model,
        "collaborative_model": collaborative_model,
        "run_id": run_id,
    }


def get_user_history(
    user_id: int,
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Devuelve películas calificadas por el usuario.

    Utiliza exclusivamente el historial de Train.
    """

    if (
        isinstance(user_id, bool)
        or not isinstance(user_id, int)
        or user_id <= 0
    ):
        raise ValueError(
            "user_id debe ser un entero positivo."
        )

    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
    ):
        raise ValueError(
            "n debe ser un entero positivo."
        )

    columnas = [
        "movie_id",
        "title",
        "genres",
        "tmdb_id",
        "rating",
        "rated_at",
    ]

    historial = train.loc[
        train["user_id"] == user_id
    ].copy()

    if historial.empty:
        return pd.DataFrame(columns=columnas)

    historial = historial.sort_values(
        ["rated_at", "movie_id"],
        ascending=[False, True],
    ).head(n)

    resultado = historial.merge(
        catalog[
            ["movie_id", "title", "genres", "tmdb_id"]
        ],
        on="movie_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if (resultado["_merge"] != "both").any():
        raise ValueError(
            "Existen valoraciones sin película."
        )

    return resultado[
        columnas
    ].reset_index(drop=True)


def main():

    from src.services.recommendation_service import (
        recommend_movies_for_user,
    )

    context = load_mvp_context()

    user_id = int(
        sorted(context["train"]["user_id"].unique())[0]
    )

    print("\nCINEMATCH - PRUEBA REAL MVP")
    print("--------------------------------")
    print("Run ID:", context["run_id"])
    print("Usuario:", user_id)

    historial = get_user_history(
        user_id=user_id,
        train=context["train"],
        catalog=context["catalog"],
        n=5,
    )

    print("\nULTIMAS PELICULAS CALIFICADAS")
    print(
        historial[
            ["title", "rating"]
        ].to_string(index=False)
    )

    for algorithm in ("collaborative", "content"):

        recomendaciones = recommend_movies_for_user(
            user_id=user_id,
            algorithm=algorithm,
            n=5,
            train=context["train"],
            catalog=context["catalog"],
            content_model=context["content_model"],
            collaborative_model=context[
                "collaborative_model"
            ],
        )

        print(f"\nRECOMENDACIONES {algorithm.upper()}")

        if recomendaciones.empty:
            print("No se encontraron recomendaciones.")
        else:
            print(
                recomendaciones[
                    ["title", "score"]
                ].to_string(index=False)
            )

        # No recomendar ninguna película que
        # el usuario haya valorado en Train.
        vistos = set(
            context["train"].loc[
                context["train"]["user_id"] == user_id,
                "movie_id",
            ]
        )

        assert set(
            recomendaciones["movie_id"]
        ).isdisjoint(vistos)


if __name__ == "__main__":
    main()
