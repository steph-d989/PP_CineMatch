
from pathlib import Path

import pandas as pd

from src.models.content_based import (
    fit_content_based_model,
)
from src.models.collaborative import (
    prepare_collaborative_ratings,
    build_user_movie_matrix,
    fit_collaborative_svd,
)
from src.models.evaluate import (
    select_hybrid_weights,
)


EXPERIMENT_DIR = (
    Path("data")
    / "experiments"
    / "movielens_32m"
)


def tune_real_hybrid(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    catalog: pd.DataFrame,
    n_factors: int = 50,
    k: int = 10,
) -> dict:
    """
    Selecciona pesos del Hybrid con Validation.

    Los modelos se ajustan usando Train.
    No utiliza ni lee Test.
    """

    features = catalog[
        ["movie_id", "title", "genres"]
    ].copy()

    features["genres"] = features["genres"].apply(
        lambda value: [
            genre.strip()
            for genre in value.split("|")
            if genre.strip()
            and genre.strip() != "(no genres listed)"
        ]
        if isinstance(value, str)
        else []
    )

    content_model = fit_content_based_model(
        content_features=features,
        min_df=1,
    )

    prepared = prepare_collaborative_ratings(
        train
    )

    matrix_data = build_user_movie_matrix(
        prepared
    )

    collaborative_model = fit_collaborative_svd(
        matrix_data=matrix_data,
        n_factors=n_factors,
        random_state=42,
    )

    candidates = [
        (0.00, 1.00),
        (0.25, 0.75),
        (0.50, 0.50),
        (0.75, 0.25),
        (1.00, 0.00),
    ]

    seleccion = select_hybrid_weights(
        train=train,
        validation=validation,
        content_model=content_model,
        collaborative_model=collaborative_model,
        k=k,
        candidates=candidates,
    )

    return seleccion


def main():

    train = pd.read_parquet(
        EXPERIMENT_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        EXPERIMENT_DIR / "validation.parquet"
    )

    catalog = pd.read_parquet(
        EXPERIMENT_DIR / "catalog_sample.parquet"
    )

    print("\nCINEMATCH - OPTIMIZACION RRF")
    print("--------------------------------")

    resultado = tune_real_hybrid(
        train=train,
        validation=validation,
        catalog=catalog,
        n_factors=50,
        k=10,
    )

    print("\nRESULTADOS VALIDATION")

    for fila in resultado[
        "resultados_validation"
    ]:
        print(
            f"Content: {fila['content_weight']:.2f} | "
            f"Collaborative: "
            f"{fila['collaborative_weight']:.2f} | "
            f"Hit Rate@10: "
            f"{fila['hit_rate_global']:.2%} | "
            f"Aciertos: {fila['aciertos']} | "
            f"Cobertura: {fila['cobertura']:.2%}"
        )

    mejor = resultado["mejor_configuracion"]

    print("\nMEJOR CONFIGURACION EN VALIDATION")
    print(
        "Peso Content:",
        mejor["content_weight"],
    )
    print(
        "Peso Collaborative:",
        mejor["collaborative_weight"],
    )
    print(
        "Hit Rate@10:",
        f"{mejor['hit_rate_global']:.2%}",
    )
    print(
        "Aciertos:",
        mejor["aciertos"],
    )


if __name__ == "__main__":
    main()
