
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.models.collaborative import (
    prepare_collaborative_ratings,
    build_user_movie_matrix,
    fit_collaborative_svd,
)

from src.models.evaluate import (
    evaluate_collaborative_hit_rate,
)


EXPERIMENT_DIR = (
    Path("data")
    / "experiments"
    / "movielens_32m"
)


def evaluate_real_collaborative(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    n_factors: int = 50,
    k_values=(5, 10, 20),
) -> dict:
    """
    Entrena TruncatedSVD exclusivamente con Train.

    Evalúa sobre Validation.
    No utiliza Test ni modifica los Parquet.
    """

    start = perf_counter()

    prepared = prepare_collaborative_ratings(
        train
    )

    matrix_data = build_user_movie_matrix(
        prepared
    )

    fitted_model = fit_collaborative_svd(
        matrix_data=matrix_data,
        n_factors=n_factors,
        random_state=42,
    )

    training_seconds = perf_counter() - start

    resultados = {}

    for k in k_values:

        resultados[k] = (
            evaluate_collaborative_hit_rate(
                train=train,
                test=validation,
                fitted_model=fitted_model,
                k=k,
            )
        )

    return {
        "total_users": len(
            matrix_data["user_ids"]
        ),
        "total_movies": len(
            matrix_data["movie_ids"]
        ),
        "total_interactions": len(prepared),
        "matrix_shape": matrix_data["matrix"].shape,
        "matrix_nnz": matrix_data["matrix"].nnz,
        "n_factors": fitted_model["n_factors"],
        "training_seconds": training_seconds,
        "evaluaciones": resultados,
    }


def main():

    train = pd.read_parquet(
        EXPERIMENT_DIR / "train.parquet"
    )

    validation = pd.read_parquet(
        EXPERIMENT_DIR / "validation.parquet"
    )

    print("\nCINEMATCH - COLLABORATIVE REAL")
    print("--------------------------------")

    print("Train:", len(train))
    print("Validation:", len(validation))

    resultado = evaluate_real_collaborative(
        train=train,
        validation=validation,
        n_factors=50,
    )

    print("\nMODELO ENTRENADO")
    print(
        "Usuarios:",
        resultado["total_users"],
    )
    print(
        "Películas:",
        resultado["total_movies"],
    )
    print(
        "Interacciones:",
        resultado["total_interactions"],
    )
    print(
        "Dimensión matriz:",
        resultado["matrix_shape"],
    )
    print(
        "Valores almacenados:",
        resultado["matrix_nnz"],
    )
    print(
        "Factores latentes:",
        resultado["n_factors"],
    )
    print(
        "Tiempo entrenamiento:",
        f"{resultado['training_seconds']:.2f} s",
    )

    for k, evaluacion in resultado[
        "evaluaciones"
    ].items():

        print(f"\nRESULTADO HIT RATE@{k}")

        print(
            "Usuarios relevantes evaluables:",
            evaluacion["usuarios_relevantes"],
        )

        print(
            "Aciertos:",
            evaluacion["aciertos"],
        )

        hit_rate = evaluacion["hit_rate_global"]

        print(
            "Hit Rate global:",
            f"{hit_rate:.2%}",
        )

        cobertura = evaluacion.get(
            "cobertura_recomendaciones"
        )

        if cobertura is not None:
            print(
                "Cobertura reportada:",
                f"{cobertura:.2%}",
            )
        else:
            print(
                "Cobertura reportada:",
                "No disponible",
            )


if __name__ == "__main__":
    main()
