
from pathlib import Path
from time import perf_counter

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
    evaluate_hybrid_hit_rate,
)


EXPERIMENT_DIR = (
    Path("data")
    / "experiments"
    / "movielens_32m"
)


def evaluate_real_hybrid(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    catalog: pd.DataFrame,
    n_factors: int = 50,
    content_weight: float = 0.5,
    collaborative_weight: float = 0.5,
    k_values=(5, 10, 20),
) -> dict:
    """
    Entrena Content-Based y Collaborative
    exclusivamente con Train.

    Evalúa Hybrid RRF sobre Validation.

    Test permanece reservado.
    """

    start = perf_counter()

    # ----------------------------------
    # 1. Características del catálogo
    # ----------------------------------

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

    # ----------------------------------
    # 2. Modelo Content-Based
    # ----------------------------------

    content_model = fit_content_based_model(
        content_features=features,
        min_df=1,
    )

    # ----------------------------------
    # 3. Modelo Collaborative
    # ----------------------------------

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

    training_seconds = perf_counter() - start

    # ----------------------------------
    # 4. Evaluación Hybrid
    # ----------------------------------

    resultados = {}

    for k in k_values:

        resultados[k] = evaluate_hybrid_hit_rate(
            train=train,
            test=validation,
            content_model=content_model,
            collaborative_model=collaborative_model,
            k=k,
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
        )

    return {
        "total_users": len(matrix_data["user_ids"]),
        "total_movies_content": content_model[
            "total_movies"
        ],
        "total_features_content": content_model[
            "total_features"
        ],
        "total_movies_collaborative": len(
            matrix_data["movie_ids"]
        ),
        "n_factors": collaborative_model[
            "n_factors"
        ],
        "content_weight": content_weight,
        "collaborative_weight": collaborative_weight,
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

    catalog = pd.read_parquet(
        EXPERIMENT_DIR / "catalog_sample.parquet"
    )

    print("\nCINEMATCH - HYBRID REAL")
    print("--------------------------------")

    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Catálogo:", len(catalog))

    resultado = evaluate_real_hybrid(
        train=train,
        validation=validation,
        n_factors=50,
        catalog=catalog,
    )

    print("\nMODELOS ENTRENADOS")
    print(
        "Películas Content:",
        resultado["total_movies_content"]
    )
    print(
        "Características TF-IDF:",
        resultado["total_features_content"]
    )
    print(
        "Películas Collaborative:",
        resultado["total_movies_collaborative"]
    )
    print(
        "Factores SVD:",
        resultado["n_factors"]
    )
    print(
        "Pesos RRF:",
        resultado["content_weight"],
        resultado["collaborative_weight"],
    )
    print(
        "Tiempo entrenamiento:",
        f"{resultado['training_seconds']:.2f} s"
    )

    for k, evaluacion in resultado[
        "evaluaciones"
    ].items():

        print(f"\nRESULTADO HIT RATE@{k}")

        print(
            "Usuarios relevantes:",
            evaluacion["usuarios_relevantes"]
        )
        print(
            "Aciertos:",
            evaluacion["aciertos"]
        )
        print(
            "Hit Rate global:",
            f"{evaluacion['hit_rate_global']:.2%}"
        )

        cobertura = evaluacion.get(
            "cobertura_recomendaciones"
        )

        if cobertura is not None:
            print(
                "Cobertura de recomendaciones:",
                f"{cobertura:.2%}"
            )


if __name__ == "__main__":
    main()
