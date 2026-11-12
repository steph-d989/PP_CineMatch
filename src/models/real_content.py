
from pathlib import Path

import pandas as pd

from src.models.content_based import (
    fit_content_based_model,
)

from src.models.evaluate import (
    evaluate_content_hit_rate,
)


EXPERIMENT_DIR = (
    Path("data")
    / "experiments"
    / "movielens_32m"
)


def evaluate_real_content(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    catalog: pd.DataFrame,
    k_values=(5, 10, 20),
) -> dict:
    """
    Entrena Content-Based con los géneros
    del catálogo MovieLens.

    Los perfiles de usuario se construyen
    exclusivamente desde Train.

    Validation se utiliza para evaluar.
    Test permanece reservado.
    """

    requeridas = {
        "movie_id",
        "title",
        "genres",
    }

    if not requeridas.issubset(catalog.columns):
        raise ValueError(
            "El catálogo requiere movie_id, "
            "title y genres."
        )

    features = catalog[
        ["movie_id", "title", "genres"]
    ].copy()

    # MovieLens utiliza géneros separados por |.
    # Convertimos esos valores a listas.
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

    modelo = fit_content_based_model(
        content_features=features,
        min_df=1,
    )

    resultados = {}

    for k in k_values:

        resultados[k] = evaluate_content_hit_rate(
            train=train,
            test=validation,
            content_df=modelo["content_df"],
            tfidf_matrix=modelo["tfidf_matrix"],
            k=k,
        )

    return {
        "total_movies": modelo["total_movies"],
        "total_features": modelo["total_features"],
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

    print("\nCINEMATCH - CONTENT-BASED REAL")
    print("--------------------------------")

    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Catálogo:", len(catalog))

    resultado = evaluate_real_content(
        train=train,
        validation=validation,
        catalog=catalog,
    )

    print(
        "Películas del modelo:",
        resultado["total_movies"],
    )
    print(
        "Características TF-IDF:",
        resultado["total_features"],
    )

    for k, evaluacion in resultado[
        "evaluaciones"
    ].items():

        print(f"\nRESULTADO HIT RATE@{k}")
        print(
            "Usuarios relevantes:",
            evaluacion["usuarios_relevantes"],
        )
        print(
            "Aciertos:",
            evaluacion["aciertos"],
        )
        print(
            "Hit Rate global:",
            f"{evaluacion['hit_rate_global']:.2%}",
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
