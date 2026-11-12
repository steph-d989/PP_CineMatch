
from pathlib import Path

import pandas as pd

from src.models.evaluate import (
    evaluate_baseline_hit_rate,
)


EXPERIMENT_DIR = (
    Path("data")
    / "experiments"
    / "movielens_32m"
)


def evaluate_real_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    catalog: pd.DataFrame,
    k: int = 10,
) -> dict:
    """
    Evalúa Baseline usando solamente
    Train y Validation.

    Test queda reservado para la
    evaluación final.
    """

    return evaluate_baseline_hit_rate(
        train=train,
        test=validation,
        dim_movie=catalog,
        k=k,
    )


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

    print("\nCINEMATCH - BASELINE REAL")
    print("--------------------------------")

    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Catálogo:", len(catalog))

    for k in (5, 10, 20):

        resultado = evaluate_real_baseline(
            train=train,
            validation=validation,
            catalog=catalog,
            k=k,
        )

        print(f"\nRESULTADO HIT RATE@{k}")
        print(
            "Usuarios relevantes:",
            resultado["usuarios_relevantes"],
        )
        print(
            "Aciertos:",
            resultado["aciertos"],
        )
        print(
            "Hit Rate global:",
            f"{resultado['hit_rate_global']:.2%}",
        )
        
        cobertura = resultado.get(
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
                "No disponible en este evaluador",
            )



if __name__ == "__main__":
    main()
