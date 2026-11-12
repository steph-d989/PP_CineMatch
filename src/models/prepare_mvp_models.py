
from pathlib import Path

import pandas as pd

from src.models.content_based import (
    fit_content_based_model,
    save_content_based_model,
    load_content_based_model,
)

from src.models.collaborative import (
    prepare_collaborative_ratings,
    build_user_movie_matrix,
    fit_collaborative_svd,
    save_collaborative_model,
    load_collaborative_model,
)


EXPERIMENT_DIR = (
    Path("data")
    / "experiments"
    / "movielens_32m"
)

MODEL_RUN_ID = "6f7b4d21-8e30-4af1-9c62-31d5e9b84720"


def prepare_and_save_mvp_models(
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    output_dir,
    run_id: str = MODEL_RUN_ID,
    n_factors: int = 50,
) -> dict:
    """
    Entrena y persiste los modelos del MVP.

    - Content-Based: TF-IDF con géneros MovieLens.
    - Collaborative: TruncatedSVD con Train.
    - No utiliza Validation ni Test.
    - No sobrescribe versiones existentes.
    - Verifica los artefactos después de guardarlos.
    """

    output_dir = Path(output_dir)

    content_dir = output_dir / "content_based"
    collaborative_dir = output_dir / "collaborative"

    if content_dir.exists() or collaborative_dir.exists():
        raise FileExistsError(
            "Ya existen modelos para esta versión. "
            "Utiliza un nuevo run_id."
        )


    from uuid import UUID

    try:
        run_id = str(UUID(run_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(
            "run_id debe ser un UUID válido."
        ) from exc


    # 1. Preparar atributos de las películas.
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

    # 2. Entrenar Content-Based.
    content_model = fit_content_based_model(
        content_features=features,
        min_df=1,
    )

    # 3. Entrenar Collaborative únicamente con Train.
    prepared = prepare_collaborative_ratings(train)

    matrix_data = build_user_movie_matrix(prepared)

    collaborative_model = fit_collaborative_svd(
        matrix_data=matrix_data,
        n_factors=n_factors,
        random_state=42,
    )

    # 4. Guardar ambos modelos.
    output_dir.mkdir(parents=True, exist_ok=True)

    save_content_based_model(
        content_model,
        output_dir=content_dir,
        run_id=run_id,
    )

    save_collaborative_model(
        collaborative_model,
        output_dir=collaborative_dir,
        run_id=run_id,
    )

    # 5. Verificar la carga y los manifiestos.
    # joblib solo debe cargarse desde fuentes confiables.
    loaded_content = load_content_based_model(
        content_dir,
        trusted_source=True,
        expected_run_id=run_id,
    )

    loaded_collaborative = load_collaborative_model(
        collaborative_dir,
        expected_run_id=run_id,
    )

    return {
        "run_id": run_id,
        "content_dir": str(content_dir),
        "collaborative_dir": str(collaborative_dir),
        "content_movies": loaded_content["total_movies"],
        "content_features": loaded_content["total_features"],
        "collaborative_users": len(
            loaded_collaborative["user_ids"]
        ),
        "collaborative_movies": len(
            loaded_collaborative["movie_ids"]
        ),
        "n_factors": loaded_collaborative["n_factors"],
        "verification": "OK",
    }


def main():

    train = pd.read_parquet(
        EXPERIMENT_DIR / "train.parquet"
    )

    catalog = pd.read_parquet(
        EXPERIMENT_DIR / "catalog_sample.parquet"
    )

    output_dir = (
        EXPERIMENT_DIR
        / "models"
        / MODEL_RUN_ID
    )

    print("\nCINEMATCH - MODELOS MVP")
    print("------------------------------")

    resultado = prepare_and_save_mvp_models(
        train=train,
        catalog=catalog,
        output_dir=output_dir,
        run_id=MODEL_RUN_ID,
        n_factors=50,
    )

    for clave, valor in resultado.items():
        print(f"{clave}: {valor}")


if __name__ == "__main__":
    main()
