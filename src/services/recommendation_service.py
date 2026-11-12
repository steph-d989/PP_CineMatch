
import pandas as pd

from src.models.content_based import (
    recommend_content_for_user,
)

from src.models.collaborative import (
    recommend_collaborative_top_n,
)


def recommend_movies_for_user(
    user_id: int,
    algorithm: str,
    n: int,
    train: pd.DataFrame,
    catalog: pd.DataFrame,
    content_model: dict,
    collaborative_model: dict,
) -> pd.DataFrame:
    """
    Servicio reutilizable de recomendaciones.

    Utiliza modelos previamente entrenados.

    No entrena, no consulta APIs y no modifica
    los datos originales.

    Algoritmos disponibles:
      - collaborative
      - content
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

    if algorithm not in {
        "collaborative",
        "content",
    }:
        raise ValueError(
            "Algoritmo no disponible."
        )

    columnas_train = {
        "user_id",
        "movie_id",
        "rating",
    }

    columnas_catalog = {
        "movie_id",
        "title",
        "genres",
        "tmdb_id",
    }

    if not columnas_train.issubset(train.columns):
        raise ValueError(
            "Train no contiene las columnas requeridas."
        )

    if not columnas_catalog.issubset(catalog.columns):
        raise ValueError(
            "Catálogo incompleto."
        )

    if catalog["movie_id"].duplicated().any():
        raise ValueError(
            "El catálogo contiene movie_id duplicados."
        )

    historial = train.loc[
        train["user_id"] == user_id
    ]

    if historial.empty:
        raise ValueError(
            "El usuario no tiene historial en Train."
        )

    # ---------------------------------------
    # 1. Generar ranking con el modelo elegido
    # ---------------------------------------

    if algorithm == "collaborative":

        ranking = recommend_collaborative_top_n(
            fitted_model=collaborative_model,
            user_id=user_id,
            train_interactions=train,
            n=n,
        )

        score_column = "collaborative_score"

    else:

        ranking = recommend_content_for_user(
            user_id=user_id,
            interactions=train,
            content_df=content_model["content_df"],
            tfidf_matrix=content_model["tfidf_matrix"],
            n=n,
        )

        score_column = "content_score"

    # ---------------------------------------
    # 2. Columnas estandarizadas
    # ---------------------------------------

    columnas_salida = [
        "movie_id",
        "title",
        "genres",
        "tmdb_id",
        "score",
        "algorithm",
    ]

    if ranking.empty:
        return pd.DataFrame(
            columns=columnas_salida
        )

    if (
        "movie_id" not in ranking.columns
        or score_column not in ranking.columns
    ):
        raise ValueError(
            "El modelo devolvió un ranking inválido."
        )

    ranking = ranking[
        ["movie_id", score_column]
    ].copy()

    ranking = ranking.rename(
        columns={
            score_column: "score",
        }
    )

    # ---------------------------------------
    # 3. Excluir películas ya calificadas
    # ---------------------------------------

    vistos = set(historial["movie_id"])

    ranking = ranking.loc[
        ~ranking["movie_id"].isin(vistos)
    ].copy()

    if ranking["movie_id"].duplicated().any():
        raise ValueError(
            "El modelo devolvió películas duplicadas."
        )

    # ---------------------------------------
    # 4. Incorporar catálogo
    # ---------------------------------------

    metadata = catalog[
        ["movie_id", "title", "genres", "tmdb_id"]
    ].copy()

    resultado = ranking.merge(
        metadata,
        on="movie_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if (resultado["_merge"] != "both").any():
        raise ValueError(
            "Hay recomendaciones sin película en el catálogo."
        )

    resultado = resultado.drop(columns="_merge")

    # Mantener el orden generado por el modelo.
    resultado["algorithm"] = algorithm

    resultado = resultado.head(n)

    return resultado[
        columnas_salida
    ].reset_index(drop=True)
