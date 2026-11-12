
"""
Recomendador híbrido de CineMatch.

Combina rankings Content-Based y Collaborative
mediante Reciprocal Rank Fusion (RRF).

No suma directamente puntuaciones incompatibles.
No entrena modelos ni consulta APIs.
"""

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

from src.models.content_based import (
    recommend_content_for_user,
)

from src.models.collaborative import (
    recommend_collaborative_top_n,
)

def fuse_hybrid_rankings(
    content_ranking: pd.DataFrame,
    collaborative_ranking: pd.DataFrame,
    n: int = 10,
    content_weight: float = 0.5,
    collaborative_weight: float = 0.5,
    rrf_k: int = 60,
    seen_movie_ids=None,
) -> pd.DataFrame:
    """
    Combina dos rankings de películas mediante RRF.

    Los DataFrames deben contener:
    - Content-Based: movie_id, content_score
    - Collaborative: movie_id, collaborative_score

    Devuelve un ranking híbrido reproducible.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n debe ser un entero positivo.")

    if (
        isinstance(rrf_k, bool)
        or not isinstance(rrf_k, int)
        or rrf_k < 1
    ):
        raise ValueError("rrf_k debe ser positivo.")

    pesos = (content_weight, collaborative_weight)

    if (
        any(
            isinstance(peso, (bool, np.bool_))
            or not isinstance(peso, (int, float, np.number))
            for peso in pesos
        )
        or not all(np.isfinite(peso) for peso in pesos)
        or any(peso < 0 for peso in pesos)
        or not np.isclose(sum(pesos), 1.0)
    ):
        raise ValueError(
            "Los pesos deben ser no negativos "
            "y sumar 1."
        )

    def preparar_ranking(datos, columna, nombre):

        requeridas = {"movie_id", columna}

        if not requeridas.issubset(datos.columns):
            raise ValueError(
                f"{nombre} contiene columnas faltantes."
            )

        df = datos[["movie_id", columna]].copy()

        if df["movie_id"].isna().any():
            raise ValueError(
                f"{nombre} contiene movie_id nulos."
            )

        if df["movie_id"].duplicated().any():
            raise ValueError(
                f"{nombre} contiene movie_id duplicados."
            )

        ids = pd.to_numeric(
            df["movie_id"],
            errors="coerce",
        )

        scores = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

        if (
            ids.isna().any()
            or not np.isfinite(ids.to_numpy(dtype=float)).all()
            or (ids <= 0).any()
            or (ids % 1 != 0).any()
            or scores.isna().any()
            or not np.isfinite(scores.to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                f"{nombre} contiene valores inválidos."
            )

        df["movie_id"] = ids.astype("int64")
        df[columna] = scores.astype("float64")

        # Ordenar independientemente cada modelo.
        df = df.sort_values(
            by=[columna, "movie_id"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

        df["rank"] = np.arange(1, len(df) + 1)

        return df[["movie_id", "rank"]]

    contenido = preparar_ranking(
        content_ranking,
        "content_score",
        "Content-Based",
    ).rename(columns={"rank": "content_rank"})

    colaborativo = preparar_ranking(
        collaborative_ranking,
        "collaborative_score",
        "Collaborative",
    ).rename(columns={"rank": "collaborative_rank"})

    # Conservar candidatos de cualquiera
    # de los dos modelos.
    resultado = contenido.merge(
        colaborativo,
        on="movie_id",
        how="outer",
        validate="one_to_one",
    )

    if seen_movie_ids is not None:
        resultado = resultado.loc[
            ~resultado["movie_id"].isin(
                seen_movie_ids
            )
        ]

    resultado["content_contribution"] = (
        content_weight
        / (rrf_k + resultado["content_rank"])
    ).fillna(0.0)

    resultado["collaborative_contribution"] = (
        collaborative_weight
        / (rrf_k + resultado["collaborative_rank"])
    ).fillna(0.0)

    resultado["hybrid_score"] = (
        resultado["content_contribution"]
        + resultado["collaborative_contribution"]
    )

    resultado = resultado.sort_values(
        by=["hybrid_score", "movie_id"],
        ascending=[False, True],
        kind="stable",
    )

    return resultado.head(n).reset_index(
        drop=True
    )


def recommend_hybrid_for_user(
    user_id: int,
    train_interactions: pd.DataFrame,
    content_model: dict,
    collaborative_model: dict,
    n: int = 10,
    content_weight: float = 0.5,
    collaborative_weight: float = 0.5,
    baseline_ranking: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Genera recomendaciones híbridas para un usuario.

    Utiliza modelos previamente entrenados y combina
    sus rankings mediante Reciprocal Rank Fusion.

    No vuelve a entrenar TF-IDF ni TruncatedSVD.
    No utiliza interacciones de Test.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n debe ser un entero positivo."
        )

    columnas = {"user_id", "movie_id", "rating"}

    if not columnas.issubset(train_interactions.columns):
        raise ValueError(
            "Train no contiene columnas obligatorias."
        )

    campos_content = {
        "content_df",
        "tfidf_matrix",
    }

    if not campos_content.issubset(content_model):
        raise ValueError(
            "El modelo Content-Based está incompleto."
        )

    campos_collaborative = {
        "user_to_idx",
        "movie_ids",
    }

    if not campos_collaborative.issubset(
        collaborative_model
    ):
        raise ValueError(
            "El modelo Collaborative está incompleto."
        )

    content_df = content_model["content_df"]
    tfidf_matrix = content_model["tfidf_matrix"]

    # Solicitar suficientes candidatos antes
    # de aplicar la fusión híbrida.
    total_content = len(content_df)
    total_collaborative = len(
        collaborative_model["movie_ids"]
    )

    content_ranking = pd.DataFrame({
        "movie_id": pd.Series(dtype="int64"),
        "content_score": pd.Series(dtype="float64"),
    })

    collaborative_ranking = pd.DataFrame({
        "movie_id": pd.Series(dtype="int64"),
        "collaborative_score": pd.Series(dtype="float64"),
    })

    # ---------------------------------------
    # 1. COMPROBAR PERFIL CONTENT-BASED
    # ---------------------------------------

    positivas = train_interactions.loc[
        (train_interactions["user_id"] == user_id)
        & (train_interactions["rating"] >= 4.0)
    ]

    ids_content = {
        movie_id: indice
        for indice, movie_id in enumerate(
            content_df["movie_id"]
        )
    }

    tiene_perfil = any(
        movie_id in ids_content
        and tfidf_matrix[
            ids_content[movie_id]
        ].nnz > 0
        for movie_id in positivas["movie_id"]
    )

    if tiene_perfil:

        content_ranking = recommend_content_for_user(
            user_id=user_id,
            interactions=train_interactions,
            content_df=content_df,
            tfidf_matrix=tfidf_matrix,
            n=total_content,
        )

    # ---------------------------------------
    # 2. COMPROBAR MODELO COLABORATIVO
    # ---------------------------------------

    if user_id in collaborative_model["user_to_idx"]:

        collaborative_ranking = (
            recommend_collaborative_top_n(
                fitted_model=collaborative_model,
                user_id=user_id,
                train_interactions=train_interactions,
                n=total_collaborative,
            )
        )

    # ---------------------------------------
    # 3. EXCLUIR PELÍCULAS YA CALIFICADAS
    # ---------------------------------------

    vistas = set(
        train_interactions.loc[
            train_interactions["user_id"] == user_id,
            "movie_id",
        ].dropna().tolist()
    )

    # ---------------------------------------
    # 4. GENERAR RANKING HÍBRIDO
    # ---------------------------------------

    resultado = fuse_hybrid_rankings(
        content_ranking=content_ranking,
        collaborative_ranking=collaborative_ranking,
        n=n,
        content_weight=content_weight,
        collaborative_weight=collaborative_weight,
        seen_movie_ids=vistas,
    )


    # Si alguno de los modelos personalizados
    # tiene candidatos, conservar su ranking RRF.
    if not resultado.empty:
        resultado["baseline_score"] = np.nan
        resultado["recommendation_source"] = "hybrid"
        return resultado

    # No inventar preferencias ni factores.
    # El respaldo Baseline es opcional.
    if baseline_ranking is None:
        return resultado

    requeridas_baseline = {
        "movie_id",
        "weighted_score",
    }

    if not requeridas_baseline.issubset(
        baseline_ranking.columns
    ):
        raise ValueError(
            "Baseline requiere movie_id y weighted_score."
        )

    respaldo = baseline_ranking.copy()

    # Solo consideramos candidatos con puntuación.
    respaldo = respaldo.dropna(
        subset=["movie_id", "weighted_score"]
    )

    if respaldo["movie_id"].duplicated().any():
        raise ValueError(
            "Baseline contiene movie_id duplicados."
        )

    ids = pd.to_numeric(
        respaldo["movie_id"],
        errors="coerce",
    )

    scores = pd.to_numeric(
        respaldo["weighted_score"],
        errors="coerce",
    )

    if (
        not np.isfinite(ids.to_numpy(dtype=float)).all()
        or (ids <= 0).any()
        or (ids % 1 != 0).any()
        or not np.isfinite(
            scores.to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            "Baseline contiene puntuaciones "
            "o identificadores inválidos."
        )

    respaldo["movie_id"] = ids.astype("int64")
    respaldo["weighted_score"] = scores

    # Excluir también películas ya calificadas.
    respaldo = respaldo.loc[
        ~respaldo["movie_id"].isin(vistas)
    ]

    respaldo = respaldo.sort_values(
        by=["weighted_score", "movie_id"],
        ascending=[False, True],
        kind="stable",
    ).head(n).copy()

    # Mantener las columnas comunes del híbrido.
    # No convertir las puntuaciones Baseline en RRF.
    respaldo["content_rank"] = np.nan
    respaldo["collaborative_rank"] = np.nan
    respaldo["content_contribution"] = 0.0
    respaldo["collaborative_contribution"] = 0.0
    respaldo["hybrid_score"] = np.nan

    respaldo["baseline_score"] = (
        respaldo["weighted_score"]
    )

    respaldo["recommendation_source"] = "baseline"

    columnas_salida = [
        "movie_id",
        "content_rank",
        "collaborative_rank",
        "content_contribution",
        "collaborative_contribution",
        "hybrid_score",
        "baseline_score",
        "recommendation_source",
    ]

    return respaldo[
        columnas_salida
    ].reset_index(drop=True)
