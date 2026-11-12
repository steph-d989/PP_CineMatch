"""
Modelo Baseline de CineMatch.

Construye puntuaciones ponderadas de películas
utilizando métricas de calificaciones.

No entrena modelos complejos, no modifica datos
y no realiza conexiones externas.
"""

import pandas as pd


def build_baseline_scores(
    dim_movie: pd.DataFrame,
    movie_metrics: pd.DataFrame,
    quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Calcula un ranking base mediante puntuación
    ponderada por cantidad de calificaciones.

    Parámetros
    ----------
    dim_movie:
        Catálogo de películas de Gold Analytics.

    movie_metrics:
        Métricas agregadas por película.

    quantile:
        Percentil utilizado para calcular m.

    Devuelve
    --------
    DataFrame con weighted_score.
    """

    if not 0 <= quantile <= 1:
        raise ValueError(
            "quantile debe estar entre 0 y 1."
        )

    requeridas_movie = {"movie_id", "title"}

    requeridas_metrics = {
        "movie_id",
        "rating_count",
        "rating_mean",
    }

    if not requeridas_movie.issubset(
        dim_movie.columns
    ):
        raise ValueError(
            "dim_movie no contiene columnas obligatorias."
        )

    if not requeridas_metrics.issubset(
        movie_metrics.columns
    ):
        raise ValueError(
            "movie_metrics no contiene columnas obligatorias."
        )

    # Cada película debe aparecer una sola vez.
    if (
        dim_movie["movie_id"].isna().any()
        or dim_movie["movie_id"].duplicated().any()
        or movie_metrics["movie_id"].isna().any()
        or movie_metrics["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "Baseline contiene movie_id nulos o duplicados."
        )

    df = dim_movie.merge(
        movie_metrics,
        on="movie_id",
        how="left",
        validate="one_to_one",
    )

    # Normalizar métricas numéricas.
    df["rating_count"] = pd.to_numeric(
        df["rating_count"],
        errors="coerce",
    ).fillna(0)

    df["rating_mean"] = pd.to_numeric(
        df["rating_mean"],
        errors="coerce",
    )

    if (
        (df["rating_count"] < 0).any()
        or (df["rating_count"] % 1 != 0).any()
    ):
        raise ValueError(
            "Baseline contiene cantidades de ratings inválidas."
        )

    if (
        (df["rating_count"] > 0)
        & df["rating_mean"].isna()
    ).any():
        raise ValueError(
            "Baseline contiene películas calificadas "
            "sin promedio válido."
        )

    # Seleccionar películas con ratings reales.
    calificadas = df.loc[
        df["rating_count"] > 0
    ].copy()

    if calificadas.empty:
        raise ValueError(
            "Baseline requiere al menos una película "
            "con calificaciones."
        )

    # Promedio global ponderado por número de ratings.
    C = (
        (
            calificadas["rating_count"]
            * calificadas["rating_mean"]
        ).sum()
        / calificadas["rating_count"].sum()
    )

    # Parámetro de regularización.
    m = float(
        calificadas["rating_count"].quantile(
            quantile
        )
    )

    # Calcular las puntuaciones solo donde
    # tenemos calificaciones observadas.
    v = calificadas["rating_count"]
    R = calificadas["rating_mean"]

    calificadas["weighted_score"] = (
        (v / (v + m)) * R
        + (m / (v + m)) * C
    )

    # Conservar las películas sin ratings en el
    # catálogo, pero sin asignarles un score.
    df["weighted_score"] = float("nan")

    df.loc[
        calificadas.index,
        "weighted_score",
    ] = calificadas["weighted_score"]

    # Registrar parámetros para trazabilidad.
    df.attrs["global_mean"] = float(C)
    df.attrs["min_votes_parameter"] = m
    df.attrs["quantile"] = quantile

    return df

def recommend_baseline_top_n(
    baseline_scores: pd.DataFrame,
    n: int = 10,
    min_ratings: int = 1,
) -> pd.DataFrame:
    """
    Genera las N películas mejor posicionadas
    según su puntuación ponderada.

    No entrena modelos ni consulta servicios.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n debe ser un entero positivo."
        )

    if (
        isinstance(min_ratings, bool)
        or not isinstance(min_ratings, int)
        or min_ratings < 1
    ):
        raise ValueError(
            "min_ratings debe ser un entero positivo."
        )

    requeridas = {
        "movie_id",
        "title",
        "rating_count",
        "weighted_score",
    }

    if not requeridas.issubset(baseline_scores.columns):
        raise ValueError(
            "Baseline no contiene columnas obligatorias "
            "para generar recomendaciones."
        )

    ranking = baseline_scores.dropna(
        subset=[
            "movie_id",
            "title",
            "weighted_score",
        ]
    ).copy()

    ranking = ranking.loc[
        ranking["rating_count"] >= min_ratings
    ]

    ranking = ranking.sort_values(
        by=[
            "weighted_score",
            "rating_count",
            "movie_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    )

    return ranking.head(n).reset_index(drop=True)

def build_training_movie_metrics(
    train_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula métricas de películas utilizando
    exclusivamente las calificaciones recibidas
    como conjunto de entrenamiento.

    No realiza la división train/test.
    No consulta Gold Analytics ni modifica archivos.
    """

    requeridas = {
        "movie_id",
        "rating",
    }

    if not requeridas.issubset(train_ratings.columns):
        raise ValueError(
            "El entrenamiento requiere movie_id y rating."
        )

    df = train_ratings[
        ["movie_id", "rating"]
    ].copy()

    if df.empty:
        raise ValueError(
            "El entrenamiento no contiene calificaciones."
        )

    if df.isna().any().any():
        raise ValueError(
            "El entrenamiento contiene valores nulos."
        )

    # Validar los identificadores de películas.
    ids = pd.to_numeric(
        df["movie_id"],
        errors="coerce",
    )

    if (
        ids.isna().any()
        or (ids <= 0).any()
        or (ids % 1 != 0).any()
    ):
        raise ValueError(
            "El entrenamiento contiene movie_id inválidos."
        )

    df["movie_id"] = ids.astype("int64")

    # Normalizar las calificaciones.
    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    # MovieLens utiliza calificaciones entre
    # 0.5 y 5.0.
    if (
        df["rating"].isna().any()
        or (~df["rating"].between(0.5, 5.0)).any()
    ):
        raise ValueError(
            "El entrenamiento contiene ratings inválidos."
        )

    metricas = (
        df.groupby("movie_id", as_index=False)
        .agg(
            rating_count=("rating", "size"),
            rating_mean=("rating", "mean"),
            rating_std=("rating", "std"),
        )
    )

    # Una sola valoración no permite calcular
    # desviación estándar muestral.
    metricas["rating_std"] = (
        metricas["rating_std"].fillna(0.0)
    )

    return metricas
