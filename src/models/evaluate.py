
"""
Evaluación de los recomendadores CineMatch.

Funciones para separar datos de entrenamiento
y prueba sin utilizar interacciones futuras
del mismo usuario durante el entrenamiento.
"""

import pandas as pd


def temporal_train_test_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reserva la última interacción de cada usuario
    que tenga al menos dos valoraciones.

    Los usuarios con una sola valoración
    permanecen en el conjunto de entrenamiento.

    No modifica el DataFrame original.
    """

    requeridas = {
        "user_id",
        "movie_id",
        "rating",
        "rated_at",
    }

    faltantes = requeridas - set(interactions.columns)

    if faltantes:
        raise ValueError(
            "Faltan columnas para la división temporal: "
            + ", ".join(sorted(faltantes))
        )

    if interactions.empty:
        raise ValueError(
            "No existen interacciones para dividir."
        )

    df = interactions.copy()

    if df[list(requeridas)].isna().any().any():
        raise ValueError(
            "Las interacciones contienen valores nulos."
        )

    # Evitar que una misma combinación usuario-película
    # pueda aparecer en entrenamiento y prueba.
    if df.duplicated(
        subset=["user_id", "movie_id"]
    ).any():
        raise ValueError(
            "Existen interacciones usuario-película "
            "duplicadas."
        )

    # Normalizar las fechas.
    try:
        df["rated_at"] = pd.to_datetime(
            df["rated_at"],
            utc=True,
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Existen fechas de rating inválidas."
        ) from exc

    if df["rated_at"].isna().any():
        raise ValueError(
            "Existen fechas de rating inválidas."
        )

    # Orden determinista, incluso si dos ratings
    # tienen la misma fecha.
    df = df.sort_values(
        by=["user_id", "rated_at", "movie_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Reservar la última interacción solamente
    # para usuarios que tengan historial suficiente.
    cantidades = df.groupby("user_id")[
        "movie_id"
    ].transform("size")

    ultima = ~df.duplicated(
        subset=["user_id"],
        keep="last",
    )

    es_test = (cantidades >= 2) & ultima

    train = df.loc[
        ~es_test
    ].copy().reset_index(drop=True)

    test = df.loc[
        es_test
    ].copy().reset_index(drop=True)

    if test.empty:
        raise ValueError(
            "No existen usuarios con al menos "
            "dos interacciones para evaluar."
        )

    return train, test

def evaluate_baseline_hit_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dim_movie: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Evalúa Hit Rate@K para un baseline global.

    Calcula la popularidad exclusivamente desde train.
    Excluye películas vistas durante entrenamiento.
    Evalúa usuarios con una interacción relevante
    en test y cuya película pueda recomendarse.

    Los casos no evaluables se reportan por separado.
    """

    from src.models.baseline import (
        build_training_movie_metrics,
        build_baseline_scores,
        recommend_baseline_top_n,
    )

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo.")

    requeridas = {"user_id", "movie_id", "rating"}

    for nombre, datos in (("train", train), ("test", test)):
        if not requeridas.issubset(datos.columns):
            raise ValueError(
                f"{nombre} no contiene columnas obligatorias."
            )

    if not {"movie_id", "title"}.issubset(dim_movie.columns):
        raise ValueError(
            "dim_movie requiere movie_id y title."
        )

    if train.empty or test.empty:
        raise ValueError(
            "Entrenamiento y prueba deben contener registros."
        )

    if train[list(requeridas)].isna().any().any():
        raise ValueError("Train contiene valores nulos.")

    if test[list(requeridas)].isna().any().any():
        raise ValueError("Test contiene valores nulos.")

    
    # Comprobar que las combinaciones usuario-película
    # no se repitan dentro de cada conjunto ni
    # entre entrenamiento y prueba.

    columnas_interaccion = [
        "user_id",
        "movie_id",
    ]

    if train.duplicated(
        subset=columnas_interaccion
    ).any():
        raise ValueError(
            "Train contiene interacciones "
            "usuario-película duplicadas."
        )

    if test.duplicated(
        subset=columnas_interaccion
    ).any():
        raise ValueError(
            "Test contiene interacciones "
            "usuario-película duplicadas."
        )

    pares_train = set(
        zip(
            train["user_id"],
            train["movie_id"],
        )
    )

    pares_test = set(
        zip(
            test["user_id"],
            test["movie_id"],
        )
    )

    if not pares_train.isdisjoint(pares_test):
        raise ValueError(
            "Existe solapamiento usuario-película "
            "entre Train y Test."
        )

    
    # El baseline nunca calcula métricas usando test.
    metricas = build_training_movie_metrics(train)

    scores = build_baseline_scores(
        dim_movie=dim_movie,
        movie_metrics=metricas,
    )

    # Ranking global completo. Después filtraremos
    # las películas vistas por cada usuario.
    ranking = recommend_baseline_top_n(
        baseline_scores=scores,
        n=len(scores),
        min_ratings=1,
    )

    candidatos = set(ranking["movie_id"].tolist())

    vistos = (
        train.groupby("user_id")["movie_id"]
        .agg(set)
        .to_dict()
    )

    relevantes = test.loc[
        test["rating"] >= relevance_threshold
    ]

    aciertos = 0
    evaluables = 0
    excluidos = 0
    
    
    # Cantidad de usuarios con al menos una
    # interacción relevante en el conjunto test.
    usuarios_relevantes = (
        relevantes["user_id"].nunique()
    )

    
    for usuario, grupo in relevantes.groupby("user_id"):

        peliculas_objetivo = set(
            grupo["movie_id"].tolist()
        )

        # Las películas relevantes deben tener
        # puntuación disponible en el baseline.
        if not peliculas_objetivo.intersection(candidatos):
            excluidos += 1
            continue

        peliculas_vistas = vistos.get(usuario, set())

        recomendaciones = (
            ranking.loc[
                ~ranking["movie_id"].isin(peliculas_vistas),
                "movie_id",
            ]
            .head(k)
            .tolist()
        )

        evaluables += 1

        if peliculas_objetivo.intersection(recomendaciones):
            aciertos += 1

    
    return {
        "metrica": f"Hit Rate@{k}",

        # Aciertos entre usuarios evaluables.
        "hit_rate": (
            aciertos / evaluables
            if evaluables > 0
            else None
        ),

        # Aciertos entre todos los usuarios
        # con interacciones relevantes en test.
        # Los excluidos cuentan como no acertados.
        "hit_rate_global": (
            aciertos / usuarios_relevantes
            if usuarios_relevantes > 0
            else None
        ),

        # Proporción de usuarios relevantes
        # con objetivos que el baseline puede
        # puntuar utilizando train.
        "cobertura_evaluacion": (
            evaluables / usuarios_relevantes
            if usuarios_relevantes > 0
            else None
        ),

        "usuarios_relevantes": int(
            usuarios_relevantes
        ),
        "usuarios_evaluables": evaluables,
        "usuarios_excluidos": excluidos,
        "aciertos": aciertos,
    }

def evaluate_content_hit_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    content_df: pd.DataFrame,
    tfidf_matrix,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Evalúa recomendaciones Content-Based mediante
    Hit Rate@K con interacciones separadas.

    Los perfiles utilizan exclusivamente train.

    Usuarios sin perfil positivo o sin un objetivo
    representado en el catálogo se reportan como
    no evaluables.
    """

    from src.models.content_based import (
        recommend_content_for_user,
    )

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k debe ser un entero positivo.")

    columnas = {"user_id", "movie_id", "rating"}

    for nombre, datos in (("Train", train), ("Test", test)):

        if not columnas.issubset(datos.columns):
            raise ValueError(
                f"{nombre} no contiene columnas obligatorias."
            )

        if datos[list(columnas)].isna().any().any():
            raise ValueError(
                f"{nombre} contiene valores nulos."
            )

        if datos.duplicated(
            subset=["user_id", "movie_id"]
        ).any():
            raise ValueError(
                f"{nombre} contiene interacciones duplicadas."
            )

    if not {"movie_id", "title"}.issubset(content_df.columns):
        raise ValueError(
            "Content-Based requiere movie_id y title."
        )

    if (
        content_df["movie_id"].isna().any()
        or content_df["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "El catálogo contiene movie_id nulos o duplicados."
        )

    if tfidf_matrix.shape[0] != len(content_df):
        raise ValueError(
            "La matriz TF-IDF no coincide con el catálogo."
        )

    pares_train = set(
        zip(train["user_id"], train["movie_id"])
    )

    pares_test = set(
        zip(test["user_id"], test["movie_id"])
    )

    if not pares_train.isdisjoint(pares_test):
        raise ValueError(
            "Existe solapamiento usuario-película "
            "entre Train y Test."
        )

    relevantes = test.loc[
        test["rating"] >= relevance_threshold
    ]

    usuarios_relevantes = int(
        relevantes["user_id"].nunique()
    )

    catalogo = set(content_df["movie_id"].tolist())

    aciertos = 0
    evaluables = 0
    excluidos = 0

    for user_id, grupo in relevantes.groupby("user_id"):

        objetivos = set(grupo["movie_id"].tolist())

        historial = train.loc[
            train["user_id"] == user_id
        ]

        positivas = historial.loc[
            historial["rating"] >= relevance_threshold
        ]

        # No se puede construir un perfil sin
        # películas favoritas con contenido.
        positivas_validas = (
            set(positivas["movie_id"].tolist())
            & catalogo
        )

        if not positivas_validas:
            excluidos += 1
            continue

        # El objetivo debe pertenecer al catálogo.
        if not (objetivos & catalogo):
            excluidos += 1
            continue

        try:
            recomendaciones = recommend_content_for_user(
                user_id=user_id,
                interactions=train,
                content_df=content_df,
                tfidf_matrix=tfidf_matrix,
                n=k,
                min_rating=relevance_threshold,
            )

        except ValueError as exc:
            # Solo se considera no evaluable la
            # ausencia de características semánticas
            # en las películas favoritas.
            if "características semánticas" not in str(exc):
                raise

            excluidos += 1
            continue

        evaluables += 1

        recomendadas = set(
            recomendaciones["movie_id"].tolist()
        )

        if objetivos & recomendadas:
            aciertos += 1

    return {
        "metrica": f"Hit Rate@{k}",
        "hit_rate": (
            aciertos / evaluables
            if evaluables
            else None
        ),
        "hit_rate_global": (
            aciertos / usuarios_relevantes
            if usuarios_relevantes
            else None
        ),
        "cobertura_evaluacion": (
            evaluables / usuarios_relevantes
            if usuarios_relevantes
            else None
        ),
        "usuarios_relevantes": usuarios_relevantes,
        "usuarios_evaluables": evaluables,
        "usuarios_excluidos": excluidos,
        "aciertos": aciertos,
    }
