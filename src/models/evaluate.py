
"""
Evaluación de los recomendadores CineMatch.

Funciones para separar datos de entrenamiento
y prueba sin utilizar interacciones futuras
del mismo usuario durante el entrenamiento.
"""

import pandas as pd

from math import log2

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

def evaluate_collaborative_hit_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fitted_model: dict,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Evalúa el modelo colaborativo mediante Hit Rate@K.

    fitted_model debe haber sido entrenado
    exclusivamente con train.

    Excluye del ranking las películas vistas en train.
    Reporta por separado usuarios no evaluables.
    """

    from src.models.collaborative import (
        recommend_collaborative_top_n,
    )

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(
            "k debe ser un entero positivo."
        )

    if not 0.5 <= relevance_threshold <= 5.0:
        raise ValueError(
            "relevance_threshold debe estar entre 0.5 y 5.0."
        )

    requeridas = {"user_id", "movie_id", "rating"}

    for nombre, datos in (("Train", train), ("Test", test)):

        faltantes = requeridas - set(datos.columns)

        if faltantes:
            raise ValueError(
                f"{nombre} contiene columnas faltantes: "
                + ", ".join(sorted(faltantes))
            )

        if datos[list(requeridas)].isna().any().any():
            raise ValueError(
                f"{nombre} contiene valores nulos."
            )

        if datos.duplicated(
            subset=["user_id", "movie_id"]
        ).any():
            raise ValueError(
                f"{nombre} contiene interacciones duplicadas."
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

    if not {
        "user_to_idx",
        "movie_to_idx",
        "movie_ids",
    }.issubset(fitted_model):
        raise ValueError(
            "El modelo colaborativo está incompleto."
        )

    usuarios_modelo = set(
        fitted_model["user_to_idx"]
    )

    peliculas_modelo = set(
        fitted_model["movie_to_idx"]
    )

    relevantes = test.loc[
        test["rating"] >= relevance_threshold
    ]

    usuarios_relevantes = int(
        relevantes["user_id"].nunique()
    )

    aciertos = 0
    evaluables = 0
    excluidos = 0

    for user_id, grupo in relevantes.groupby("user_id"):

        objetivos = set(
            grupo["movie_id"].tolist()
        )

        # Para ser evaluable necesitamos un usuario
        # y al menos un objetivo conocidos por el modelo.
        if (
            user_id not in usuarios_modelo
            or not objetivos.intersection(peliculas_modelo)
        ):
            excluidos += 1
            continue

        recomendaciones = recommend_collaborative_top_n(
            fitted_model=fitted_model,
            user_id=user_id,
            train_interactions=train,
            n=k,
        )

        peliculas_recomendadas = set(
            recomendaciones["movie_id"].tolist()
        )

        evaluables += 1

        if objetivos.intersection(peliculas_recomendadas):
            aciertos += 1

    return {
        "metrica": f"Hit Rate@{k}",
        "hit_rate": (
            aciertos / evaluables
            if evaluables > 0
            else None
        ),
        "hit_rate_global": (
            aciertos / usuarios_relevantes
            if usuarios_relevantes > 0
            else None
        ),
        "cobertura_evaluacion": (
            evaluables / usuarios_relevantes
            if usuarios_relevantes > 0
            else None
        ),
        "usuarios_relevantes": usuarios_relevantes,
        "usuarios_evaluables": evaluables,
        "usuarios_excluidos": excluidos,
        "aciertos": aciertos,
    }

def evaluate_hybrid_hit_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    content_model: dict,
    collaborative_model: dict,
    baseline_ranking: pd.DataFrame | None = None,
    k: int = 10,
    relevance_threshold: float = 4.0,
    content_weight: float = 0.5,
    collaborative_weight: float = 0.5,
) -> dict:
    """
    Evalúa el recomendador híbrido con Hit Rate@K.

    - Utiliza exclusivamente Train como historial.
    - Test solo determina los objetivos relevantes.
    - Los usuarios sin recomendaciones cuentan
      en el denominador global.
    - No reentrena ninguno de los modelos.

    Los modelos recibidos deben haber sido
    entrenados sin utilizar interacciones Test.
    """

    from src.models.hybrid import (
        recommend_hybrid_for_user,
    )

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(
            "k debe ser un entero positivo."
        )

    if not 0.5 <= relevance_threshold <= 5.0:
        raise ValueError(
            "relevance_threshold debe estar entre 0.5 y 5.0."
        )

    requeridas = {"user_id", "movie_id", "rating"}

    for nombre, datos in (
        ("Train", train),
        ("Test", test),
    ):
        faltantes = requeridas - set(datos.columns)

        if faltantes:
            raise ValueError(
                f"{nombre} contiene columnas faltantes: "
                + ", ".join(sorted(faltantes))
            )

        if datos[list(requeridas)].isna().any().any():
            raise ValueError(
                f"{nombre} contiene valores nulos."
            )

        if datos.duplicated(
            subset=["user_id", "movie_id"]
        ).any():
            raise ValueError(
                f"{nombre} contiene interacciones duplicadas."
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

    aciertos = 0
    usuarios_con_recomendaciones = 0

    for user_id, grupo in relevantes.groupby("user_id"):

        objetivos = set(
            grupo["movie_id"].tolist()
        )

        recomendaciones = recommend_hybrid_for_user(
            user_id=user_id,
            train_interactions=train,
            content_model=content_model,
            collaborative_model=collaborative_model,
            baseline_ranking=baseline_ranking,
            n=k,
            content_weight=content_weight,
            collaborative_weight=collaborative_weight,
        )

        if recomendaciones.empty:
            continue

        usuarios_con_recomendaciones += 1

        recomendadas = set(
            recomendaciones["movie_id"].tolist()
        )

        if objetivos.intersection(recomendadas):
            aciertos += 1

    sin_recomendaciones = (
        usuarios_relevantes
        - usuarios_con_recomendaciones
    )

    return {
        "metrica": f"Hit Rate@{k}",

        "hit_rate_global": (
            aciertos / usuarios_relevantes
            if usuarios_relevantes > 0
            else None
        ),

        "hit_rate_con_recomendaciones": (
            aciertos / usuarios_con_recomendaciones
            if usuarios_con_recomendaciones > 0
            else None
        ),

        "cobertura_recomendaciones": (
            usuarios_con_recomendaciones / usuarios_relevantes
            if usuarios_relevantes > 0
            else None
        ),

        "usuarios_relevantes": usuarios_relevantes,
        "usuarios_con_recomendaciones": (
            usuarios_con_recomendaciones
        ),
        "usuarios_sin_recomendaciones": (
            sin_recomendaciones
        ),
        "aciertos": aciertos,
    }

def evaluate_ranking_metrics(
    relevant_by_user: dict,
    recommendations_by_user: dict,
    k: int = 10,
) -> dict:
    """
    Evalúa recomendaciones Top-K mediante:

    - Hit Rate@K
    - Recall@K
    - NDCG@K
    - Cobertura de recomendaciones

    relevant_by_user:
        {user_id: {movie_id_relevante, ...}}

    recommendations_by_user:
        {user_id: [movie_id_1, movie_id_2, ...]}

    Calcula promedios macro por usuario.
    Los usuarios sin recomendaciones cuentan
    con métricas de ranking iguales a cero.

    No entrena modelos ni utiliza archivos.
    """

    if (
        isinstance(k, bool)
        or not isinstance(k, int)
        or k <= 0
    ):
        raise ValueError(
            "k debe ser un entero positivo."
        )

    if not isinstance(relevant_by_user, dict):
        raise ValueError(
            "Las películas relevantes deben ser un diccionario."
        )

    if not isinstance(recommendations_by_user, dict):
        raise ValueError(
            "Las recomendaciones deben ser un diccionario."
        )

    resultados = []

    for user_id, relevantes in relevant_by_user.items():

        if (
            not isinstance(relevantes, (set, list, tuple))
            or not relevantes
        ):
            raise ValueError(
                "Cada usuario evaluado requiere "
                "al menos una película relevante."
            )

        relevantes = set(relevantes)

        recomendadas = recommendations_by_user.get(
            user_id, []
        )

        if not isinstance(recomendadas, (list, tuple)):
            raise ValueError(
                "Las recomendaciones deben ser una lista."
            )

        # Una película no puede ocupar dos posiciones
        # en el ranking del mismo usuario.
        if len(recomendadas) != len(set(recomendadas)):
            raise ValueError(
                "El ranking contiene películas duplicadas."
            )

        top_k = recomendadas[:k]

        # Hit Rate: al menos una película relevante.
        aciertos = sum(
            movie_id in relevantes
            for movie_id in top_k
        )

        hit = float(aciertos > 0)

        # Recall: películas relevantes recuperadas
        # dividido entre todas las relevantes.
        recall = aciertos / len(relevantes)

        # DCG: recompensa los aciertos tempranos.
        dcg = sum(
            1.0 / log2(posicion + 2)
            for posicion, movie_id in enumerate(top_k)
            if movie_id in relevantes
        )

        # IDCG: mejor ranking posible para ese
        # usuario, limitado a las primeras K posiciones.
        ideal_hits = min(k, len(relevantes))

        idcg = sum(
            1.0 / log2(posicion + 2)
            for posicion in range(ideal_hits)
        )

        ndcg = dcg / idcg

        resultados.append({
            "user_id": user_id,
            "hit": hit,
            "recall": recall,
            "ndcg": ndcg,
            "tiene_recomendaciones": bool(top_k),
        })

    if not resultados:
        raise ValueError(
            "No existen usuarios relevantes para evaluar."
        )

    total = len(resultados)

    return {
        "k": k,
        "usuarios_relevantes": total,
        "usuarios_con_recomendaciones": sum(
            fila["tiene_recomendaciones"]
            for fila in resultados
        ),
        "hit_rate_at_k": sum(
            fila["hit"] for fila in resultados
        ) / total,
        "recall_at_k": sum(
            fila["recall"] for fila in resultados
        ) / total,
        "ndcg_at_k": sum(
            fila["ndcg"] for fila in resultados
        ) / total,
        "cobertura_recomendaciones": sum(
            fila["tiene_recomendaciones"]
            for fila in resultados
        ) / total,
        "detalle_usuarios": resultados,
    }

def temporal_train_validation_test_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide las interacciones temporalmente por usuario.

    Para usuarios con tres o más interacciones:
    - Train: todas excepto las dos últimas.
    - Validation: penúltima interacción.
    - Test: última interacción.

    Usuarios con una o dos interacciones:
    - Permanecen completamente en Train.

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
            "Existen interacciones con valores nulos."
        )

    if df.duplicated(
        subset=["user_id", "movie_id"]
    ).any():
        raise ValueError(
            "Existen interacciones usuario-película "
            "duplicadas."
        )

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

    df = df.sort_values(
        by=["user_id", "rated_at", "movie_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Número de interacciones de cada usuario.
    cantidades = df.groupby("user_id")[
        "movie_id"
    ].transform("size")

    # Posición cronológica dentro del usuario.
    posicion = df.groupby("user_id").cumcount()

    # Solo usuarios con tres o más interacciones
    # aportan registros a Validation y Test.
    elegibles = cantidades >= 3

    es_validation = (
        elegibles
        & (posicion == cantidades - 2)
    )

    es_test = (
        elegibles
        & (posicion == cantidades - 1)
    )

    train = df.loc[
        ~(es_validation | es_test)
    ].copy().reset_index(drop=True)

    validation = df.loc[
        es_validation
    ].copy().reset_index(drop=True)

    test = df.loc[
        es_test
    ].copy().reset_index(drop=True)

    if validation.empty or test.empty:
        raise ValueError(
            "Se requieren usuarios con al menos "
            "tres interacciones para dividir."
        )

    return train, validation, test


def select_hybrid_weights(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    content_model: dict,
    collaborative_model: dict,
    k: int = 10,
    candidates: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Selecciona pesos RRF utilizando Validation.

    Los modelos deben estar entrenados solo con Train.

    No recibe Test, no reentrena modelos
    ni modifica los datos originales.
    """

    if candidates is None:
        candidates = [
            (0.25, 0.75),
            (0.50, 0.50),
            (0.75, 0.25),
        ]

    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise ValueError(
            "Se requiere al menos una configuración RRF."
        )

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(
            "k debe ser un entero positivo."
        )

    resultados = []

    for pesos in candidates:

        if not isinstance(pesos, (tuple, list)) or len(pesos) != 2:
            raise ValueError(
                "Cada configuración requiere dos pesos."
            )

        wc, wf = pesos

        if (
            any(
                isinstance(p, bool)
                or not isinstance(p, (int, float))
                for p in (wc, wf)
            )
            or not all(
                __import__("math").isfinite(p)
                for p in (wc, wf)
            )
            or wc < 0
            or wf < 0
            or not __import__("math").isclose(
                wc + wf, 1.0, abs_tol=1e-9
            )
        ):
            raise ValueError(
                "Los pesos RRF deben ser válidos y sumar 1."
            )

        evaluacion = evaluate_hybrid_hit_rate(
            train=train,
            test=validation,
            content_model=content_model,
            collaborative_model=collaborative_model,
            k=k,
            content_weight=float(wc),
            collaborative_weight=float(wf),
        )

        puntuacion = evaluacion["hit_rate_global"]

        if puntuacion is None:
            raise ValueError(
                "Validation no contiene usuarios relevantes."
            )

        resultados.append({
            "content_weight": float(wc),
            "collaborative_weight": float(wf),
            "hit_rate_global": float(puntuacion),
            "cobertura": evaluacion[
                "cobertura_recomendaciones"
            ],
            "aciertos": evaluacion["aciertos"],
        })

    # Primero optimizamos Hit Rate global.
    # Ante empates, preferimos mayor cobertura.
    # Si persiste el empate, conservamos el
    # primer candidato, para ser deterministas.
    mejor = max(
        resultados,
        key=lambda r: (
            r["hit_rate_global"],
            r["cobertura"] if r["cobertura"] is not None else -1,
        ),
    )

    return {
        "estado": "selected",
        "metrica_seleccion": f"Hit Rate@{k} global",
        "mejor_configuracion": mejor,
        "resultados_validation": resultados,
    }
