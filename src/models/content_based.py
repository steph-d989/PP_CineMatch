
"""
Recomendador Content-Based de CineMatch.

Preparación de atributos semánticos de películas
para su posterior vectorización mediante TF-IDF.

No entrena modelos ni realiza conexiones externas.
"""

import numpy as np
import pandas as pd
import json
import joblib
import hashlib

from scipy.sparse import save_npz, load_npz
from uuid import UUID
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.gold_runs import (
    validate_gold_run_directories,
)

def normalize_list_field(value) -> str:
    """
    Convierte una colección de características
    en una cadena de texto normalizada.

    Ejemplo:
    ["Science Fiction", "Drama"]
    -> "science_fiction drama"
    """

    if value is None:
        return ""

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple, set)):
        return ""

    valores = []

    for item in value:

        if not isinstance(item, str):
            continue

        texto = item.strip().lower()

        if texto:
            valores.append(
                "_".join(texto.split())
            )

    return " ".join(valores)

def normalize_text(value) -> str:
    """
    Normaliza campos textuales simples.

    Los valores nulos se convierten
    en una cadena vacía.
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        return ""

    return " ".join(
        value.strip().lower().split()
    )

def build_content_soup(
    content_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye una representación textual de
    las características de cada película.

    No entrena TF-IDF ni modifica los datos
    originales.
    """

    if "movie_id" not in content_features.columns:
        raise ValueError(
            "Content-Based requiere movie_id."
        )

    if (
        content_features["movie_id"].isna().any()
        or content_features["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "Content-Based contiene movie_id "
            "nulos o duplicados."
        )

    df = content_features.copy()

    campos_lista = [
        "genres",
        "tmdb_genres",
        "keywords",
        "actors",
    ]

    campos_texto = [
        "director",
        "overview",
        "original_language",
    ]

    partes = []

    for columna in campos_lista:

        if columna in df.columns:
            normalizada = df[columna].apply(
                normalize_list_field
            )

            partes.append(normalizada)

    for columna in campos_texto:

        if columna in df.columns:
            normalizada = df[columna].apply(
                normalize_text
            )

            if columna == "director":
                normalizada = normalizada.str.replace(
                    " ", "_", regex=False
                )

            partes.append(normalizada)

    if not partes:
        raise ValueError(
            "Content-Based no contiene "
            "características semánticas."
        )

    df["content_soup"] = ""

    for parte in partes:
        df["content_soup"] = (
            df["content_soup"] + " " + parte
        )

    df["content_soup"] = (
        df["content_soup"]
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
    )

    if df["content_soup"].eq("").all():
        raise ValueError(
            "Content-Based no contiene "
            "información semántica aprovechable."
        )

    return df

def build_tfidf_features(
    content_df: pd.DataFrame,
    max_features: int = 50_000,
    min_df: int = 1,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Ajusta TF-IDF utilizando content_soup.

    Devuelve:
    - Vectorizador entrenado.
    - Matriz dispersa película-características.

    No modifica los datos originales.
    """

    if "content_soup" not in content_df.columns:
        raise ValueError(
            "Content-Based requiere content_soup."
        )

    if content_df.empty:
        raise ValueError(
            "Content-Based no contiene películas."
        )

    if (
        isinstance(max_features, bool)
        or not isinstance(max_features, int)
        or max_features <= 0
    ):
        raise ValueError(
            "max_features debe ser un entero positivo."
        )

    if (
        isinstance(min_df, bool)
        or not isinstance(min_df, int)
        or min_df < 1
    ):
        raise ValueError(
            "min_df debe ser un entero positivo."
        )

    if (
        not isinstance(ngram_range, tuple)
        or len(ngram_range) != 2
        or not all(
            isinstance(valor, int)
            and not isinstance(valor, bool)
            for valor in ngram_range
        )
        or ngram_range[0] < 1
        or ngram_range[1] < ngram_range[0]
    ):
        raise ValueError(
            "ngram_range debe contener límites válidos."
        )

    documentos = (
        content_df["content_soup"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if documentos.eq("").all():
        raise ValueError(
            "No existen textos aprovechables para TF-IDF."
        )

    vectorizador = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )

    try:
        matriz = vectorizador.fit_transform(
            documentos
        )
    except ValueError as exc:
        raise ValueError(
            "TF-IDF no pudo construir un vocabulario "
            "con los textos y parámetros indicados."
        ) from exc

    return vectorizador, matriz.tocsr()

def recommend_similar_movies(
    movie_id: int,
    content_df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    n: int = 10,
) -> pd.DataFrame:
    """
    Recomienda las N películas más similares
    a una película utilizando similitud coseno.

    Excluye la película consultada y los
    documentos sin características TF-IDF.

    No entrena el vectorizador ni modifica datos.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n debe ser un entero positivo."
        )

    if not {"movie_id", "title"}.issubset(
        content_df.columns
    ):
        raise ValueError(
            "Content-Based requiere movie_id y title."
        )

    if (
        content_df["movie_id"].isna().any()
        or content_df["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "Content-Based contiene movie_id "
            "nulos o duplicados."
        )

    if tfidf_matrix.shape[0] != len(content_df):
        raise ValueError(
            "La matriz TF-IDF no coincide "
            "con el catálogo de películas."
        )

    if movie_id not in set(content_df["movie_id"]):
        raise ValueError(
            "La película consultada no existe."
        )

    posicion = content_df.index[
        content_df["movie_id"] == movie_id
    ][0]

    # La posición de las filas de la matriz
    # corresponde al orden, no al índice pandas.
    indice = content_df.index.get_loc(posicion)

    vector_consulta = tfidf_matrix[indice]

    if vector_consulta.nnz == 0:
        raise ValueError(
            "La película no tiene características "
            "semánticas para calcular similitud."
        )

    similitudes = cosine_similarity(
        vector_consulta,
        tfidf_matrix,
    ).ravel()

    resultado = content_df[
        ["movie_id", "title"]
    ].copy().reset_index(drop=True)

    resultado["content_score"] = similitudes

    # Excluir la propia película.
    resultado = resultado.loc[
        resultado["movie_id"] != movie_id
    ]

    # No recomendar por similitud nula.
    resultado = resultado.loc[
        resultado["content_score"] > 0
    ]

    resultado = resultado.sort_values(
        by=["content_score", "movie_id"],
        ascending=[False, True],
        kind="stable",
    )

    return resultado.head(n).reset_index(drop=True)

def build_user_content_profile(
    user_id: int,
    interactions: pd.DataFrame,
    content_df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    min_rating: float = 4.0,
) -> np.ndarray:
    """
    Construye un perfil de usuario mediante
    el promedio de los vectores TF-IDF de las
    películas valoradas positivamente.

    interactions debe contener solamente los
    datos permitidos para construir el perfil,
    normalmente el conjunto de entrenamiento.

    No utiliza datos de prueba ni modifica
    los DataFrames originales.
    """

    requeridas = {
        "user_id",
        "movie_id",
        "rating",
    }

    if not requeridas.issubset(interactions.columns):
        raise ValueError(
            "Faltan columnas de interacciones."
        )

    if "movie_id" not in content_df.columns:
        raise ValueError(
            "Content-Based requiere movie_id."
        )

    if (
        content_df["movie_id"].isna().any()
        or content_df["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "El catálogo contiene movie_id "
            "nulos o duplicados."
        )

    if tfidf_matrix.shape[0] != len(content_df):
        raise ValueError(
            "La matriz TF-IDF no coincide "
            "con el catálogo."
        )

    if not 0.5 <= min_rating <= 5.0:
        raise ValueError(
            "min_rating debe estar entre 0.5 y 5.0."
        )

    # Identificar calificaciones positivas.
    positivas = interactions.loc[
        (interactions["user_id"] == user_id)
        & (interactions["rating"] >= min_rating)
    ]

    if positivas.empty:
        raise ValueError(
            "El usuario no tiene calificaciones "
            "positivas suficientes."
        )

    # Relacionar movie_id con la posición
    # correspondiente en la matriz TF-IDF.
    posiciones = {
        movie_id: indice
        for indice, movie_id in enumerate(
            content_df["movie_id"]
        )
    }

    indices = [
        posiciones[movie_id]
        for movie_id in positivas["movie_id"]
        if movie_id in posiciones
    ]

    # Evitar películas sin información semántica.
    indices = [
        indice
        for indice in indices
        if tfidf_matrix[indice].nnz > 0
    ]

    if not indices:
        raise ValueError(
            "Las películas favoritas no tienen "
            "características semánticas disponibles."
        )

    perfil = tfidf_matrix[
        indices
    ].mean(axis=0)

    return np.asarray(perfil).ravel()

def recommend_content_for_user(
    user_id: int,
    interactions: pd.DataFrame,
    content_df: pd.DataFrame,
    tfidf_matrix: csr_matrix,
    n: int = 10,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """
    Genera recomendaciones personalizadas mediante
    similitud coseno entre el perfil del usuario
    y las películas del catálogo.

    Excluye todas las películas que el usuario
    ya calificó, independientemente del rating.

    Para evaluación, interactions debe contener
    exclusivamente los datos de entrenamiento.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n debe ser un entero positivo."
        )

    if not {"movie_id", "title"}.issubset(
        content_df.columns
    ):
        raise ValueError(
            "Content-Based requiere movie_id y title."
        )

    if not {"user_id", "movie_id", "rating"}.issubset(
        interactions.columns
    ):
        raise ValueError(
            "Faltan columnas de interacciones."
        )

    # Reutilizar la función ya implementada.
    perfil = build_user_content_profile(
        user_id=user_id,
        interactions=interactions,
        content_df=content_df,
        tfidf_matrix=tfidf_matrix,
        min_rating=min_rating,
    )

    if not np.isfinite(perfil).all():
        raise ValueError(
            "El perfil contiene valores no finitos."
        )

    if not np.any(perfil):
        raise ValueError(
            "El perfil no contiene información útil."
        )

    # Una puntuación por película.
    similitudes = cosine_similarity(
        perfil.reshape(1, -1),
        tfidf_matrix,
    ).ravel()

    resultado = content_df[
        ["movie_id", "title"]
    ].copy().reset_index(drop=True)

    resultado["content_score"] = similitudes

    # Excluir absolutamente todas las películas
    # calificadas por el usuario.
    vistas = set(
        interactions.loc[
            interactions["user_id"] == user_id,
            "movie_id",
        ].dropna().tolist()
    )

    resultado = resultado.loc[
        ~resultado["movie_id"].isin(vistas)
    ]

    # Eliminar puntuaciones nulas, no finitas
    # o sin similitud semántica positiva.
    resultado = resultado.loc[
        np.isfinite(resultado["content_score"])
        & (resultado["content_score"] > 0)
    ]

    # Ranking reproducible.
    resultado = resultado.sort_values(
        by=["content_score", "movie_id"],
        ascending=[False, True],
        kind="stable",
    )

    return resultado.head(n).reset_index(drop=True)

def fit_content_based_model(
    content_features: pd.DataFrame,
    max_features: int = 50_000,
    min_df: int = 1,
    ngram_range: tuple[int, int] = (1, 2),
) -> dict:
    """
    Prepara y entrena el modelo Content-Based.

    Ejecuta:
    1. Construcción de content_soup.
    2. Vectorización TF-IDF.

    Devuelve el catálogo alineado, el
    vectorizador y la matriz de características.

    No guarda archivos ni consulta servicios.
    """

    # Construir la representación semántica.
    content_df = build_content_soup(
        content_features
    )

    # Entrenar TF-IDF sobre el catálogo.
    vectorizador, matriz = build_tfidf_features(
        content_df=content_df,
        max_features=max_features,
        min_df=min_df,
        ngram_range=ngram_range,
    )

    # La matriz debe conservar exactamente
    # el orden de las películas.
    if matriz.shape[0] != len(content_df):
        raise ValueError(
            "La matriz TF-IDF no coincide "
            "con el catálogo Content-Based."
        )

    return {
        "content_df": content_df,
        "vectorizer": vectorizador,
        "tfidf_matrix": matriz,
        "total_movies": len(content_df),
        "total_features": matriz.shape[1],
    }

def load_content_features_from_gold_ml(
    gold_dirs: dict[str, Path],
) -> pd.DataFrame:
    """
    Carga las características Content-Based desde
    una ejecución Gold ML versionada.

    Comprueba:
    - Estructura de los directorios.
    - Existencia del archivo Parquet.
    - Columnas obligatorias.
    - Identificadores válidos y únicos.

    No verifica todavía un manifiesto Gold ML.
    No entrena modelos ni modifica archivos.
    """

    validate_gold_run_directories(
        gold_dirs=gold_dirs,
    )

    ruta = (
        Path(gold_dirs["ml_content"])
        / "content_based_features.parquet"
    )

    if not ruta.is_file():
        raise FileNotFoundError(
            "No se encontró content_based_features.parquet "
            "en el run_id seleccionado."
        )

    try:
        df = pd.read_parquet(ruta)

    except (
        OSError,
        ValueError,
        ImportError,
    ) as exc:
        raise ValueError(
            "Gold ML contiene un archivo Content-Based "
            "ilegible o inválido."
        ) from exc

    requeridas = {
        "movie_id",
        "title",
    }

    faltantes = requeridas - set(df.columns)

    if faltantes:
        raise ValueError(
            "Gold ML no contiene columnas obligatorias: "
            + ", ".join(sorted(faltantes))
        )

    if df.empty:
        raise ValueError(
            "Gold ML no contiene películas."
        )

    if (
        df["movie_id"].isna().any()
        or df["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "Gold ML contiene movie_id nulos "
            "o duplicados."
        )

    return df

def save_content_based_model(
    model: dict,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    """
    Guarda los componentes de un modelo
    Content-Based previamente entrenado.

    No sobrescribe archivos existentes.
    No publica resultados en servicios externos.
    """

    requeridos = {
        "content_df",
        "vectorizer",
        "tfidf_matrix",
    }

    if not requeridos.issubset(model):
        raise ValueError(
            "El modelo Content-Based está incompleto."
        )

    df = model["content_df"]
    vectorizer = model["vectorizer"]
    matrix = model["tfidf_matrix"]

    if (
        "movie_id" not in df.columns
        or df["movie_id"].isna().any()
        or df["movie_id"].duplicated().any()
        or matrix.shape[0] != len(df)
        or matrix.shape[1] != len(vectorizer.vocabulary_)
    ):
        raise ValueError(
            "Los componentes Content-Based "
            "no son consistentes."
        )

    output_dir = Path(output_dir)

    # No reutilizar una carpeta anterior.
    if output_dir.exists():
        raise FileExistsError(
            "El directorio del modelo ya existe."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    rutas = {
        "catalog": output_dir / "catalog.parquet",
        "vectorizer": output_dir / "vectorizer.joblib",
        "matrix": output_dir / "tfidf_matrix.npz",
        "metadata": output_dir / "metadata.json",
    }

    df.to_parquet(
        rutas["catalog"],
        index=False,
    )

    joblib.dump(
        vectorizer,
        rutas["vectorizer"],
    )

    save_npz(
        rutas["matrix"],
        matrix,
    )

    metadata = {
        "model_type": "content_based_tfidf",
        "total_movies": len(df),
        "total_features": matrix.shape[1],
        "movie_ids": [
            int(movie_id)
            for movie_id in df["movie_id"]
        ],
    }

    
    # Registrar la ejecución de origen cuando
    # el modelo se haya entrenado con Gold ML
    # versionado.
    if run_id is not None:

        try:
            metadata["run_id"] = str(UUID(run_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "El modelo Content-Based requiere "
                "un run_id UUID válido."
            ) from exc


    rutas["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Registrar las huellas de los cuatro
    # componentes físicos recién guardados.
    manifest = build_content_model_manifest(
        model_dir=output_dir,
    )

    manifest_path = output_dir / "_manifest.json"

    # No sobrescribir un manifiesto anterior.
    with manifest_path.open(
        "x",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            manifest,
            archivo,
            indent=2,
            ensure_ascii=False,
        )
        archivo.write("\n")

    rutas["manifest"] = manifest_path

    return rutas

def build_content_model_manifest(
    model_dir: Path,
) -> dict:
    """
    Construye un manifiesto SHA-256 para los cuatro
    archivos de un modelo Content-Based guardado.

    No modifica ni carga el vectorizador joblib.
    No escribe archivos.
    """

    model_dir = Path(model_dir).resolve()

    archivos = {
        "catalog": "catalog.parquet",
        "vectorizer": "vectorizer.joblib",
        "matrix": "tfidf_matrix.npz",
        "metadata": "metadata.json",
    }

    entradas = []

    for componente, nombre in archivos.items():

        ruta = model_dir / nombre

        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta un componente del modelo: {nombre}"
            )

        tamanio = ruta.stat().st_size

        if tamanio <= 0:
            raise ValueError(
                f"El componente del modelo está vacío: {nombre}"
            )

        sha256 = hashlib.sha256()

        with ruta.open("rb") as archivo:
            for bloque in iter(
                lambda: archivo.read(1024 * 1024),
                b"",
            ):
                sha256.update(bloque)

        entradas.append({
            "component": componente,
            "path": nombre,
            "bytes": tamanio,
            "sha256": sha256.hexdigest(),
        })

    entradas.sort(
        key=lambda entrada: entrada["path"]
    )

    return {
        "model_type": "content_based_tfidf",
        "algorithm": "sha256",
        "total_files": len(entradas),
        "files": entradas,
    }

def verify_content_model_manifest(
    model_dir: Path,
) -> dict:
    """
    Verifica la integridad SHA-256 de un modelo
    Content-Based previamente guardado.

    No deserializa joblib ni modifica archivos.
    """

    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / "_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Content-Based requiere _manifest.json."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "El manifiesto Content-Based es inválido."
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError(
            "El manifiesto debe ser un objeto JSON."
        )

    if (
        manifest.get("model_type")
        != "content_based_tfidf"
        or manifest.get("algorithm") != "sha256"
    ):
        raise ValueError(
            "El manifiesto tiene un tipo o algoritmo inválido."
        )

    esperados = {
        "catalog": "catalog.parquet",
        "vectorizer": "vectorizer.joblib",
        "matrix": "tfidf_matrix.npz",
        "metadata": "metadata.json",
    }

    entradas = manifest.get("files")

    if (
        not isinstance(entradas, list)
        or len(entradas) != len(esperados)
        or manifest.get("total_files") != len(esperados)
    ):
        raise ValueError(
            "El manifiesto contiene componentes incompletos."
        )

    registrados = set()

    for entrada in entradas:

        if not isinstance(entrada, dict):
            raise ValueError(
                "El manifiesto contiene una entrada inválida."
            )

        componente = entrada.get("component")
        nombre = entrada.get("path")
        hash_esperado = entrada.get("sha256")
        bytes_esperados = entrada.get("bytes")

        if (
            componente not in esperados
            or nombre != esperados[componente]
            or componente in registrados
        ):
            raise ValueError(
                "El manifiesto contiene componentes inesperados "
                "o duplicados."
            )

        registrados.add(componente)

        if (
            type(bytes_esperados) is not int
            or bytes_esperados <= 0
            or not isinstance(hash_esperado, str)
            or len(hash_esperado) != 64
            or any(
                caracter not in "0123456789abcdef"
                for caracter in hash_esperado
            )
        ):
            raise ValueError(
                "El manifiesto contiene datos inválidos."
            )

        ruta = (model_dir / nombre).resolve()

        if ruta.parent != model_dir:
            raise ValueError(
                "El manifiesto contiene una ruta externa."
            )

        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta un componente Content-Based: {nombre}"
            )

        if ruta.stat().st_size != bytes_esperados:
            raise ValueError(
                f"El componente tiene un tamaño diferente: {nombre}"
            )

        sha256 = hashlib.sha256()

        with ruta.open("rb") as archivo:
            for bloque in iter(
                lambda: archivo.read(1024 * 1024),
                b"",
            ):
                sha256.update(bloque)

        if sha256.hexdigest() != hash_esperado:
            raise ValueError(
                f"El componente no coincide con SHA-256: {nombre}"
            )

    if registrados != set(esperados):
        raise ValueError(
            "El manifiesto no contiene todos los componentes."
        )

    return {
        "estado": "manifest_verified",
        "model_type": "content_based_tfidf",
        "total_files": len(registrados),
    }


def load_content_based_model(
    model_dir: Path,
    *,
    trusted_source: bool = False,
    expected_run_id: str | None = None,
) -> dict:

    """
    Recupera un modelo Content-Based guardado.

    Primero verifica SHA-256 y después carga
    los componentes.

    IMPORTANTE:
    joblib solo debe deserializarse cuando
    el origen del modelo sea confiable.
    """

    model_dir = Path(model_dir).resolve()

    # No permitir deserializar joblib sin una
    # autorización explícita sobre su origen.
    if trusted_source is not True:
        raise ValueError(
            "La carga de joblib requiere "
            "una fuente confiable."
        )

    # 1. Verificar integridad antes de cargar.
    verify_content_model_manifest(
        model_dir=model_dir,
    )

    # 2. Definir los archivos del modelo.
    catalog_path = model_dir / "catalog.parquet"
    vectorizer_path = model_dir / "vectorizer.joblib"
    matrix_path = model_dir / "tfidf_matrix.npz"
    metadata_path = model_dir / "metadata.json"

    # 3. Cargar los metadatos.
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            "Los metadatos del modelo son inválidos."
        )

    
    # Comprobar la procedencia de los datos
    # antes de recuperar el vectorizador.
    if expected_run_id is not None:

        try:
            esperado = str(UUID(expected_run_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "El run_id esperado es inválido."
            ) from exc

        if metadata.get("run_id") != esperado:
            raise ValueError(
                "El modelo Content-Based pertenece "
                "a un run_id diferente."
            )

    
    # 4. Recuperar catálogo y matriz.
    content_df = pd.read_parquet(
        catalog_path
    )

    matrix = load_npz(
        matrix_path
    ).tocsr()

    # 5. Recuperar el vectorizador entrenado.
    # Solo para archivos de origen confiable.
    vectorizer = joblib.load(
        vectorizer_path
    )

    # 6. Verificar la consistencia dimensional.
    if not {"movie_id", "title"}.issubset(
        content_df.columns
    ):
        raise ValueError(
            "El catálogo recuperado está incompleto."
        )

    if (
        content_df["movie_id"].isna().any()
        or content_df["movie_id"].duplicated().any()
    ):
        raise ValueError(
            "El catálogo recuperado contiene "
            "movie_id nulos o duplicados."
        )

    if matrix.shape[0] != len(content_df):
        raise ValueError(
            "La matriz no coincide con el catálogo."
        )

    if not isinstance(
        vectorizer,
        TfidfVectorizer,
    ):
        raise ValueError(
            "El vectorizador recuperado no es TF-IDF."
        )

    if matrix.shape[1] != len(
        vectorizer.vocabulary_
    ):
        raise ValueError(
            "La matriz no coincide con el vectorizador."
        )

    # 7. Verificar metadatos contra archivos reales.
    if (
        metadata.get("model_type")
        != "content_based_tfidf"
        or type(metadata.get("total_movies")) is not int
        or type(metadata.get("total_features")) is not int
        or metadata["total_movies"] != len(content_df)
        or metadata["total_features"] != matrix.shape[1]
    ):
        raise ValueError(
            "Los metadatos no coinciden "
            "con el modelo recuperado."
        )

    ids_catalogo = [
        int(movie_id)
        for movie_id in content_df["movie_id"]
    ]

    if metadata.get("movie_ids") != ids_catalogo:
        raise ValueError(
            "El orden de películas no coincide "
            "con los metadatos del modelo."
        )

    return {
        "content_df": content_df,
        "vectorizer": vectorizer,
        "tfidf_matrix": matrix,
        "total_movies": len(content_df),
        "total_features": matrix.shape[1],
    }
