
"""
Modelo colaborativo de CineMatch.

Primera etapa:
validación y preparación de interacciones
antes de construir la matriz usuario-película.
"""

import numpy as np
import pandas as pd
import json
import hashlib

from pathlib import Path
from uuid import UUID
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

def prepare_collaborative_ratings(
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara las valoraciones para el modelo.

    Reglas:
    - user_id y movie_id enteros positivos.
    - ratings entre 0.5 y 5.0.
    - sin valores nulos.
    - una interacción por usuario-película.

    No modifica el DataFrame original.
    No divide Train/Test ni entrena el modelo.
    """

    requeridas = {
        "user_id",
        "movie_id",
        "rating",
    }

    faltantes = requeridas - set(interactions.columns)

    if faltantes:
        raise ValueError(
            "Collaborative requiere columnas: "
            + ", ".join(sorted(faltantes))
        )

    if interactions.empty:
        raise ValueError(
            "Collaborative no contiene interacciones."
        )

    df = interactions[
        ["user_id", "movie_id", "rating"]
    ].copy()

    for columna in ("user_id", "movie_id"):

        valores = pd.to_numeric(
            df[columna],
            errors="coerce",
        )

        if (
            valores.isna().any()
            or not np.isfinite(
                valores.to_numpy(dtype=float)
            ).all()
            or (valores <= 0).any()
            or (valores % 1 != 0).any()
        ):
            raise ValueError(
                f"Collaborative contiene {columna} inválidos."
            )

        df[columna] = valores.astype("int64")

    ratings = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    if (
        ratings.isna().any()
        or not np.isfinite(
            ratings.to_numpy(dtype=float)
        ).all()
        or not ratings.between(0.5, 5.0).all()
    ):
        raise ValueError(
            "Collaborative contiene ratings inválidos."
        )

    df["rating"] = ratings.astype("float64")

    if df.duplicated(
        subset=["user_id", "movie_id"]
    ).any():
        raise ValueError(
            "Collaborative contiene interacciones "
            "usuario-película duplicadas."
        )

    return df.sort_values(
        by=["user_id", "movie_id"],
        kind="stable",
    ).reset_index(drop=True)

def build_user_movie_matrix(
    prepared_ratings: pd.DataFrame,
) -> dict:
    """
    Construye una matriz dispersa usuario-película.

    Requiere interacciones ya validadas mediante
    prepare_collaborative_ratings().

    Devuelve:
    - matrix: matriz CSR.
    - user_ids: identificadores de usuarios.
    - movie_ids: identificadores de películas.
    - user_to_idx: usuario -> fila.
    - movie_to_idx: película -> columna.

    No entrena modelos ni modifica archivos.
    """

    # Validar también cuando esta función
    # se utiliza de forma independiente.
    df = prepare_collaborative_ratings(
        prepared_ratings
    )

    user_ids = sorted(
        df["user_id"].unique().tolist()
    )

    movie_ids = sorted(
        df["movie_id"].unique().tolist()
    )

    user_to_idx = {
        user_id: indice
        for indice, user_id in enumerate(user_ids)
    }

    movie_to_idx = {
        movie_id: indice
        for indice, movie_id in enumerate(movie_ids)
    }

    filas = (
        df["user_id"]
        .map(user_to_idx)
        .to_numpy(dtype=np.int64)
    )

    columnas = (
        df["movie_id"]
        .map(movie_to_idx)
        .to_numpy(dtype=np.int64)
    )

    valores = df["rating"].to_numpy(
        dtype=np.float64
    )

    matriz = csr_matrix(
        (valores, (filas, columnas)),
        shape=(
            len(user_ids),
            len(movie_ids),
        ),
    )

    return {
        "matrix": matriz,
        "user_ids": user_ids,
        "movie_ids": movie_ids,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "total_users": len(user_ids),
        "total_movies": len(movie_ids),
        "total_interactions": len(df),
    }

def fit_collaborative_svd(
    matrix_data: dict,
    n_factors: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Entrena TruncatedSVD utilizando una matriz
    dispersa usuario-película.

    Ajusta el número de factores a las dimensiones
    disponibles.

    No modifica la matriz original ni guarda archivos.
    """

    if (
        isinstance(n_factors, bool)
        or not isinstance(n_factors, int)
        or n_factors < 1
    ):
        raise ValueError(
            "n_factors debe ser un entero positivo."
        )

    if "matrix" not in matrix_data:
        raise ValueError(
            "Collaborative requiere una matriz."
        )

    matrix = matrix_data["matrix"]

    if not isinstance(matrix, csr_matrix):
        raise ValueError(
            "Collaborative requiere una matriz CSR."
        )

    total_users, total_movies = matrix.shape

    if total_users < 2 or total_movies < 2:
        raise ValueError(
            "TruncatedSVD requiere al menos dos "
            "usuarios y dos películas."
        )

    # Mantener un rango conservador compatible
    # con las dimensiones de matrices pequeñas.
    max_factors = min(
        total_users,
        total_movies,
    ) - 1

    effective_factors = min(
        n_factors,
        max_factors,
    )

    svd = TruncatedSVD(
        n_components=effective_factors,
        algorithm="randomized",
        random_state=random_state,
    )

    user_factors = svd.fit_transform(
        matrix
    )

    movie_components = svd.components_

    if (
        not np.isfinite(user_factors).all()
        or not np.isfinite(movie_components).all()
    ):
        raise ValueError(
            "TruncatedSVD produjo factores no finitos."
        )

    return {
        "model": svd,
        "user_factors": user_factors,
        "movie_components": movie_components,
        "user_ids": matrix_data["user_ids"],
        "movie_ids": matrix_data["movie_ids"],
        "user_to_idx": matrix_data["user_to_idx"],
        "movie_to_idx": matrix_data["movie_to_idx"],
        "n_factors": effective_factors,
        "explained_variance_ratio": float(
            svd.explained_variance_ratio_.sum()
        ),
    }

def predict_collaborative_score(
    fitted_model: dict,
    user_id: int,
    movie_id: int,
) -> float:
    """
    Calcula la puntuación latente de una pareja
    usuario-película mediante TruncatedSVD.

    No reentrena el modelo.

    La puntuación no está calibrada como rating
    ni representa una probabilidad.
    """

    requeridos = {
        "user_factors",
        "movie_components",
        "user_to_idx",
        "movie_to_idx",
    }

    if not requeridos.issubset(fitted_model):
        raise ValueError(
            "El modelo colaborativo está incompleto."
        )

    user_to_idx = fitted_model["user_to_idx"]
    movie_to_idx = fitted_model["movie_to_idx"]

    if user_id not in user_to_idx:
        raise ValueError(
            "Usuario desconocido en el modelo colaborativo."
        )

    if movie_id not in movie_to_idx:
        raise ValueError(
            "Película desconocida en el modelo colaborativo."
        )

    user_factors = fitted_model["user_factors"]
    movie_components = fitted_model["movie_components"]

    if (
        user_factors.ndim != 2
        or movie_components.ndim != 2
        or user_factors.shape[1]
        != movie_components.shape[0]
    ):
        raise ValueError(
            "Las dimensiones del modelo "
            "colaborativo son incompatibles."
        )

    user_index = user_to_idx[user_id]
    movie_index = movie_to_idx[movie_id]

    score = float(
        np.dot(
            user_factors[user_index],
            movie_components[:, movie_index],
        )
    )

    if not np.isfinite(score):
        raise ValueError(
            "La puntuación colaborativa no es finita."
        )

    return score

def recommend_collaborative_top_n(
    fitted_model: dict,
    user_id: int,
    train_interactions: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Genera recomendaciones colaborativas Top-N.

    Calcula las puntuaciones de un usuario de forma
    vectorizada y excluye todas las películas que
    calificó en el conjunto de entrenamiento.

    No reentrena TruncatedSVD ni utiliza Test.
    """

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(
            "n debe ser un entero positivo."
        )

    requeridos = {
        "user_factors",
        "movie_components",
        "user_to_idx",
        "movie_to_idx",
        "movie_ids",
    }

    if not requeridos.issubset(fitted_model):
        raise ValueError(
            "El modelo colaborativo está incompleto."
        )

    if not {"user_id", "movie_id"}.issubset(
        train_interactions.columns
    ):
        raise ValueError(
            "Train requiere user_id y movie_id."
        )

    user_to_idx = fitted_model["user_to_idx"]

    if user_id not in user_to_idx:
        raise ValueError(
            "Usuario desconocido en el modelo colaborativo."
        )

    user_factors = fitted_model["user_factors"]
    movie_components = fitted_model["movie_components"]
    movie_ids = fitted_model["movie_ids"]

    if (
        user_factors.ndim != 2
        or movie_components.ndim != 2
        or user_factors.shape[1] != movie_components.shape[0]
        or movie_components.shape[1] != len(movie_ids)
    ):
        raise ValueError(
            "El modelo colaborativo tiene "
            "dimensiones incompatibles."
        )

    indice_usuario = user_to_idx[user_id]

    # Una operación vectorizada produce una
    # puntuación para todas las películas del modelo.
    scores = np.asarray(
        user_factors[indice_usuario]
        @ movie_components
    ).ravel()

    if not np.isfinite(scores).all():
        raise ValueError(
            "El modelo produjo puntuaciones no finitas."
        )

    resultado = pd.DataFrame({
        "movie_id": movie_ids,
        "collaborative_score": scores,
    })

    # Excluir todas las películas calificadas
    # por el usuario, incluso las que no le gustaron.
    vistas = set(
        train_interactions.loc[
            train_interactions["user_id"] == user_id,
            "movie_id",
        ].dropna().tolist()
    )

    resultado = resultado.loc[
        ~resultado["movie_id"].isin(vistas)
    ]

    resultado = resultado.sort_values(
        by=["collaborative_score", "movie_id"],
        ascending=[False, True],
        kind="stable",
    )

    return resultado.head(n).reset_index(drop=True)

def save_collaborative_model(
    fitted_model: dict,
    output_dir: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Path]:
    """
    Guarda los factores de un modelo colaborativo
    TruncatedSVD previamente entrenado.

    No guarda objetos Python mediante joblib.
    No permite sobrescribir el directorio.
    """

    requeridos = {
        "user_factors",
        "movie_components",
        "user_ids",
        "movie_ids",
    }

    if not requeridos.issubset(fitted_model):
        raise ValueError(
            "El modelo colaborativo está incompleto."
        )

    usuarios = np.asarray(fitted_model["user_ids"])
    peliculas = np.asarray(fitted_model["movie_ids"])

    factores_usuario = np.asarray(
        fitted_model["user_factors"],
        dtype=np.float64,
    )

    componentes_pelicula = np.asarray(
        fitted_model["movie_components"],
        dtype=np.float64,
    )

    # Verificar dimensiones.
    if (
        factores_usuario.ndim != 2
        or componentes_pelicula.ndim != 2
        or factores_usuario.shape[0] != len(usuarios)
        or componentes_pelicula.shape[1] != len(peliculas)
        or factores_usuario.shape[1]
        != componentes_pelicula.shape[0]
    ):
        raise ValueError(
            "Los factores colaborativos tienen "
            "dimensiones incompatibles."
        )

    if (
        len(usuarios) == 0
        or len(peliculas) == 0
        or not np.isfinite(factores_usuario).all()
        or not np.isfinite(componentes_pelicula).all()
    ):
        raise ValueError(
            "El modelo colaborativo contiene "
            "factores inválidos."
        )

    # Identificadores numéricos sin objetos Python.
    try:
        usuarios = usuarios.astype(np.int64)
        peliculas = peliculas.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "El modelo contiene identificadores inválidos."
        ) from exc

    if (
        (usuarios <= 0).any()
        or (peliculas <= 0).any()
        or len(np.unique(usuarios)) != len(usuarios)
        or len(np.unique(peliculas)) != len(peliculas)
    ):
        raise ValueError(
            "El modelo contiene identificadores "
            "duplicados o no positivos."
        )

    # Registrar la ejecución de origen.
    run_id_normalizado = None

    if run_id is not None:
        try:
            run_id_normalizado = str(UUID(run_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(
                "El modelo requiere un run_id UUID válido."
            ) from exc

    output_dir = Path(output_dir)

    if output_dir.exists():
        raise FileExistsError(
            "El directorio del modelo ya existe."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    rutas = {
        "factors": output_dir / "factors.npz",
        "metadata": output_dir / "metadata.json",
    }

    np.savez_compressed(
        rutas["factors"],
        user_factors=factores_usuario,
        movie_components=componentes_pelicula,
        user_ids=usuarios,
        movie_ids=peliculas,
    )

    metadata = {
        "model_type": "collaborative_truncated_svd",
        "run_id": run_id_normalizado,
        "total_users": len(usuarios),
        "total_movies": len(peliculas),
        "n_factors": factores_usuario.shape[1],
    }


    rutas["metadata"].write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Construir las huellas de los componentes
    # después de completar sus escrituras.
    manifest = build_collaborative_manifest(
        model_dir=output_dir,
    )

    manifest_path = output_dir / "_manifest.json"

    # No sobrescribir un manifiesto existente.
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

def load_collaborative_model(
    model_dir: Path,
    *,
    expected_run_id: str | None = None,
) -> dict:
    """
    Recupera los factores de un modelo colaborativo
    previamente guardado.

    Comprueba:
    - Existencia de archivos.
    - Identidad de la ejecución.
    - Dimensiones y valores de los factores.
    - Identificadores únicos y positivos.

    No vuelve a entrenar TruncatedSVD.
    No utiliza deserialización pickle.
    """


    model_dir = Path(model_dir).resolve()

    # Verificar los archivos contra el manifiesto
    # antes de recuperar cualquier factor.
    verificacion = verify_collaborative_manifest(
        model_dir=model_dir,
    )

    if verificacion["estado"] != "manifest_verified":
        raise ValueError(
            "La integridad del modelo colaborativo "
            "no pudo verificarse."
        )

    factors_path = model_dir / "factors.npz"
    metadata_path = model_dir / "metadata.json"


    if not factors_path.is_file():
        raise FileNotFoundError(
            "No se encontró factors.npz."
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "No se encontró metadata.json."
        )

    # 1. Recuperar metadatos.
    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Los metadatos colaborativos son inválidos."
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(
            "Los metadatos deben ser un objeto JSON."
        )

    if metadata.get("model_type") != (
        "collaborative_truncated_svd"
    ):
        raise ValueError(
            "El tipo de modelo colaborativo es inválido."
        )

    # 2. Comprobar la ejecución de origen.
    if expected_run_id is not None:

        try:
            esperado = str(UUID(expected_run_id))
        except (
            ValueError,
            TypeError,
            AttributeError,
        ) as exc:
            raise ValueError(
                "El run_id esperado es inválido."
            ) from exc

        if metadata.get("run_id") != esperado:
            raise ValueError(
                "El modelo colaborativo pertenece "
                "a un run_id diferente."
            )

    # 3. Recuperar factores sin permitir pickle.
    campos = {
        "user_factors",
        "movie_components",
        "user_ids",
        "movie_ids",
    }

    try:
        with np.load(
            factors_path,
            allow_pickle=False,
        ) as datos:

            if set(datos.files) != campos:
                raise ValueError(
                    "El archivo de factores está incompleto."
                )

            user_factors = datos[
                "user_factors"
            ].copy()

            movie_components = datos[
                "movie_components"
            ].copy()

            user_ids = datos[
                "user_ids"
            ].copy()

            movie_ids = datos[
                "movie_ids"
            ].copy()

    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(
            "No se pudieron recuperar los factores."
        ) from exc

    # 4. Validar las estructuras recuperadas.
    if (
        user_factors.ndim != 2
        or movie_components.ndim != 2
        or user_ids.ndim != 1
        or movie_ids.ndim != 1
        or user_factors.shape[0] != len(user_ids)
        or movie_components.shape[1] != len(movie_ids)
        or user_factors.shape[1]
        != movie_components.shape[0]
    ):
        raise ValueError(
            "Las dimensiones recuperadas "
            "son incompatibles."
        )

    if (
        not np.issubdtype(user_factors.dtype, np.number)
        or not np.issubdtype(
            movie_components.dtype, np.number
        )
        or not np.isfinite(user_factors).all()
        or not np.isfinite(movie_components).all()
    ):
        raise ValueError(
            "El modelo contiene factores inválidos."
        )

    for ids in (user_ids, movie_ids):

        if not np.issubdtype(
            ids.dtype,
            np.integer,
        ):
            raise ValueError(
                "Los identificadores deben ser enteros."
            )

        if (
            (ids <= 0).any()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError(
                "Existen identificadores inválidos "
                "o duplicados."
            )

    # 5. Comparar dimensiones con metadata.
    if (
        type(metadata.get("total_users")) is not int
        or type(metadata.get("total_movies")) is not int
        or type(metadata.get("n_factors")) is not int
        or metadata["total_users"] != len(user_ids)
        or metadata["total_movies"] != len(movie_ids)
        or metadata["n_factors"]
        != user_factors.shape[1]
    ):
        raise ValueError(
            "Los metadatos no coinciden "
            "con los factores recuperados."
        )

    # 6. Reconstruir los índices.
    usuarios = [
        int(valor) for valor in user_ids
    ]

    peliculas = [
        int(valor) for valor in movie_ids
    ]

    return {
        "user_factors": user_factors,
        "movie_components": movie_components,
        "user_ids": usuarios,
        "movie_ids": peliculas,
        "user_to_idx": {
            usuario: indice
            for indice, usuario in enumerate(
                usuarios
            )
        },
        "movie_to_idx": {
            pelicula: indice
            for indice, pelicula in enumerate(
                peliculas
            )
        },
        "n_factors": user_factors.shape[1],
    }

def build_collaborative_manifest(
    model_dir: Path,
) -> dict:
    """
    Construye el manifiesto SHA-256 del modelo
    colaborativo TruncatedSVD.

    Registra factors.npz y metadata.json.

    No modifica archivos ni carga el modelo.
    """

    model_dir = Path(model_dir).resolve()

    componentes = {
        "factors": "factors.npz",
        "metadata": "metadata.json",
    }

    entradas = []

    for componente, nombre in componentes.items():

        ruta = model_dir / nombre

        if not ruta.is_file():
            raise FileNotFoundError(
                f"Falta un componente colaborativo: {nombre}"
            )

        tamanio = ruta.stat().st_size

        if tamanio <= 0:
            raise ValueError(
                f"El componente está vacío: {nombre}"
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
        "model_type": "collaborative_truncated_svd",
        "algorithm": "sha256",
        "total_files": len(entradas),
        "files": entradas,
    }

def verify_collaborative_manifest(
    model_dir: Path,
) -> dict:
    """
    Verifica el manifiesto SHA-256 de un modelo
    colaborativo TruncatedSVD.

    No carga los factores con NumPy.
    No modifica archivos ni entrena modelos.
    """

    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / "_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Collaborative requiere _manifest.json."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "El manifiesto colaborativo es inválido."
        ) from exc

    if not isinstance(manifest, dict):
        raise ValueError(
            "El manifiesto debe ser un objeto JSON."
        )

    if (
        manifest.get("model_type")
        != "collaborative_truncated_svd"
        or manifest.get("algorithm") != "sha256"
    ):
        raise ValueError(
            "El manifiesto contiene un tipo "
            "o algoritmo inválido."
        )

    esperados = {
        "factors": "factors.npz",
        "metadata": "metadata.json",
    }

    entradas = manifest.get("files")

    if (
        not isinstance(entradas, list)
        or len(entradas) != len(esperados)
        or type(manifest.get("total_files")) is not int
        or manifest["total_files"] != len(esperados)
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
        bytes_esperados = entrada.get("bytes")
        hash_esperado = entrada.get("sha256")

        if (
            componente not in esperados
            or nombre != esperados[componente]
            or componente in registrados
        ):
            raise ValueError(
                "El manifiesto contiene componentes "
                "inesperados o duplicados."
            )

        registrados.add(componente)

        if (
            type(bytes_esperados) is not int
            or bytes_esperados <= 0
            or not isinstance(hash_esperado, str)
            or len(hash_esperado) != 64
            or any(
                c not in "0123456789abcdef"
                for c in hash_esperado
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
                f"Falta un componente colaborativo: {nombre}"
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
            "El manifiesto no contiene todos "
            "los componentes obligatorios."
        )

    return {
        "estado": "manifest_verified",
        "model_type": "collaborative_truncated_svd",
        "total_files": len(registrados),
    }
