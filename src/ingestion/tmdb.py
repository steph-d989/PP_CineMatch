import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.paths import (
    PROJECT_ROOT,
    BRONZE_MOVIELENS_DIR,
    BRONZE_TMDB_DIR,
)

from src.utils.config import TMDB_BASE_URL
from src.utils.movie_ids import load_tmdb_ids_from_movielens


TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"


### FUNCIÓN PARA LEER CREDENCIALES

def load_tmdb_token() -> str:  # Carga el Token desde .env

    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("TMDB_TOKEN")

    if not token:
        raise ValueError(
            "No se encontró TMDB_TOKEN en el archivo .env"
        )

    return token


### FUNCION QUE CONSULTA UNA PELICULA Y DEVUELVE UN JSON

def extract_tmdb_movie(
    tmdb_id: int,
    token: str,
    language: str = "es-ES",
) -> dict:

    headers = {
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
    }

    params = {
        "language": language,
        "append_to_response": "credits,keywords",
    }

    url = f"{TMDB_BASE_URL}/{tmdb_id}"

    respuesta = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    respuesta.raise_for_status()

    return respuesta.json()


### FUNCIÓN QUE GUARDA LA RESPUESTA CRUDA DE TMBd EN BRONZE

def save_tmdb_raw(
    tmdb_id: int,
    data: dict,
) -> Path:

    BRONZE_TMDB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archivo_salida = (
        BRONZE_TMDB_DIR
        / f"movie_{tmdb_id}.json"
    )

    with open(
        archivo_salida,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            data,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    return archivo_salida


### FUNCION PARA EL PROCESAMIENTO DE VARIAS PELICULAS Y NP EXTRAE LAS QUE YA EXISTEN

def extract_tmdb_movies(
    tmdb_ids: list[int],
    token: str,
    limit: int | None = None,
    sleep_seconds: float = 0.1,
) -> dict:

    ids_procesar = (
        tmdb_ids[:limit]
        if limit is not None
        else tmdb_ids
    )

    exitosos = 0
    existentes = 0
    errores = []

    for i, tmdb_id in enumerate(
        ids_procesar,
        start=1,
    ):

        archivo_salida = (
            BRONZE_TMDB_DIR
            / f"movie_{tmdb_id}.json"
        )

        if archivo_salida.exists():
            existentes += 1

            print(
                f"[{i}/{len(ids_procesar)}] "
                f"{tmdb_id}: ya existe"
            )

            continue

        try:
            data = extract_tmdb_movie(
                tmdb_id,
                token,
            )

            save_tmdb_raw(
                tmdb_id,
                data,
            )

            exitosos += 1

            print(
                f"[{i}/{len(ids_procesar)}] "
                f"{tmdb_id}: OK"
            )

            time.sleep(sleep_seconds)

        except requests.RequestException as error:

            errores.append(
                {
                    "tmdb_id": tmdb_id,
                    "error": str(error),
                }
            )

            print(
                f"[{i}/{len(ids_procesar)}] "
                f"{tmdb_id}: ERROR"
            )

    return {
        "solicitados": len(ids_procesar),
        "exitosos": exitosos,
        "existentes": existentes,
        "errores": errores,
    }


### FUNCION PARA VALIDAR LOS JSON ALMACENADOS

def validate_tmdb() -> dict:

    archivos = list(
        BRONZE_TMDB_DIR.glob("movie_*.json")
    )

    validos = 0
    invalidos = []

    for archivo in archivos:

        try:
            with open(
                archivo,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            if isinstance(data, dict) and data.get("id"):
                validos += 1
            else:
                invalidos.append(archivo.name)

        except Exception:
            invalidos.append(archivo.name)

    return {
        "archivos_totales": len(archivos),
        "archivos_validos": validos,
        "archivos_invalidos": invalidos,
    }
    
    
### FUNCION PARA ALMACENAR LA METADATA

def save_tmdb_metadata(
    resultado_extraccion: dict,
    resultado_validacion: dict,
    fecha_inicio: datetime,
    fecha_fin: datetime,
) -> Path:

    metadata = {
        "fuente": "TMDb",
        "capa": "bronze",
        "fecha_inicio": fecha_inicio.isoformat(),
        "fecha_fin": fecha_fin.isoformat(),
        "solicitados": resultado_extraccion["solicitados"],
        "exitosos": resultado_extraccion["exitosos"],
        "existentes": resultado_extraccion["existentes"],
        "errores": len(resultado_extraccion["errores"]),
        "archivos_validos": resultado_validacion["archivos_validos"],
        "archivos_invalidos": len(
            resultado_validacion["archivos_invalidos"]
        ),
        "estado": (
            "success"
            if len(resultado_extraccion["errores"]) == 0
            else "partial"
        ),
    }

    ruta_metadata = (
        BRONZE_TMDB_DIR
        / "_metadata.json"
    )

    with open(
        ruta_metadata,
        "w",
        encoding="utf-8",
    ) as archivo:
        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    return ruta_metadata


### FUNCION QUE EJECUTA TODO

def run_tmdb_ingestion(
    limit: int | None = 10,
) -> dict:

    print("=" * 50)
    print("INICIO INGESTA TMDb")
    print("=" * 50)

    fecha_inicio = datetime.now()

    token = load_tmdb_token()

    tmdb_ids = load_tmdb_ids_from_movielens()

    print(
        "IDs TMDb disponibles:",
        len(tmdb_ids),
    )

    resultado_extraccion = extract_tmdb_movies(
        tmdb_ids,
        token,
        limit=limit,
    )

    resultado_validacion = validate_tmdb()

    fecha_fin = datetime.now()

    save_tmdb_metadata(
        resultado_extraccion,
        resultado_validacion,
        fecha_inicio,
        fecha_fin,
    )

    print("=" * 50)
    print("FIN INGESTA TMDb")
    print("=" * 50)

    return {
        "extraccion": resultado_extraccion,
        "validacion": resultado_validacion,
    }
    
if __name__ == "__main__":
    run_tmdb_ingestion()