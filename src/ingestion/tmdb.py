import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import date

from src.utils.supabase_client import (
    build_storage_path,
    upload_file_to_storage,
)

from src.validation.data_quality import (
    validate_file_exists,
    validate_json_file,
)

from src.utils.paths import (
    PROJECT_ROOT,
    BRONZE_TMDB_DIR,
)

from src.utils.config import TMDB_BASE_URL
from src.utils.movie_ids import load_tmdb_ids_from_movielens


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

        existe = validate_file_exists(archivo)

        if not existe:
            invalidos.append({
                "archivo": archivo.name,
                "error": "file_not_found",
            })
            continue

        resultado_json = validate_json_file(archivo)

        if not resultado_json["valid"]:
            invalidos.append({
                "archivo": archivo.name,
                "error": resultado_json.get("error"),
            })
            continue

        # Validación específica de TMDb:
        # la respuesta debe ser un objeto JSON con un id.
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
                invalidos.append({
                    "archivo": archivo.name,
                    "error": "invalid_tmdb_structure",
                })

        except Exception as error:
            invalidos.append({
                "archivo": archivo.name,
                "error": str(error),
            })

    return {
        "archivos_totales": len(archivos),
        "archivos_validos": validos,
        "archivos_invalidos": invalidos,
    }
    

### FUNCION QUE SUBE A SUPABASE STORAGE LOS JSON DE TMDB

def upload_tmdb_to_supabase(
    archivos: list[Path],
    ruta_metadata: Path,
) -> dict:

    ingestion_date = date.today().isoformat()

    resultados = {
        "archivos_subidos": 0,
        "metadata_subida": False,
        "errores": [],
    }

    # Subir películas
    for archivo in archivos:

        remote_path = build_storage_path(
            source="tmdb",
            ingestion_date=ingestion_date,
            filename=archivo.name,
        )

        try:
            upload_file_to_storage(
                bucket="Bronze",
                local_path=archivo,
                remote_path=remote_path,
                upsert=False,
            )

            resultados["archivos_subidos"] += 1

        except Exception as error:
            resultados["errores"].append({
                "archivo": archivo.name,
                "error": str(error),
            })

    # Subir metadata
    remote_metadata = build_storage_path(
        source="tmdb",
        ingestion_date=ingestion_date,
        filename=ruta_metadata.name,
    )

    try:
        upload_file_to_storage(
            bucket="Bronze",
            local_path=ruta_metadata,
            remote_path=remote_metadata,
            upsert=False,
        )

        resultados["metadata_subida"] = True

    except Exception as error:
        resultados["errores"].append({
            "archivo": ruta_metadata.name,
            "error": str(error),
        })

    return resultados


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
        "detalle_invalidos": resultado_validacion[
            "archivos_invalidos"
        ],
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

    ruta_metadata = save_tmdb_metadata(
        resultado_extraccion,
        resultado_validacion,
        fecha_inicio,
        fecha_fin,
    )
    
    archivos_tmdb = list(
    BRONZE_TMDB_DIR.glob("movie_*.json")
    )

    print("\nSubiendo TMDb a Supabase Storage...")

    resultado_supabase = upload_tmdb_to_supabase(
        archivos_tmdb,
        ruta_metadata,
    )

    print(
        "Archivos subidos:",
        resultado_supabase["archivos_subidos"]
    )

    print(
        "Metadata subida:",
        resultado_supabase["metadata_subida"]
    )

    print(
        "Errores Supabase:",
        len(resultado_supabase["errores"])
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