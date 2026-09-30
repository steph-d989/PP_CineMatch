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
    BRONZE_WATCHMODE_DIR,
)

from src.utils.config import WATCHMODE_BASE_URL
from src.utils.movie_ids import load_tmdb_ids_from_movielens

MIN_CREDITS_RESERVE = 300


### FUNCION PARA CARGAR LA APIKEY

def load_watchmode_api_key() -> str:

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("WATCHMODE_API_KEY")

    if not api_key:
        raise ValueError(
            "No se encontró WATCHMODE_API_KEY en el archivo .env"
        )

    return api_key


### FUCNION PARA  CONSULTAR EL ESTADO DE LA CUOTA DE WATCHMODE

def get_watchmode_quota(api_key: str) -> dict:
    """
    Consulta el estado de la cuota disponible en Watchmode.
    """

    headers = {
        "X-API-Key": api_key,
        "accept": "application/json",
    }

    url = f"{WATCHMODE_BASE_URL}/status"

    respuesta = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    respuesta.raise_for_status()

    return respuesta.json()


### FUNCION PARA CONSULATR UNA PELICULA DE WATCHMODE. DISPONIBILIDAD EN PLATAFORMAS POR PAIS

def extract_watchmode_sources(
    tmdb_id: int,
    api_key: str,
    region: str = "PE",
) -> list:

    headers = {
        "X-API-Key": api_key,
        "accept": "application/json",
    }

    watchmode_id = f"movie-{tmdb_id}"

    url = (
        f"{WATCHMODE_BASE_URL}"
        f"/title/{watchmode_id}/sources"
    )

    params = {
        "regions": region,
    }

    respuesta = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=30,
    )

    respuesta.raise_for_status()

    return respuesta.json()


### FUNCION PARA GUARDAR EL JSON CRUDO EN BRONZE

def save_watchmode_raw(
    tmdb_id: int,
    data: list,
) -> Path:

    BRONZE_WATCHMODE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    archivo_salida = (
        BRONZE_WATCHMODE_DIR
        / f"movie_{tmdb_id}_sources.json"
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


### FUNCION PARA EXTRAER VARIAS PELICULAS DE WATCHMODE, SIN CNSULTAR LAS QUE YA SE TIENEN

def extract_watchmode_movies(
    tmdb_ids: list[int],
    api_key: str,
    region: str = "PE",
    limit: int | None = None,
    sleep_seconds: float = 0.5,
) -> dict:

    ids_procesar = (
        tmdb_ids[:limit]
        if limit is not None
        else tmdb_ids
    )

    exitosos = 0
    existentes = 0
    sin_disponibilidad = 0
    errores = []

    for i, tmdb_id in enumerate(
        ids_procesar,
        start=1,
    ):

        archivo_salida = (
            BRONZE_WATCHMODE_DIR
            / f"movie_{tmdb_id}_sources.json"
        )

        if archivo_salida.exists():
            existentes += 1

            print(
                f"[{i}/{len(ids_procesar)}] "
                f"{tmdb_id}: ya existe"
            )

            continue

        try:
            data = extract_watchmode_sources(
                tmdb_id,
                api_key,
                region=region,
            )

            save_watchmode_raw(
                tmdb_id,
                data,
            )

            exitosos += 1

            if data:
                print(
                    f"[{i}/{len(ids_procesar)}] "
                    f"{tmdb_id}: "
                    f"{len(data)} fuentes"
                )
            else:
                sin_disponibilidad += 1

                print(
                    f"[{i}/{len(ids_procesar)}] "
                    f"{tmdb_id}: "
                    f"sin disponibilidad en {region}"
                )

            time.sleep(sleep_seconds)

        except requests.HTTPError as error:

            status_code = (
                error.response.status_code
                if error.response is not None
                else None
            )

            errores.append(
                {
                    "tmdb_id": tmdb_id,
                    "status_code": status_code,
                    "error": str(error),
                }
            )

            print(
                f"[{i}/{len(ids_procesar)}] "
                f"{tmdb_id}: ERROR HTTP {status_code}"
            )

            if status_code == 429:
                print(
                    "Límite de Watchmode alcanzado. "
                    "Se detiene la ingesta."
                )
                break

        except requests.RequestException as error:

            errores.append(
                {
                    "tmdb_id": tmdb_id,
                    "status_code": None,
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
        "sin_disponibilidad": sin_disponibilidad,
        "errores": errores,
        "region": region,
    }

### FUNCION PARA VALIDAR EL JSON, SI HAY [] ES VALIDO PORQUE PUEDE NO HABER DISPONIBILIDAD EN PERÚ

def validate_watchmode() -> dict:

    archivos = list(
        BRONZE_WATCHMODE_DIR.glob(
            "movie_*_sources.json"
        )
    )

    validos = 0
    vacios = 0
    invalidos = []

    for archivo in archivos:

        try:
            with open(
                archivo,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            if isinstance(data, list):
                validos += 1

                if len(data) == 0:
                    vacios += 1

            else:
                invalidos.append(
                    archivo.name
                )

        except Exception:
            invalidos.append(
                archivo.name
            )

    return {
        "archivos_totales": len(archivos),
        "archivos_validos": validos,
        "archivos_vacios": vacios,
        "archivos_invalidos": invalidos,
    }
    

### FUNCION PARA GUARDAR METADATA

def save_watchmode_metadata(
    resultado_extraccion: dict,
    resultado_validacion: dict,
    fecha_inicio: datetime,
    fecha_fin: datetime,
) -> Path:

    metadata = {
        "fuente": "Watchmode",
        "capa": "bronze",
        "region": resultado_extraccion["region"],
        "fecha_inicio": fecha_inicio.isoformat(),
        "fecha_fin": fecha_fin.isoformat(),
        "solicitados": resultado_extraccion["solicitados"],
        "exitosos": resultado_extraccion["exitosos"],
        "existentes": resultado_extraccion["existentes"],
        "sin_disponibilidad": (
            resultado_extraccion["sin_disponibilidad"]
        ),
        "errores": len(
            resultado_extraccion["errores"]
        ),
        "archivos_validos": (
            resultado_validacion["archivos_validos"]
        ),
        "archivos_vacios": (
            resultado_validacion["archivos_vacios"]
        ),
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
        BRONZE_WATCHMODE_DIR
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

def run_watchmode_ingestion(
    limit: int | None = 10,
    region: str = "PE",
) -> dict:

    print("=" * 50)
    print("INICIO INGESTA WATCHMODE")
    print("=" * 50)

    fecha_inicio = datetime.now()

    api_key = load_watchmode_api_key()
    
    quota = get_watchmode_quota(api_key)

    quota_total = quota.get("quota", 0)
    quota_used = quota.get("quotaUsed", 0)

    restantes = quota_total - quota_used

    print("Cuota total:", quota_total)
    print("Créditos usados:", quota_used)
    print("Créditos restantes:", restantes)

    if restantes <= MIN_CREDITS_RESERVE:
        raise RuntimeError(
            "No se ejecutará la ingesta: "
            "se alcanzó la reserva mínima de créditos Watchmode."
        )

    tmdb_ids = load_tmdb_ids_from_movielens()

    print(
        "IDs TMDb disponibles:",
        len(tmdb_ids),
    )

    resultado_extraccion = extract_watchmode_movies(
        tmdb_ids,
        api_key,
        region=region,
        limit=limit,
    )

    resultado_validacion = validate_watchmode()

    fecha_fin = datetime.now()

    save_watchmode_metadata(
        resultado_extraccion,
        resultado_validacion,
        fecha_inicio,
        fecha_fin,
    )

    print("\nResumen:")
    print(
        "Solicitados:",
        resultado_extraccion["solicitados"],
    )
    print(
        "Nuevos consultados:",
        resultado_extraccion["exitosos"],
    )
    print(
        "Ya existentes:",
        resultado_extraccion["existentes"],
    )
    print(
        "Sin disponibilidad:",
        resultado_extraccion[
            "sin_disponibilidad"
        ],
    )
    print(
        "Errores:",
        len(
            resultado_extraccion["errores"]
        ),
    )

    print("=" * 50)
    print("FIN INGESTA WATCHMODE")
    print("=" * 50)

    return {
        "extraccion": resultado_extraccion,
        "validacion": resultado_validacion,
    }
    
    
if __name__ == "__main__":
    run_watchmode_ingestion()