import json
import os
import time
import re
import shutil
from uuid import uuid4
from datetime import datetime
from pathlib import Path
import hashlib

import requests
from dotenv import load_dotenv
from datetime import date, timedelta, timezone
import pandas as pd

from src.utils.supabase_client import (
    build_storage_path,
    upload_file_to_storage,
    list_storage_filenames,
    build_watchmode_movie_path,
    get_supabase_client,
    build_watchmode_history_paths,
)

from src.validation.data_quality import (
    validate_file_exists,
    validate_json_file,
)

from src.utils.paths import (
    PROJECT_ROOT,
    BRONZE_WATCHMODE_DIR,
)

from src.utils.config import WATCHMODE_BASE_URL
from src.utils.movie_ids import load_tmdb_ids_from_movielens

MIN_CREDITS_RESERVE = 300


CREDITS_PER_REQUEST = 2


def calculate_watchmode_budget(
    quota_total: int,
    quota_used: int,
    requested_movies: int,
) -> dict:

    if quota_total < 0 or quota_used < 0:
        raise ValueError("Las cuotas no pueden ser negativas")

    if requested_movies < 0:
        raise ValueError("requested_movies no puede ser negativo")

    remaining = max(0, quota_total - quota_used)

    available = max(
        0,
        remaining - MIN_CREDITS_RESERVE
    )

    max_requests = available // CREDITS_PER_REQUEST

    allowed = min(
        requested_movies,
        max_requests
    )

    return {
        "remaining_credits": remaining,
        "available_credits": available,
        "max_requests": max_requests,
        "allowed_requests": allowed,
    }


def get_pending_watchmode_ids(
    tmdb_ids: list[int],
    limit: int | None = None,
    region: str = "PE",
    max_age_days: int = 7,
) -> dict:
    """
    Selecciona películas que requieren consulta.

    Considera integridad, procedencia y vigencia.
    No realiza llamadas a la API.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit no puede ser negativo")

    pendientes = []
    existentes = 0

    estados = {
        "missing": 0,
        "corrupt": 0,
        "region_unverified": 0,
        "unknown_provenance": 0,
        "stale": 0,
        "fresh": 0,
        "invalid_date": 0,
    }

    for tmdb_id in dict.fromkeys(tmdb_ids):

        archivo = build_local_watchmode_path(
            tmdb_id=tmdb_id,
            region=region,
        )

        if not archivo.is_file():
            estados["missing"] += 1
            pendientes.append(tmdb_id)
            continue

        validacion = validate_watchmode_local_file(
            archivo
        )

        if not validacion["valid"]:
            estados["corrupt"] += 1
            pendientes.append(tmdb_id)
            continue

        vigencia = check_watchmode_freshness(
            tmdb_id=tmdb_id,
            region=region,
            max_age_days=max_age_days,
        )

        estado = vigencia["status"]
        estados[estado] = estados.get(estado, 0) + 1

        if vigencia["fresh"]:
            existentes += 1
        else:
            pendientes.append(tmdb_id)

    seleccionados = (
        pendientes[:limit]
        if limit is not None
        else pendientes
    )

    return {
        "pendientes_totales": len(pendientes),
        "seleccionados": seleccionados,
        "existentes": existentes,
        "estados": estados,
    }

def plan_watchmode_requests(
    tmdb_ids: list[int],
    quota_total: int,
    quota_used: int,
    requested_limit: int | None = None,
    region: str = "PE",
) -> dict:
    """
    Planifica las consultas pendientes respetando
    el presupuesto de créditos de Watchmode.

    No realiza consultas a la API.
    """

    pendientes = get_pending_watchmode_ids(
        tmdb_ids=tmdb_ids,
        limit=None,
        region=region,
    )

    total_pendientes = pendientes["pendientes_totales"]

    if requested_limit is not None:
        if requested_limit < 0:
            raise ValueError(
                "requested_limit no puede ser negativo"
            )

        total_solicitado = min(
            total_pendientes,
            requested_limit,
        )
    else:
        total_solicitado = total_pendientes

    presupuesto = calculate_watchmode_budget(
        quota_total=quota_total,
        quota_used=quota_used,
        requested_movies=total_solicitado,
    )

    cantidad_permitida = presupuesto["allowed_requests"]

    seleccion = pendientes["seleccionados"][
        :cantidad_permitida
    ]

    return {
        "existentes": pendientes["existentes"],
        "pendientes_totales": total_pendientes,
        "ids_seleccionados": seleccion,
        "consultas_planificadas": len(seleccion),
        "creditos_restantes": presupuesto[
            "remaining_credits"
        ],
        "creditos_disponibles": presupuesto[
            "available_credits"
        ],
    }


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


def build_local_watchmode_path(
    tmdb_id: int,
    region: str = "PE",
) -> Path:
    """
    Construye la ruta local de una película
    según su región.
    No crea carpetas ni modifica archivos.
    """

    if (
        not isinstance(tmdb_id, int)
        or isinstance(tmdb_id, bool)
        or tmdb_id <= 0
    ):
        raise ValueError("tmdb_id debe ser un entero positivo")

    if (
        not isinstance(region, str)
        or len(region) != 2
        or not region.isalpha()
    ):
        raise ValueError("region debe tener dos letras")

    region = region.upper()

    return (
        BRONZE_WATCHMODE_DIR
        / "regions"
        / region
        / f"movie_{tmdb_id}_sources.json"
    )


### FUNCION PARA GUARDAR EL JSON CRUDO EN BRONZE


def save_watchmode_raw(
    tmdb_id: int,
    data: list,
    region: str = "PE",
) -> Path:
    """
    Guarda una respuesta Watchmode mediante escritura
    temporal y reemplazo atómico.

    No consulta la API ni Supabase.
    """

    if not isinstance(data, list) or not all(
        isinstance(fuente, dict) for fuente in data
    ):
        raise ValueError("Respuesta Watchmode inválida")

    archivo_salida = build_local_watchmode_path(
        tmdb_id=tmdb_id,
        region=region,
    )

    archivo_salida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    archivo_temporal = archivo_salida.with_name(
        archivo_salida.name + f".{uuid4().hex}.tmp"
    )

    try:
        with open(
            archivo_temporal,
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(
                data,
                archivo,
                ensure_ascii=False,
                indent=2,
            )

            archivo.flush()
            os.fsync(archivo.fileno())

        # Reemplazo dentro del mismo directorio
        archivo_temporal.replace(archivo_salida)

    finally:
        if archivo_temporal.exists():
            archivo_temporal.unlink()

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

        archivo_salida = build_local_watchmode_path(
            tmdb_id=tmdb_id,
            region=region,
        )

        if archivo_salida.is_file():

            validacion = validate_watchmode_local_file(
                archivo_salida
            )

            if not validacion["valid"]:

                print(
                    f"{tmdb_id}: archivo corrupto o inválido. "
                    "Se intentará recuperar."
                )

            else:

                vigencia = check_watchmode_freshness(
                    tmdb_id=tmdb_id,
                    region=region,
                    max_age_days=7,
                )

                if vigencia["fresh"]:

                    existentes += 1

                    print(
                        f"[{i}/{len(ids_procesar)}] "
                        f"{tmdb_id}: ya existe y está vigente"
                    )

                    continue

                print(
                    f"{tmdb_id}: requiere actualización "
                    f"({vigencia['status']})"
                )


        try:
            
            data = extract_watchmode_sources(
                tmdb_id,
                api_key,
                region=region,
            )

            # Validar la respuesta antes de guardarla
            if not isinstance(data, list) or not all(
                isinstance(fuente, dict)
                for fuente in data
            ):
                raise ValueError(
                    "Respuesta Watchmode inválida"
                )

            # Respaldar el archivo corrupto, si existe

            if archivo_salida.is_file():

                validacion_anterior = (
                    validate_watchmode_local_file(
                        archivo_salida
                    )
                )

                tipo_respaldo = (
                    "history"
                    if validacion_anterior["valid"]
                    else "corrupt"
                )

                respaldo = archivo_salida.with_name(
                    archivo_salida.name
                    + f".{tipo_respaldo}.{uuid4().hex}.bak"
                )

                shutil.copy2(
                    archivo_salida,
                    respaldo,
                )


            update_watchmode_safely(
                tmdb_id=tmdb_id,
                data=data,
                region=region,
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
        
        except ValueError as error:
            errores.append({
                "tmdb_id": tmdb_id,
                "status_code": None,
                "error": str(error),
            })

            print(f"{tmdb_id}: ERROR - {error}")

    return {
        "solicitados": len(ids_procesar),
        "exitosos": exitosos,
        "existentes": existentes,
        "sin_disponibilidad": sin_disponibilidad,
        "errores": errores,
        "region": region,
    }

### FUNCION PARA VALIDAR EL JSON, SI HAY [] ES VALIDO PORQUE PUEDE NO HABER DISPONIBILIDAD EN PERÚ

def validate_watchmode(
    region: str = "PE",
) -> dict:

    archivos = get_local_watchmode_files(
        region=region
    )

    validos = 0
    vacios = 0
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

        try:
            with open(
                archivo,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            # Regla específica de Watchmode:
            # la respuesta debe ser una lista.
            if isinstance(data, list):
                validos += 1

                if len(data) == 0:
                    vacios += 1

            else:
                invalidos.append({
                    "archivo": archivo.name,
                    "error": "invalid_watchmode_structure",
                })

        except Exception as error:
            invalidos.append({
                "archivo": archivo.name,
                "error": str(error),
            })

    return {
        "region": region.upper(),
        "archivos_totales": len(archivos),
        "archivos_validos": validos,
        "archivos_vacios": vacios,
        "archivos_invalidos": invalidos,
    }
    

### FUNCION QUE SUMBE A STORAGE LOS JSON BRONZE DE WATCHMODE Y LA METADATA

def upload_watchmode_to_supabase(
    archivos: list[Path],
    ruta_metadata: Path,
) -> dict:

    ingestion_date = date.today().isoformat()

    resultados = {
        "archivos_subidos": 0,
        "metadata_subida": False,
        "errores": [],
    }

    for archivo in archivos:
        
        
        # Validar el archivo antes de sincronizarlo
        validacion = validate_watchmode_local_file(archivo)

        if not validacion["valid"]:
            resultados["errores"].append({
                "archivo": archivo.name,
                "error": validacion["error"],
            })

            print(
                f"{archivo.name}: JSON inválido - "
                f"{validacion['error']}"
            )

            continue

        remote_path = build_storage_path(
            source="watchmode",
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

    remote_metadata = build_storage_path(
        source="watchmode",
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

def get_local_watchmode_files(
    region: str = "PE",
) -> list[Path]:
    """
    Obtiene los archivos JSON locales de Watchmode
    correspondientes a una región.

    No consulta la API ni Supabase.
    """

    # Validar la región mediante la función de rutas
    ruta_ejemplo = build_local_watchmode_path(
        tmdb_id=1,
        region=region,
    )

    carpeta_region = ruta_ejemplo.parent

    if not carpeta_region.is_dir():
        return []

    return sorted(
        archivo
        for archivo in carpeta_region.glob(
            "movie_*_sources.json"
        )
        if archivo.is_file()
    )

def get_watchmode_provenance(
    tmdb_id: int,
    region: str = "PE",
) -> dict:
    """
    Consulta la procedencia de un JSON Watchmode.

    Prioridad:
    1. Metadatos individuales verificados con SHA-256.
    2. Metadatos de migración histórica.
    3. Procedencia desconocida.

    No modifica archivos ni consulta la API.
    """

    archivo = build_local_watchmode_path(
        tmdb_id=tmdb_id,
        region=region,
    )

    # Verificar que el archivo existe
    if not archivo.is_file():
        return {
            "exists": False,
            "region_verified": False,
            "status": "missing",
            "retrieved_at": None,
        }

    region_normalizada = region.upper()

    metadata_individual = archivo.with_name(
        archivo.stem + "_metadata.json"
    )

    # 1. Consultar metadatos individuales
    if metadata_individual.is_file():
        try:
            metadata = json.loads(
                metadata_individual.read_text(
                    encoding="utf-8"
                )
            )

            # Calcular SHA-256 del JSON actual
            hash_actual = hashlib.sha256(
                archivo.read_bytes()
            ).hexdigest()

            # Verificar identidad, región y contenido
            if (
                isinstance(metadata, dict)
                and metadata.get("tmdb_id") == tmdb_id
                and metadata.get("region") == region_normalizada
                and metadata.get("region_verified") is True
                and isinstance(
                    metadata.get("retrieved_at"), str
                )
                and metadata.get("sha256") == hash_actual
            ):
                return {
                    "exists": True,
                    "region_verified": True,
                    "status": "verified",
                    "retrieved_at": metadata["retrieved_at"],
                }

        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ):
            pass

    # 2. Consultar metadatos históricos
    metadata_migracion = (
        archivo.parent / "_migration_metadata.json"
    )

    if metadata_migracion.is_file():
        try:
            metadata = json.loads(
                metadata_migracion.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(metadata, dict):
                for registro in metadata.get("records", []):
                    if (
                        isinstance(registro, dict)
                        and registro.get("archivo") == archivo.name
                    ):
                        return {
                            "exists": True,
                            "region_verified": False,
                            "status": "region_unverified",
                            "retrieved_at": None,
                        }

        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            AttributeError,
            TypeError,
        ):
            pass

    # 3. Procedencia desconocida
    return {
        "exists": True,
        "region_verified": False,
        "status": "unknown_provenance",
        "retrieved_at": None,
    }


def check_watchmode_freshness(
    tmdb_id: int,
    region: str = "PE",
    max_age_days: int = 7,
    now: datetime | None = None,
) -> dict:
    """
    Evalúa la vigencia de una consulta Watchmode.

    No consulta la API ni modifica archivos.
    """

    if max_age_days < 0:
        raise ValueError(
            "max_age_days no puede ser negativo"
        )

    procedencia = get_watchmode_provenance(
        tmdb_id=tmdb_id,
        region=region,
    )

    if not procedencia["exists"]:
        return {
            "status": "missing",
            "fresh": False,
            "age_days": None,
        }

    if not procedencia["region_verified"]:
        return {
            "status": procedencia["status"],
            "fresh": False,
            "age_days": None,
        }

    try:
        fecha_consulta = datetime.fromisoformat(
            procedencia["retrieved_at"].replace(
                "Z", "+00:00"
            )
        )

        fecha_actual = (
            now if now is not None
            else datetime.now(timezone.utc)
        )

        if (
            fecha_consulta.tzinfo is None
            or fecha_actual.tzinfo is None
        ):
            raise ValueError(
                "Las fechas deben tener zona horaria"
            )

        antiguedad = fecha_actual - fecha_consulta

        if antiguedad.total_seconds() < 0:
            return {
                "status": "invalid_date",
                "fresh": False,
                "age_days": None,
            }

        dias = antiguedad.total_seconds() / 86400

        vigente = antiguedad <= timedelta(
            days=max_age_days
        )

        return {
            "status": "fresh" if vigente else "stale",
            "fresh": vigente,
            "age_days": round(dias, 2),
        }

    except (ValueError, TypeError, AttributeError):
        return {
            "status": "invalid_date",
            "fresh": False,
            "age_days": None,
        }



def validate_watchmode_local_file(
    archivo: Path,
) -> dict:
    """
    Valida un JSON local de Watchmode.

    Acepta listas vacías como respuestas válidas.
    No modifica archivos ni consulta la API.
    """

    if not archivo.is_file():
        return {
            "valid": False,
            "error": "file_not_found",
        }

    coincidencia = re.fullmatch(
        r"movie_(\d+)_sources\.json",
        archivo.name,
    )

    if not coincidencia:
        return {
            "valid": False,
            "error": "invalid_filename",
        }

    try:
        with open(
            archivo,
            "r",
            encoding="utf-8"
        ) as f:
            datos = json.load(f)

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as error:
        return {
            "valid": False,
            "error": str(error),
        }

    if not isinstance(datos, list):
        return {
            "valid": False,
            "error": "invalid_json_structure",
        }

    if not all(
        isinstance(fuente, dict)
        for fuente in datos
    ):
        return {
            "valid": False,
            "error": "invalid_source_structure",
        }

    return {
        "valid": True,
        "tmdb_id": int(coincidencia.group(1)),
        "sources_count": len(datos),
        "empty": len(datos) == 0,
    }


def inventory_legacy_watchmode_files() -> dict:
    """
    Identifica archivos de la estructura antigua.

    No modifica ni elimina ningún archivo.
    """

    archivos = sorted(
        BRONZE_WATCHMODE_DIR.glob(
            "movie_*_sources.json"
        )
    )

    validos = []
    invalidos = []

    for archivo in archivos:
        resultado = validate_watchmode_local_file(
            archivo
        )

        registro = {
            "archivo": archivo.name,
            "ruta": str(archivo),
        }

        if resultado["valid"]:
            registro["sources_count"] = resultado[
                "sources_count"
            ]
            validos.append(registro)
        else:
            registro["error"] = resultado["error"]
            invalidos.append(registro)

    return {
        "total": len(archivos),
        "validos": validos,
        "invalidos": invalidos,
    }


def save_watchmode_provenance(
    tmdb_id: int,
    region: str = "PE",
) -> Path:
    """
    Registra la procedencia de una consulta
    exitosa de Watchmode.
    """

    from datetime import timezone

    archivo = build_local_watchmode_path(
        tmdb_id=tmdb_id,
        region=region,
    )

    if not archivo.is_file():
        raise FileNotFoundError(
            f"No existe el JSON: {archivo}"
        )

    validacion = validate_watchmode_local_file(archivo)

    if not validacion["valid"]:
        raise ValueError(
            f"JSON inválido: {validacion['error']}"
        )

    sha256 = hashlib.sha256(
        archivo.read_bytes()
    ).hexdigest()
    
    metadata = {
        "tmdb_id": tmdb_id,
        "region": region.upper(),
        "region_verified": True,
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sources_count": validacion["sources_count"],
        "sha256": sha256,
    }

    ruta_metadata = archivo.with_name(
        archivo.stem + "_metadata.json"
    )

    
    ruta_temporal = ruta_metadata.with_name(
        ruta_metadata.name + f".{uuid4().hex}.tmp"
    )

    try:
        with open(
            ruta_temporal,
            "w",
            encoding="utf-8",
        ) as archivo:
            json.dump(
                metadata,
                archivo,
                ensure_ascii=False,
                indent=2,
            )

            archivo.flush()
            os.fsync(archivo.fileno())

        ruta_temporal.replace(ruta_metadata)

    finally:
        if ruta_temporal.exists():
            ruta_temporal.unlink()

    return ruta_metadata



def update_watchmode_safely(
    tmdb_id: int,
    data: list,
    region: str = "PE",
) -> Path:
    """
    Actualiza el JSON y sus metadatos.
    Si ocurre un error, intenta restaurar
    el estado anterior de ambos archivos.
    """

    archivo = build_local_watchmode_path(
        tmdb_id=tmdb_id,
        region=region,
    )

    metadata = archivo.with_name(
        archivo.stem + "_metadata.json"
    )

    respaldo_json = (
        archivo.read_bytes() if archivo.is_file() else None
    )

    respaldo_metadata = (
        metadata.read_bytes() if metadata.is_file() else None
    )

    try:
        save_watchmode_raw(
            tmdb_id=tmdb_id,
            data=data,
            region=region,
        )

        save_watchmode_provenance(
            tmdb_id=tmdb_id,
            region=region,
        )

        return archivo


    except Exception as error_original:

        errores_restauracion = []

        try:
            if respaldo_json is not None:
                archivo.write_bytes(respaldo_json)
            elif archivo.exists():
                archivo.unlink()

        except Exception as error:
            errores_restauracion.append(
                f"JSON: {error}"
            )

        try:
            if respaldo_metadata is not None:
                metadata.write_bytes(respaldo_metadata)
            elif metadata.exists():
                metadata.unlink()

        except Exception as error:
            errores_restauracion.append(
                f"Metadata: {error}"
            )

        if errores_restauracion:
            raise RuntimeError(
                "Falló la actualización y también "
                "la restauración: "
                + "; ".join(errores_restauracion)
            ) from error_original

        raise



def plan_watchmode_sync(
    region: str = "PE",
) -> dict:
    """
    Compara los archivos locales con Supabase.

    No sube ni elimina archivos.
    No consume créditos de Watchmode.
    """

    region = region.upper()

    archivos_locales = get_local_watchmode_files(
        region=region
    )

    carpeta_remota = (
        f"watchmode/regions/{region}"
    )

    archivos_remotos = list_storage_filenames(
        bucket="Bronze",
        folder=carpeta_remota,
    )

    pendientes = []
    existentes = []
    incompletos = []
    invalidos = []

    for archivo in archivos_locales:

        validacion = validate_watchmode_local_file(
            archivo
        )

        if not validacion["valid"]:
            invalidos.append({
                "archivo": archivo.name,
                "error": validacion["error"],
            })
            continue

        # No publicar archivos de procedencia
        # regional no verificada.
        procedencia = get_watchmode_provenance(
            tmdb_id=validacion["tmdb_id"],
            region=region,
        )

        if not procedencia["region_verified"]:
            invalidos.append({
                "archivo": archivo.name,
                "error": procedencia["status"],
            })
            continue

        destino = build_watchmode_movie_path(
            tmdb_id=validacion["tmdb_id"],
            region=region,
        )

        registro = {
            "archivo": archivo.name,
            "origen": str(archivo),
            "destino": destino,
        }

        
        nombre_metadata = (
            archivo.stem + "_metadata.json"
        )

        json_existe = (
            archivo.name in archivos_remotos
        )

        metadata_existe = (
            nombre_metadata in archivos_remotos
        )

        registro["json_existe"] = json_existe
        registro["metadata_existe"] = metadata_existe

        if json_existe and metadata_existe:
            existentes.append(registro)

        elif not json_existe and not metadata_existe:
            pendientes.append(registro)

        else:
            incompletos.append(registro)


    return {
        "region": region,
        "archivos_locales": len(archivos_locales),
        "pendientes": pendientes,
        "existentes": existentes,
        "incompletos": incompletos,
        "invalidos": invalidos,
    }


def prepare_watchmode_sync_bundles(
    region: str = "PE",
) -> dict:
    """
    Prepara pares de JSON y metadatos para
    sincronización.

    No sube ni modifica archivos.
    """

    plan = plan_watchmode_sync(region=region)

    preparados = []
    errores = list(plan["invalidos"])

    for item in plan["pendientes"]:
        archivo = Path(item["origen"])

        metadata = archivo.with_name(
            archivo.stem + "_metadata.json"
        )

        if not metadata.is_file():
            errores.append({
                "archivo": archivo.name,
                "error": "metadata_missing",
            })
            continue

        try:
            datos = json.loads(
                metadata.read_text(encoding="utf-8")
            )

            hash_actual = hashlib.sha256(
                archivo.read_bytes()
            ).hexdigest()

            if (
                not isinstance(datos, dict)
                or datos.get("sha256") != hash_actual
                or datos.get("region_verified") is not True
                or datos.get("region") != plan["region"]
            ):
                raise ValueError(
                    "metadata_integrity_error"
                )

            preparados.append({
                "json_local": archivo,
                "metadata_local": metadata,
                "json_remoto": item["destino"],
                "metadata_remoto": (
                    item["destino"].removesuffix(".json")
                    + "_metadata.json"
                ),
            })

        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            errores.append({
                "archivo": archivo.name,
                "error": str(error),
            })

    return {
        "region": plan["region"],
        "preparados": preparados,
        "existentes": plan["existentes"],
        "errores": errores,
    }



def inspect_incomplete_watchmode_bundles(
    region: str = "PE",
    plan: dict | None = None,
) -> dict:
    """
    Inspecciona paquetes incompletos en Supabase.

    Permite reutilizar un plan previamente
    calculado para evitar consultas duplicadas.

    No modifica archivos ni consulta Watchmode.
    """

    if plan is None:
        plan = plan_watchmode_sync(region=region)

    elif plan.get("region") != region.upper():
        raise ValueError(
            "La región del plan no coincide"
        )

    resultados = {
        "region": plan["region"],
        "recuperables": [],
        "conflictos": [],
        "errores": [],
    }

    incompletos = plan["incompletos"]

    if not incompletos:
        return resultados

    storage = get_supabase_client().storage.from_("Bronze")

    for item in incompletos:

        archivo_local = Path(item["origen"])

        metadata_local = archivo_local.with_name(
            archivo_local.stem + "_metadata.json"
        )

        metadata_remota = (
            item["destino"].removesuffix(".json")
            + "_metadata.json"
        )

        try:
            if not metadata_local.is_file():
                raise FileNotFoundError(
                    "No existen metadatos locales"
                )

            datos_metadata = json.loads(
                metadata_local.read_text(
                    encoding="utf-8"
                )
            )

            hash_local = hashlib.sha256(
                archivo_local.read_bytes()
            ).hexdigest()

            if datos_metadata.get("sha256") != hash_local:
                raise ValueError(
                    "Los metadatos locales no coinciden"
                )

            # Caso A: JSON remoto existe,
            # pero faltan sus metadatos.
            if item["json_existe"] and not item["metadata_existe"]:

                json_remoto = storage.download(
                    item["destino"]
                )

                hash_remoto = hashlib.sha256(
                    json_remoto
                ).hexdigest()

                if hash_remoto != hash_local:
                    resultados["conflictos"].append({
                        "archivo": archivo_local.name,
                        "motivo": "remote_json_hash_mismatch",
                    })
                    continue

                resultados["recuperables"].append({
                    "archivo": archivo_local.name,
                    "tipo": "metadata_missing",
                    "local_path": str(metadata_local),
                    "remote_path": metadata_remota,
                })

            # Caso B: Metadatos remotos existen,
            # pero falta el JSON.
            elif item["metadata_existe"] and not item["json_existe"]:

                metadata_remota_bytes = storage.download(
                    metadata_remota
                )

                datos_remotos = json.loads(
                    metadata_remota_bytes
                )

                if datos_remotos.get("sha256") != hash_local:
                    resultados["conflictos"].append({
                        "archivo": archivo_local.name,
                        "motivo": "remote_metadata_hash_mismatch",
                    })
                    continue

                resultados["recuperables"].append({
                    "archivo": archivo_local.name,
                    "tipo": "json_missing",
                    "local_path": str(archivo_local),
                    "remote_path": item["destino"],
                })

        except Exception as error:
            resultados["errores"].append({
                "archivo": archivo_local.name,
                "error": str(error),
            })

    return resultados


def recover_incomplete_watchmode_bundles(
    region: str = "PE",
) -> dict:
    """
    Recupera paquetes incompletos cuya integridad
    fue comprobada.

    No consulta la API de Watchmode.
    """

    diagnostico = inspect_incomplete_watchmode_bundles(
        region=region
    )

    resultado = {
        "region": diagnostico["region"],
        "recuperados": 0,
        "fallidos": 0,
        "conflictos": diagnostico["conflictos"],
        "errores": list(diagnostico["errores"]),
    }

    for item in diagnostico["recuperables"]:

        try:
            
            ruta_local = Path(item["local_path"])

            # Calcular el hash del archivo local
            hash_local = hashlib.sha256(
                ruta_local.read_bytes()
            ).hexdigest()

            # Subir únicamente el archivo faltante
            upload_file_to_storage(
                bucket="Bronze",
                local_path=ruta_local,
                remote_path=item["remote_path"],
                upsert=False,
            )

            # Descargar para verificar la copia
            cliente = get_supabase_client()

            contenido_remoto = (
                cliente.storage
                .from_("Bronze")
                .download(item["remote_path"])
            )

            hash_remoto = hashlib.sha256(
                contenido_remoto
            ).hexdigest()

            if hash_local != hash_remoto:
                raise RuntimeError(
                    "La verificación SHA-256 falló "
                    f"para {item['archivo']}"
                )

            resultado["recuperados"] += 1


        except Exception as error:
            resultado["fallidos"] += 1

            resultado["errores"].append({
                "archivo": item["archivo"],
                "error": str(error),
            })

    return resultado



def audit_complete_watchmode_bundles(
    region: str = "PE",
    plan: dict | None = None,
) -> dict:
    """
    Audita los paquetes completos de Watchmode.

    Permite reutilizar un plan existente para
    evitar consultas duplicadas al inventario.

    Solo realiza lecturas a Supabase.
    No consulta la API de Watchmode.
    """

    region = region.upper()

    if plan is None:
        plan = plan_watchmode_sync(region=region)

    elif plan.get("region") != region:
        raise ValueError(
            "La región del plan no coincide"
        )

    resultados = {
        "region": plan["region"],
        "verificados": [],
        "conflictos": [],
        "errores": [],
    }

    if not plan["existentes"]:
        return resultados

    storage = get_supabase_client().storage.from_("Bronze")

    for item in plan["existentes"]:
        try:
            ruta_json = item["destino"]

            ruta_metadata = (
                ruta_json.removesuffix(".json")
                + "_metadata.json"
            )

            json_remoto = storage.download(ruta_json)
            metadata_remota = storage.download(ruta_metadata)

            datos = json.loads(metadata_remota)

            if not isinstance(datos, dict):
                raise ValueError(
                    "Metadatos remotos inválidos"
                )

            hash_remoto = hashlib.sha256(
                json_remoto
            ).hexdigest()

            
            validacion_nombre = re.fullmatch(
                r"movie_(\d+)_sources\.json",
                item["archivo"],
            )

            if not validacion_nombre:
                raise ValueError(
                    "Nombre de película inválido"
                )

            tmdb_id = int(validacion_nombre.group(1))

            if (
                datos.get("tmdb_id") != tmdb_id
                or datos.get("region") != plan["region"]
                or datos.get("region_verified") is not True
            ):
                resultados["conflictos"].append({
                    "archivo": item["archivo"],
                    "motivo": "metadata_identity_mismatch",
                })
                continue

            
            if datos.get("sha256") != hash_remoto:
                resultados["conflictos"].append({
                    "archivo": item["archivo"],
                    "motivo": "sha256_mismatch",
                })
                continue

            resultados["verificados"].append({
                "archivo": item["archivo"],
                "sha256": hash_remoto,
            })

        except Exception as error:
            resultados["errores"].append({
                "archivo": item["archivo"],
                "error": str(error),
            })

    return resultados


def verify_uploaded_watchmode_bundle(
    paquete: dict,
) -> dict:
    """
    Verifica que el JSON y sus metadatos
    se hayan almacenado correctamente.

    Solo realiza lecturas a Supabase.
    No consulta la API Watchmode.
    """

    storage = get_supabase_client().storage.from_("Bronze")

    archivos = [
        ("json_local", "json_remoto"),
        ("metadata_local", "metadata_remoto"),
    ]

    for clave_local, clave_remota in archivos:

        ruta_local = Path(paquete[clave_local])
        ruta_remota = paquete[clave_remota]

        hash_local = hashlib.sha256(
            ruta_local.read_bytes()
        ).hexdigest()

        contenido_remoto = storage.download(
            ruta_remota
        )

        hash_remoto = hashlib.sha256(
            contenido_remoto
        ).hexdigest()

        if hash_local != hash_remoto:
            return {
                "valid": False,
                "error": "sha256_mismatch",
                "archivo": ruta_remota,
            }

    return {
        "valid": True,
        "error": None,
    }


def sync_watchmode_bundles(
    region: str = "PE",
) -> dict:
    """
    Sincroniza paquetes JSON + metadatos
    preparados hacia Supabase Storage.

    Verifica ambos archivos después de subirlos.
    No consulta la API de Watchmode.
    """

    plan = prepare_watchmode_sync_bundles(
        region=region
    )

    resultados = {
        "region": plan["region"],
        "paquetes_subidos": 0,
        "paquetes_fallidos": 0,
        "errores": list(plan["errores"]),
    }

    for paquete in plan["preparados"]:

        try:
            # 1. Subir JSON de disponibilidad
            upload_file_to_storage(
                bucket="Bronze",
                local_path=paquete["json_local"],
                remote_path=paquete["json_remoto"],
                upsert=False,
            )

            # 2. Subir metadatos de procedencia
            upload_file_to_storage(
                bucket="Bronze",
                local_path=paquete["metadata_local"],
                remote_path=paquete["metadata_remoto"],
                upsert=False,
            )

            # 3. Verificar integridad de ambas subidas
            verificacion = verify_uploaded_watchmode_bundle(
                paquete
            )

            if not verificacion["valid"]:
                raise RuntimeError(
                    "Verificación del paquete fallida: "
                    f"{verificacion['error']} - "
                    f"{verificacion.get('archivo', 'desconocido')}"
                )

            # Solo contabilizar paquetes verificados
            resultados["paquetes_subidos"] += 1

            print(
                f"{paquete['json_local'].name}: "
                "sincronizado y verificado"
            )

        except Exception as error:

            resultados["paquetes_fallidos"] += 1

            resultados["errores"].append({
                "archivo": paquete["json_local"].name,
                "error": str(error),
            })

            print(
                f"{paquete['json_local'].name}: "
                f"ERROR - {error}"
            )

    # Evaluar el resultado global fuera del bucle
    if (
        resultados["errores"]
        or resultados["paquetes_fallidos"] > 0
    ):
        resultados["estado"] = "partial"
    else:
        resultados["estado"] = "success"

    return resultados


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

    
    tmdb_ids = load_tmdb_ids_from_movielens()

    print("IDs TMDb disponibles:", len(tmdb_ids))

    plan = plan_watchmode_requests(
        tmdb_ids=tmdb_ids,
        quota_total=quota_total,
        quota_used=quota_used,
        requested_limit=limit,
        region=region,
    )

    print("Archivos locales existentes:", plan["existentes"])
    print("Películas pendientes:", plan["pendientes_totales"])
    print("Consultas planificadas:", plan["consultas_planificadas"])
    print("Créditos disponibles:", plan["creditos_disponibles"])

    if plan["consultas_planificadas"] == 0:

        print(
            "No hay consultas autorizadas "
            "por el presupuesto de créditos."
        )

        return {
            "estado": "skipped",
            "motivo": "sin_consultas_autorizadas",
            "plan": plan,
            "extraccion": None,
            "validacion": None,
            "supabase": None,
        }

    resultado_extraccion = extract_watchmode_movies(
        tmdb_ids=plan["ids_seleccionados"],
        api_key=api_key,
        region=region,
        limit=None,
    )


    resultado_validacion = validate_watchmode(
        region=region
    )

    fecha_fin = datetime.now()

    ruta_metadata = save_watchmode_metadata(
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


    errores_extraccion = len(
        resultado_extraccion["errores"]
    )

    errores_validacion = len(
        resultado_validacion["archivos_invalidos"]
    )

    total_errores = (
        errores_extraccion
        + errores_validacion
    )

    estado = (
        "success"
        if total_errores == 0
        else "partial"
    )

    return {
        "estado": estado,
        "fuente": "Watchmode",
        "region": region.upper(),
        "extraccion": resultado_extraccion,
        "validacion": resultado_validacion,
        "metadata_local": str(ruta_metadata),
        "total_errores": total_errores,
    }

    


def compare_watchmode_versions(
    region: str = "PE",
    plan: dict | None = None,
    auditoria: dict | None = None,
) -> dict:
    """
    Compara versiones locales y remotas.

    Reutiliza el plan y la auditoría cuando
    son proporcionados por el coordinador.

    No modifica archivos ni consulta Watchmode.
    """

    region = region.upper()

    if plan is None:
        plan = plan_watchmode_sync(region=region)
    elif plan.get("region") != region:
        raise ValueError(
            "La región del plan no coincide"
        )

    if auditoria is None:
        auditoria = audit_complete_watchmode_bundles(
            region=region,
            plan=plan,
        )
    elif auditoria.get("region") != region:
        raise ValueError(
            "La región de la auditoría no coincide"
        )

    archivos_verificados = {
        item["archivo"]: item
        for item in auditoria["verificados"]
    }

    resultados = {
        "region": region.upper(),
        "iguales": [],
        "diferentes": [],
        "errores": [],
    }

    if not archivos_verificados:
        return resultados

    storage = get_supabase_client().storage.from_("Bronze")

    for item in plan["existentes"]:
        nombre = item["archivo"]

        if nombre not in archivos_verificados:
            continue

        try:
            archivo_local = Path(item["origen"])

            metadata_local = archivo_local.with_name(
                archivo_local.stem + "_metadata.json"
            )

            datos_locales = json.loads(
                metadata_local.read_text(
                    encoding="utf-8"
                )
            )

            ruta_metadata_remota = (
                item["destino"].removesuffix(".json")
                + "_metadata.json"
            )

            datos_remotos = json.loads(
                storage.download(ruta_metadata_remota)
            )

            hash_local = hashlib.sha256(
                archivo_local.read_bytes()
            ).hexdigest()

            hash_remoto = archivos_verificados[nombre][
                "sha256"
            ]

            if hash_local == hash_remoto:
                resultados["iguales"].append(nombre)
                continue

            fecha_local = datetime.fromisoformat(
                datos_locales["retrieved_at"]
            )

            fecha_remota = datetime.fromisoformat(
                datos_remotos["retrieved_at"]
            )

            if (
                fecha_local.tzinfo is None
                or fecha_remota.tzinfo is None
            ):
                raise ValueError(
                    "Fechas sin zona horaria"
                )


            # Determinar el orden temporal de versiones
            if fecha_local > fecha_remota:
                orden_temporal = "local_newer"

            elif fecha_local < fecha_remota:
                orden_temporal = "remote_newer"

            else:
                orden_temporal = "same_timestamp_conflict"

            resultados["diferentes"].append({
                "archivo": nombre,
                "hash_local": hash_local,
                "hash_remoto": hash_remoto,
                "fecha_local": fecha_local.isoformat(),
                "fecha_remota": fecha_remota.isoformat(),
                "local_mas_reciente": (
                    fecha_local > fecha_remota
                ),
                "orden_temporal": orden_temporal,
            })


        except Exception as error:
            resultados["errores"].append({
                "archivo": nombre,
                "error": str(error),
            })

    return resultados


class WatchmodeBackupError(RuntimeError):
    """Error de respaldo con información de recuperación."""

    def __init__(
        self,
        message: str,
        tmdb_id: int,
        region: str,
        version_id: str,
        rutas_historicas: dict,
    ):
        super().__init__(message)

        self.tmdb_id = tmdb_id
        self.region = region
        self.version_id = version_id
        self.rutas_historicas = rutas_historicas


def backup_watchmode_remote_version(
    tmdb_id: int,
    region: str = "PE",
    expected_sha256: str | None = None,
) -> dict:
    """
    Respalda una versión remota de Watchmode
    antes de actualizarla.

    Conserva los originales.
    No consulta la API Watchmode.
    """

    region = region.upper()

    rutas_actuales = {
        "json": build_watchmode_movie_path(
            tmdb_id=tmdb_id,
            region=region,
        ),
    }

    rutas_actuales["metadata"] = (
        rutas_actuales["json"].removesuffix(".json")
        + "_metadata.json"
    )

    version_id = str(uuid4())

    rutas_historicas = build_watchmode_history_paths(
        tmdb_id=tmdb_id,
        region=region,
        version_id=version_id,
    )

    # Verificación de concurrencia previa al respaldo
    if expected_sha256 is not None:

        verificacion = verify_watchmode_remote_unchanged(
            tmdb_id=tmdb_id,
            expected_sha256=expected_sha256,
            region=region,
        )

        if not verificacion["unchanged"]:
            raise RuntimeError(
                "Respaldo bloqueado: la versión remota "
                "no coincide con la versión planificada. "
                f"Motivo: {verificacion['reason']}"
            )

    storage = get_supabase_client().storage.from_("Bronze")

    # Leer y validar ambos archivos originales
    json_original = storage.download(
        rutas_actuales["json"]
    )

    metadata_original = storage.download(
        rutas_actuales["metadata"]
    )

    datos_metadata = json.loads(metadata_original)

    # Calcular la huella digital del JSON descargado.
    hash_original = hashlib.sha256(
        json_original
    ).hexdigest()
    
    # Confirmar que el contenido descargado corresponde
    # a la versión que autorizamos para el respaldo.
    if (
        expected_sha256 is not None
        and hash_original != expected_sha256.lower()
    ):
        raise RuntimeError(
            "Respaldo bloqueado: el contenido remoto "
            "cambió después de la verificación inicial"
        )

    if (
        not isinstance(datos_metadata, dict)
        or datos_metadata.get("tmdb_id") != tmdb_id
        or datos_metadata.get("region") != region
        or datos_metadata.get("region_verified") is not True
        or datos_metadata.get("sha256") != hash_original
    ):
        raise ValueError(
            "La versión remota no superó "
            "la validación de integridad"
        )

    

    datos_metadata = json.loads(metadata_original)

    # 1. Calcular SHA-256 del contenido descargado.
    hash_original = hashlib.sha256(
        json_original
    ).hexdigest()

    # 2. Validar identidad e integridad del paquete remoto.
    if (
        not isinstance(datos_metadata, dict)
        or datos_metadata.get("tmdb_id") != tmdb_id
        or datos_metadata.get("region") != region
        or datos_metadata.get("region_verified") is not True
        or datos_metadata.get("sha256") != hash_original
    ):
        raise ValueError(
            "La versión remota no superó "
            "la validación de integridad"
        )

    # 3. Comprobar que sigue siendo la versión planificada.
    if (
        expected_sha256 is not None
        and hash_original != expected_sha256.lower()
    ):
        raise RuntimeError(
            "Respaldo bloqueado: el contenido remoto "
            "cambió después de la verificación inicial"
        )

    # 4. Crear el respaldo histórico.
    try:
        storage.copy(
            rutas_actuales["json"],
            rutas_historicas["json"],
        )

        storage.copy(
            rutas_actuales["metadata"],
            rutas_historicas["metadata"],
        )

    except Exception as error:
        raise WatchmodeBackupError(
            message=f"Respaldo histórico incompleto: {error}",
            tmdb_id=tmdb_id,
            region=region,
            version_id=version_id,
            rutas_historicas=rutas_historicas,
        ) from error


    # Verificar contenido histórico
    json_historico = storage.download(
        rutas_historicas["json"]
    )

    metadata_historica = storage.download(
        rutas_historicas["metadata"]
    )

    if (
        hashlib.sha256(json_historico).hexdigest()
        != hash_original
        or metadata_historica != metadata_original
    ):
        raise RuntimeError(
            "Falló la verificación del respaldo"
        )

    return {
        "tmdb_id": tmdb_id,
        "region": region,
        "version_id": version_id,
        "rutas_historicas": rutas_historicas,
        "sha256": hash_original,
        "estado": "verified",
    }


def inspect_watchmode_backup(
    error: WatchmodeBackupError,
) -> dict:
    """
    Inspecciona un respaldo histórico fallido.

    No modifica Supabase.
    No consulta la API de Watchmode.
    """

    if not isinstance(error, WatchmodeBackupError):
        raise TypeError(
            "Se requiere un WatchmodeBackupError"
        )

    rutas = error.rutas_historicas

    storage = get_supabase_client().storage.from_("Bronze")

    resultado = {
        "tmdb_id": error.tmdb_id,
        "region": error.region,
        "version_id": error.version_id,
        "json_existe": False,
        "metadata_existe": False,
        "integridad_verificada": False,
        "estado": "unknown",
        "errores": [],
    }

    
    contenidos = {}
    errores_no_confirmados = False

    for tipo in ("json", "metadata"):
        try:
            contenidos[tipo] = storage.download(
                rutas[tipo]
            )

            resultado[f"{tipo}_existe"] = True

        except Exception as exc:

            # Detectar únicamente códigos que indican
            # que el objeto solicitado no existe.
            status = getattr(
                exc, "status_code", None
            )

            if status is None:
                status = getattr(
                    exc, "status", None
                )

            if str(status) in ("404", "400"):
                # El estado 400 puede tener otras causas:
                # solo es ausencia si el código del error
                # indica explícitamente objeto inexistente.
                codigo = getattr(exc, "code", None)
                codigo = str(codigo).lower()

                if (
                    str(status) == "404"
                    or codigo in (
                        "not_found",
                        "nosuchkey",
                        "404",
                    )
                ):
                    resultado["errores"].append({
                        "objeto": tipo,
                        "tipo": "not_found",
                        "error": str(exc),
                    })
                    continue

            errores_no_confirmados = True

            resultado["errores"].append({
                "objeto": tipo,
                "tipo": "unknown_error",
                "error": str(exc),
            })

    if errores_no_confirmados:
        resultado["estado"] = "error"
        return resultado


    if (
        resultado["json_existe"]
        and resultado["metadata_existe"]
    ):
        try:
            datos = json.loads(
                contenidos["metadata"]
            )

            hash_json = hashlib.sha256(
                contenidos["json"]
            ).hexdigest()

            if (
                isinstance(datos, dict)
                and datos.get("tmdb_id") == error.tmdb_id
                and datos.get("region") == error.region
                and datos.get("region_verified") is True
                and datos.get("sha256") == hash_json
            ):
                resultado["integridad_verificada"] = True
                resultado["estado"] = "verified"
            else:
                resultado["estado"] = "conflict"

        except (ValueError, TypeError):
            resultado["estado"] = "conflict"

    elif (
        resultado["json_existe"]
        or resultado["metadata_existe"]
    ):
        resultado["estado"] = "incomplete"

    else:
        resultado["estado"] = "missing"

    return resultado


def diagnose_watchmode_sync(
    region: str = "PE",
) -> dict:
    """
    Diagnostica el estado de sincronización
    sin modificar Supabase.

    No consulta la API de Watchmode.
    """

    plan = plan_watchmode_sync(region=region)

    return {
        "estado": "success",
        "region": plan["region"],
        "pendientes": len(plan["pendientes"]),
        "incompletos": len(plan["incompletos"]),
        "existentes": len(plan["existentes"]),
        "invalidos": len(plan["invalidos"]),
        "modo": "dry_run",
    }



def coordinate_watchmode_sync(
    region: str = "PE",
) -> dict:
    """
    Coordina la clasificación de paquetes Watchmode.

    Modo diagnóstico:
    - No sube ni elimina archivos.
    - No consulta la API de Watchmode.
    - Consulta el inventario de Supabase.
    """

    plan = plan_watchmode_sync(region=region)

    tareas = []

    # 1. Paquetes nuevos
    for item in plan["pendientes"]:
        tareas.append({
            "archivo": item["archivo"],
            "accion": "upload_new",
            "estado": "pending",
        })

    # 2. Paquetes incompletos
    for item in plan["incompletos"]:
        tareas.append({
            "archivo": item["archivo"],
            "accion": "inspect_incomplete",
            "estado": "pending",
        })

    # 3. Paquetes completos pendientes de auditoría
    for item in plan["existentes"]:
        tareas.append({
            "archivo": item["archivo"],
            "accion": "audit_complete",
            "estado": "pending",
        })

    # 4. Paquetes bloqueados
    for item in plan["invalidos"]:
        tareas.append({
            "archivo": item["archivo"],
            "accion": "manual_review",
            "estado": "blocked",
            "motivo": item["error"],
        })

    # 5. Inspeccionar paquetes incompletos
    if plan["incompletos"]:

        inspeccion = inspect_incomplete_watchmode_bundles(
            region=region,
            plan=plan,
        )

    else:

        inspeccion = {
            "region": plan["region"],
            "recuperables": [],
            "conflictos": [],
            "errores": [],
        }

    resumen_inspeccion = {
        "recuperables": len(inspeccion["recuperables"]),
        "conflictos": len(inspeccion["conflictos"]),
        "errores": len(inspeccion["errores"]),
    }
    
    
    # Clasificar los paquetes incompletos
    recuperables = {
        item["archivo"]
        for item in inspeccion["recuperables"]
    }

    conflictos = {
        item["archivo"]
        for item in inspeccion["conflictos"]
    }

    errores = {
        item["archivo"]
        for item in inspeccion["errores"]
    }

    for tarea in tareas:

        if tarea["accion"] != "inspect_incomplete":
            continue

        nombre = tarea["archivo"]

        if nombre in conflictos:
            tarea["accion"] = "manual_review"
            tarea["estado"] = "conflict"

        elif nombre in errores:
            tarea["accion"] = "retry_inspection"
            tarea["estado"] = "error"

        elif nombre in recuperables:
            tarea["accion"] = "recover_missing"
            tarea["estado"] = "recoverable"

        else:
            tarea["accion"] = "manual_review"
            tarea["estado"] = "unknown"

    
    # 6. Auditar los paquetes completos
    if plan["existentes"]:

        auditoria = audit_complete_watchmode_bundles(
            region=region,
            plan=plan,
        )

    else:

        auditoria = {
            "region": plan["region"],
            "verificados": [],
            "conflictos": [],
            "errores": [],
        }

    resumen_auditoria = {
        "verificados": len(auditoria["verificados"]),
        "conflictos": len(auditoria["conflictos"]),
        "errores": len(auditoria["errores"]),
    }

    verificados = {
        item["archivo"]
        for item in auditoria["verificados"]
    }

    conflictos_completos = {
        item["archivo"]
        for item in auditoria["conflictos"]
    }

    errores_completos = {
        item["archivo"]
        for item in auditoria["errores"]
    }

    # 8. Clasificar individualmente
    for tarea in tareas:

        if tarea["accion"] != "audit_complete":
            continue

        nombre = tarea["archivo"]

        if nombre in conflictos_completos:
            tarea["accion"] = "manual_review"
            tarea["estado"] = "conflict"

        elif nombre in errores_completos:
            tarea["accion"] = "retry_audit"
            tarea["estado"] = "error"

        elif nombre in verificados:
            tarea["accion"] = "none"
            tarea["estado"] = "verified"

        else:
            tarea["accion"] = "manual_review"
            tarea["estado"] = "unknown"

    
    # 8. Comparar versiones de paquetes verificados
    if auditoria["verificados"]:

        comparacion = compare_watchmode_versions(
            region=region,
            plan=plan,
            auditoria=auditoria,
        )

    else:
        comparacion = {
            "region": plan["region"],
            "iguales": [],
            "diferentes": [],
            "errores": [],
        }

    iguales = set(comparacion["iguales"])

    diferentes = {
        item["archivo"]: item
        for item in comparacion["diferentes"]
    }

    errores_comparacion = {
        item["archivo"]
        for item in comparacion["errores"]
    }

    for tarea in tareas:

        # Solo comparar paquetes que aprobaron
        # previamente la auditoría remota
        if tarea["estado"] != "verified":
            continue

        nombre = tarea["archivo"]

        if nombre in errores_comparacion:
            tarea["accion"] = "manual_review"
            tarea["estado"] = "conflict"

        elif nombre in iguales:
            tarea["accion"] = "none"
            tarea["estado"] = "up_to_date"


        elif nombre in diferentes:

            diferencia = diferentes[nombre]

            # Compatibilidad con resultados anteriores
            orden = diferencia.get("orden_temporal")

            if orden is None:
                orden = (
                    "local_newer"
                    if diferencia.get("local_mas_reciente")
                    else "remote_newer"
                )


            if orden == "local_newer":
                tarea["accion"] = "prepare_update"
                tarea["estado"] = "update_candidate"

                # Conservar la identidad de la versión
                # remota evaluada durante la comparación.
                tarea["expected_remote_sha256"] = (
                    diferencia.get("hash_remoto")
                )


            elif orden == "remote_newer":
                tarea["accion"] = "none"
                tarea["estado"] = "remote_newer"

            else:
                tarea["accion"] = "manual_review"
                tarea["estado"] = "conflict"
                tarea["motivo"] = orden


    resumen_comparacion = {
        "iguales": len(comparacion["iguales"]),
        "diferentes": len(comparacion["diferentes"]),
        "errores": len(comparacion["errores"]),
    }

    # 9. Resultado consolidado
    return {
        "estado": "success",
        "modo": "dry_run",
        "region": plan["region"],

        "resumen": {
            "nuevos": len(plan["pendientes"]),
            "incompletos": len(plan["incompletos"]),
            "completos_sin_auditar": len(plan["existentes"]),
            "bloqueados": len(plan["invalidos"]),
        },

        "inspeccion_incompletos": resumen_inspeccion,

        "auditoria_completos": resumen_auditoria,
        "comparacion_versiones": resumen_comparacion,
        "tareas": tareas,
    }


def plan_watchmode_updates(
    region: str = "PE",
    coordination: dict | None = None,
) -> dict:
    """
    Determina qué paquetes pueden avanzar
    hacia el respaldo histórico.

    No realiza subidas, copias ni eliminaciones.
    No consulta la API de Watchmode.
    """

    region = region.upper()

    if coordination is None:
        coordination = coordinate_watchmode_sync(
            region=region
        )

    if coordination.get("region") != region:
        raise ValueError(
            "La región del coordinador no coincide"
        )

    autorizados = []
    bloqueados = []
    sin_cambios = []

    for tarea in coordination["tareas"]:

        archivo = tarea["archivo"]
        estado = tarea["estado"]


        if estado == "update_candidate":

            hash_remoto = tarea.get(
                "expected_remote_sha256"
            )

            # Un candidato sin hash remoto válido
            # no puede avanzar al respaldo.
            if (
                not isinstance(hash_remoto, str)
                or re.fullmatch(
                    r"[0-9a-fA-F]{64}",
                    hash_remoto,
                ) is None
            ):
                bloqueados.append({
                    "archivo": archivo,
                    "motivo": "missing_remote_sha256",
                })
                continue

            autorizados.append({
                "archivo": archivo,
                "accion": "backup_required",
                "estado": "pending_backup",
                "expected_remote_sha256": (
                    hash_remoto.lower()
                ),
            })


        elif estado in ("up_to_date", "remote_newer"):
            sin_cambios.append({
                "archivo": archivo,
                "motivo": estado,
            })

        else:
            bloqueados.append({
                "archivo": archivo,
                "motivo": estado,
            })

    return {
        "estado": "success",
        "modo": "dry_run",
        "region": region,
        "autorizados": autorizados,
        "bloqueados": bloqueados,
        "sin_cambios": sin_cambios,
    }


def execute_watchmode_backup_plan(
    update_plan: dict,
    region: str = "PE",
    dry_run: bool = True,
) -> dict:
    """
    Ejecuta respaldos históricos de candidatos autorizados.

    Por defecto funciona en modo simulación.
    No publica nuevas versiones.
    """

    region = region.upper()

    if update_plan.get("region") != region:
        raise ValueError(
            "La región del plan no coincide"
        )

    resultados = []
    exitosos = 0
    errores = 0

    for candidato in update_plan.get("autorizados", []):

        archivo = candidato["archivo"]
        hash_esperado = candidato.get(
            "expected_remote_sha256"
        )

        
        # Validar autorización para el respaldo.
        if (
            candidato.get("estado") != "pending_backup"
            or candidato.get("accion") != "backup_required"
        ):
            resultados.append({
                "archivo": archivo,
                "estado": "blocked",
                "motivo": "invalid_backup_authorization",
            })
            errores += 1
            continue

        
        # Validar identidad de la versión remota.
        if (
            not isinstance(hash_esperado, str)
            or re.fullmatch(
                r"[0-9a-fA-F]{64}",
                hash_esperado,
            ) is None
        ):
            resultados.append({
                "archivo": archivo,
                "estado": "blocked",
                "motivo": "invalid_remote_sha256",
            })
            errores += 1
            continue
        
        
        # Validar el nombre también en modo simulación.
        coincidencia = re.fullmatch(
            r"movie_([1-9]\d*)_sources\.json",
            archivo,
        ) if isinstance(archivo, str) else None

        if coincidencia is None:
            resultados.append({
                "archivo": archivo,
                "estado": "blocked",
                "motivo": "invalid_filename",
            })
            errores += 1
            continue

        tmdb_id = int(coincidencia.group(1))

        
        # El modo predeterminado no realiza escrituras.
        if dry_run:
            resultados.append({
                "archivo": archivo,
                "estado": "pending_backup",
                "expected_remote_sha256": hash_esperado,
            })
            continue

        try:
            
            respaldo = backup_watchmode_remote_version(
                tmdb_id=tmdb_id,
                region=region,
                expected_sha256=hash_esperado,
            )

            # Aceptar únicamente respaldos verificados.
            if (
                not isinstance(respaldo, dict)
                or respaldo.get("estado") != "verified"
            ):
                raise RuntimeError(
                    "El respaldo histórico no fue verificado"
                )

            resultados.append({
                "archivo": archivo,
                "estado": "backup_completed",
                "respaldo": respaldo,
            })

            exitosos += 1


        except Exception as exc:
            resultados.append({
                "archivo": archivo,
                "estado": "backup_failed",
                "motivo": str(exc),
            })
            errores += 1

    return {
        "region": region,
        "modo": "dry_run" if dry_run else "execute",
        "estado": "partial" if errores else "success",
        "exitosos": exitosos,
        "errores": errores,
        "resultados": resultados,
    }


def verify_watchmode_remote_unchanged(
    tmdb_id: int,
    expected_sha256: str,
    region: str = "PE",
) -> dict:
    """
    Comprueba que la versión remota vigente
    no haya cambiado desde su planificación.

    No modifica archivos ni consulta Watchmode.
    """

    if not isinstance(expected_sha256, str):
        raise TypeError("SHA-256 debe ser texto")

    if not re.fullmatch(
        r"[0-9a-fA-F]{64}",
        expected_sha256,
    ):
        raise ValueError("SHA-256 inválido")

    ruta_json = build_watchmode_movie_path(
        tmdb_id=tmdb_id,
        region=region,
    )

    ruta_metadata = (
        ruta_json.removesuffix(".json")
        + "_metadata.json"
    )

    storage = get_supabase_client().storage.from_("Bronze")

    contenido_json = storage.download(ruta_json)
    contenido_metadata = storage.download(ruta_metadata)

    datos_metadata = json.loads(contenido_metadata)

    hash_actual = hashlib.sha256(
        contenido_json
    ).hexdigest()

    if not isinstance(datos_metadata, dict):
        return {
            "unchanged": False,
            "reason": "invalid_metadata",
        }

    if (
        datos_metadata.get("tmdb_id") != tmdb_id
        or datos_metadata.get("region") != region.upper()
        or datos_metadata.get("region_verified") is not True
        or datos_metadata.get("sha256") != hash_actual
    ):
        return {
            "unchanged": False,
            "reason": "metadata_integrity_error",
        }

    if hash_actual != expected_sha256.lower():
        return {
            "unchanged": False,
            "reason": "remote_changed",
        }

    return {
        "unchanged": True,
        "reason": None,
        "sha256": hash_actual,
    }


if __name__ == "__main__":
    run_watchmode_ingestion()