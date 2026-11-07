import json
import zipfile
from datetime import datetime
from pathlib import Path
from datetime import date

import requests

from src.validation.data_quality import (
    validate_file_exists,
    validate_csv_file,
)

from src.utils.supabase_client import (
    build_storage_path,
    upload_file_to_storage,
)

from src.utils.paths import BRONZE_MOVIELENS_DIR

from src.utils.config import MOVIELENS_URL

### FUNCION PARA DESCARGAR ARCHIVO ZIP DE MOVIELENS


def download_movielens(
    url: str = MOVIELENS_URL,
    force_download: bool = False
) -> Path:

    BRONZE_MOVIELENS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_zip = BRONZE_MOVIELENS_DIR / "ml-32m.zip"
    ruta_temporal = BRONZE_MOVIELENS_DIR / "ml-32m.zip.part"

    # Verificar el ZIP existente
    if ruta_zip.exists() and not force_download:

        if zipfile.is_zipfile(ruta_zip):
            print("MovieLens ya está descargado.")
            return ruta_zip

        print("ZIP existente inválido. Se descargará nuevamente.")

    print("Descargando MovieLens 32M...")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=60
        ) as respuesta:

            respuesta.raise_for_status()

            with open(ruta_temporal, "wb") as archivo:
                for bloque in respuesta.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if bloque:
                        archivo.write(bloque)

        # Verificar integridad del ZIP
        with zipfile.ZipFile(ruta_temporal, "r") as z:
            archivo_corrupto = z.testzip()

            if archivo_corrupto is not None:
                raise ValueError(
                    f"Archivo corrupto: {archivo_corrupto}"
                )

        # Reemplazar únicamente después de validar
        ruta_temporal.replace(ruta_zip)

        print("Descarga completada y verificada.")

        return ruta_zip

    finally:
        # Eliminar descarga parcial
        if ruta_temporal.exists():
            ruta_temporal.unlink()


### FUNCION PARA DESCOMPRIMIR


def extract_movielens(
    ruta_zip: Path,
    force_extract: bool = False
) -> Path:

    destino = BRONZE_MOVIELENS_DIR
    carpeta_dataset = destino / "ml-32m"

    archivos_esperados = [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
        "genome-scores.csv",
        "genome-tags.csv",
    ]

    archivos_completos = all(
        (carpeta_dataset / nombre).is_file()
        for nombre in archivos_esperados
    )

    if archivos_completos and not force_extract:
        print("MovieLens ya está extraído.")
        return carpeta_dataset

    print("Verificando archivo ZIP...")

    if not zipfile.is_zipfile(ruta_zip):
        raise ValueError(
            "El archivo MovieLens no es un ZIP válido."
        )

    print("Extrayendo MovieLens...")

    with zipfile.ZipFile(ruta_zip, "r") as archivo_zip:
        archivo_zip.extractall(destino)

    archivos_faltantes = [
        nombre
        for nombre in archivos_esperados
        if not (carpeta_dataset / nombre).is_file()
    ]

    if archivos_faltantes:
        raise FileNotFoundError(
            f"Archivos faltantes: {archivos_faltantes}"
        )

    print("MovieLens extraído correctamente.")

    return carpeta_dataset


### FUNCIÓN DE VALIDACIÓN, comprueba uqe se generaran los archivos que esperamos

def validate_movielens(
    carpeta_dataset: Path
) -> dict:

    archivos_esperados = [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
        "genome-scores.csv",
        "genome-tags.csv",
    ]

    resultado = {}

    for nombre in archivos_esperados:
        ruta = carpeta_dataset / nombre

        existe = validate_file_exists(ruta)

        if existe:
            validacion_csv = validate_csv_file(ruta)

            tamano_mb = round(
                ruta.stat().st_size / (1024 ** 2),
                2
            )
        else:
            validacion_csv = {
                "valid": False,
                "error": "file_not_found"
            }

            tamano_mb = None

        resultado[nombre] = {
            "existe": existe,
            "csv_valido": validacion_csv["valid"],
            "filas": validacion_csv.get("rows"),
            "columnas": validacion_csv.get("columns"),
            "tamano_mb": tamano_mb,
            "error": validacion_csv.get("error"),
        }

    return resultado


### FUNCION QUE SUBE DE BRONZE A SUPABASE STORAGE


def upload_movielens_to_supabase(
    ruta_metadata: Path,
) -> dict:

    ingestion_date = date.today().isoformat()

    # Construimos la ruta en Supabase
    remote_metadata = build_storage_path(
        source="movielens",
        ingestion_date=ingestion_date,
        filename=ruta_metadata.name,
    )

    # Los metadatos se pueden actualizar
    respuesta = upload_file_to_storage(
        bucket="Bronze",
        local_path=ruta_metadata,
        remote_path=remote_metadata,
        upsert=True,
    )

    return {
        "metadata": respuesta,
        "remote_path": remote_metadata,
    }


### FUNCION PARA METADATA

def save_metadata(
    validacion: dict
) -> Path:

    archivos_ok = all(
        info["existe"] and info["csv_valido"]
        for info in validacion.values()
    )

    metadata = { # Genera un diccionario con la informacion de la metadata
        "fuente": "MovieLens",
        "dataset": "ml-32m",
        "capa": "bronze",
        "fecha_extraccion": datetime.now().isoformat(),
        "archivo_origen": "ml-32m.zip",
        "estado": (
            "extraccion_completada"
            if archivos_ok
            else "extraccion_incompleta"
        ),
        "archivos": validacion
    }

    ruta_metadata = (
        BRONZE_MOVIELENS_DIR
        / "_metadata.json"
    )

    with open(
        ruta_metadata,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            metadata,
            archivo,
            ensure_ascii=False,
            indent=2
        )

    print("Metadata registrada.")

    return ruta_metadata


### FUNCIÓN QUE EJECUTA TODO

def run_movielens_ingestion(
    force_download: bool = False,
    force_extract: bool = False
) -> dict:

    print("=" * 50)
    print("INICIO INGESTA MOVIELENS")
    print("=" * 50)

    ruta_zip = download_movielens(
        force_download=force_download
    )

    carpeta_dataset = extract_movielens(
        ruta_zip,
        force_extract=force_extract
    )

    validacion = validate_movielens(
        carpeta_dataset
    )

    archivos_invalidos = [
        nombre
        for nombre, info in validacion.items()
        if not info["existe"] or not info["csv_valido"]
    ]

    if archivos_invalidos:
        raise RuntimeError(
            "Validación MovieLens fallida. "
            f"Archivos inválidos: {archivos_invalidos}"
        )

    print("Validación MovieLens completada correctamente.")

    ruta_metadata = save_metadata(validacion)


    print("\nSubiendo MovieLens a Supabase Storage...")

    upload_result = upload_movielens_to_supabase(
        # ruta_zip,
        ruta_metadata,
    )

    print("MovieLens subido correctamente a Supabase.")
        
    print("\nValidación:")

    for archivo, info in validacion.items():

        estado = (
            "OK"
            if info["existe"] and info["csv_valido"]
            else "ERROR"
        )

        print(
            archivo,
            "->",
            estado,
            "|",
            info["filas"],
            "filas |",
            info["columnas"],
            "columnas |",
            info["tamano_mb"],
            "MB"
        )

    print("=" * 50)
    print("FIN INGESTA MOVIELENS")
    print("=" * 50)
    
    return {
        "estado": "success",
        "fuente": "MovieLens",
        "validacion": validacion,
        "supabase": upload_result,
    }


if __name__ == "__main__":
    run_movielens_ingestion()