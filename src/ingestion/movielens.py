import json
import zipfile
from datetime import datetime
from pathlib import Path

import requests

from src.utils.paths import BRONZE_MOVIELENS_DIR


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"

### FUNCION PARA DESCARGAR ARCHIVO ZIP DE MOVIELENS

def download_movielens(
    url: str = MOVIELENS_URL, #Desde donde descarga
    force_download: bool = False #Indica si tenemos carpeta con el mismo nombre no volver a descargar
) -> Path:


    BRONZE_MOVIELENS_DIR.mkdir( # Crea la carpeta bronze si no la tenemos
        parents=True, # Permite crear carpetas padre si hace falta
        exist_ok=True # Evita errores si la capeta ya existe
    )

    ruta_zip = BRONZE_MOVIELENS_DIR / "ml-32m.zip" # Donde guardamos el ZIP

    if ruta_zip.exists() and not force_download: # Si el ZIP existe no descargar
        print("MovieLens ya está descargado.")
        print("Ruta:", ruta_zip)
        return ruta_zip

    print("Descargando MovieLens 32M...")

    with requests.get( # Consulta URL
        url,
        stream=True, # No carga todo el archivo en RAM de una sola vez
        timeout=60 # Evita que una conexion congelada espere infinitamente
    ) as respuesta:

        respuesta.raise_for_status() # Si devuelve un error 404 500 503 ....

        with open(ruta_zip, "wb") as archivo:

            for bloque in respuesta.iter_content(
                chunk_size=1024 * 1024  # Descarga el archivo por bloques de 1MB
            ):
                if bloque:
                    archivo.write(bloque)

    print("Descarga completada.")

    return ruta_zip


### FUNCION PARA DESCOMPRIMIR

def extract_movielens(
    ruta_zip: Path,
    force_extract: bool = False
) -> Path:

    destino = BRONZE_MOVIELENS_DIR
    carpeta_dataset = destino / "ml-32m"

    # Si los archivos ya fueron extraidos no lo vuleve a hacer 
    if carpeta_dataset.exists() and not force_extract:
        print("MovieLens ya está extraído.")
        print("Ruta:", carpeta_dataset)
        return carpeta_dataset

    print("Extrayendo MovieLens...")

    with zipfile.ZipFile(ruta_zip, "r") as archivo_zip:
        archivo_zip.extractall(destino)

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
        "links.csv"
    ]

    resultado = {}

    for nombre in archivos_esperados:

        ruta = carpeta_dataset / nombre

        resultado[nombre] = {
            "existe": ruta.exists(),
            "tamano_mb": (
                round(
                    ruta.stat().st_size / (1024 ** 2),
                    2
                )
                if ruta.exists()
                else None
            )
        }

    return resultado


### FUNCION PARA METADATA

def save_metadata(
    validacion: dict
) -> Path:

    archivos_ok = all( #Comprueba que los archivos existan
        info["existe"]
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

    save_metadata(validacion)

    print("\nValidación:")

    for archivo, info in validacion.items():

        print(
            archivo,
            "->",
            "OK" if info["existe"] else "FALTA",
            "|",
            info["tamano_mb"],
            "MB"
        )

    print("=" * 50)
    print("FIN INGESTA MOVIELENS")
    print("=" * 50)

    return validacion

if __name__ == "__main__":
    run_movielens_ingestion()