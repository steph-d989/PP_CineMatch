import json
import zipfile
from datetime import datetime
from pathlib import Path

import requests

from src.utils.paths import BRONZE_MOVIELENS_DIR


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


def download_movielens(
    url: str = MOVIELENS_URL,
    force_download: bool = False
) -> Path:
    """
    Descarga MovieLens 32M en la capa Bronze.

    Parameters
    ----------
    url : str
        URL oficial del dataset.
    force_download : bool
        Si True, vuelve a descargar aunque el ZIP ya exista.

    Returns
    -------
    Path
        Ruta del archivo ZIP descargado.
    """

    BRONZE_MOVIELENS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ruta_zip = BRONZE_MOVIELENS_DIR / "ml-32m.zip"

    if ruta_zip.exists() and not force_download:
        print("MovieLens ya está descargado.")
        print("Ruta:", ruta_zip)
        return ruta_zip

    print("Descargando MovieLens 32M...")

    with requests.get(
        url,
        stream=True,
        timeout=60
    ) as respuesta:

        respuesta.raise_for_status()

        with open(ruta_zip, "wb") as archivo:

            for bloque in respuesta.iter_content(
                chunk_size=1024 * 1024
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
    """
    Extrae MovieLens dentro de Bronze.

    Returns
    -------
    Path
        Carpeta donde quedaron los CSV.
    """

    destino = BRONZE_MOVIELENS_DIR
    carpeta_dataset = destino / "ml-32m"

    if carpeta_dataset.exists() and not force_extract:
        print("MovieLens ya está extraído.")
        print("Ruta:", carpeta_dataset)
        return carpeta_dataset

    print("Extrayendo MovieLens...")

    with zipfile.ZipFile(ruta_zip, "r") as archivo_zip:
        archivo_zip.extractall(destino)

    print("MovieLens extraído correctamente.")

    return carpeta_dataset


### FUNCIÓN DE VALIDACIÓN 

def validate_movielens(
    carpeta_dataset: Path
) -> dict:
    """
    Valida que existan los archivos principales
    de MovieLens.
    """

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
    """
    Guarda metadata de la ingesta Bronze.
    """

    archivos_ok = all(
        info["existe"]
        for info in validacion.values()
    )

    metadata = {
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
    """
    Ejecuta la ingesta completa de MovieLens
    hacia la capa Bronze.
    """

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