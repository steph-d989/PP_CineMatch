import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

from src.utils.paths import PROJECT_ROOT


### FUNCION QUE CREA Y DEVUELVE UN CLIENTE AUTENTICADO

def get_supabase_client() -> Client:

    load_dotenv(PROJECT_ROOT / ".env")

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv(
        "SUPABASE_SERVICE_ROLE_KEY"
    )

    if not supabase_url:
        raise ValueError(
            "No se encontró SUPABASE_URL "
            "en el archivo .env"
        )

    if not supabase_key:
        raise ValueError(
            "No se encontró "
            "SUPABASE_SERVICE_ROLE_KEY "
            "en el archivo .env"
        )

    return create_client(
        supabase_url,
        supabase_key,
    )


### FUNCION QUE CONTRUYE UNA RUTA ESTANDAR PARA STORAGE

def build_storage_path(
    source: str,
    ingestion_date: str,
    filename: str,
) -> str:

    return (
        f"{source}/"
        f"ingestion_date={ingestion_date}/"
        f"{filename}"
    )


### FUNCION QUE SUBE UN ARCHIVO LOCAL A SUPABASE

def upload_file_to_storage(
    bucket: str,
    local_path: Path,
    remote_path: str,
    upsert: bool = False,
):

    if not local_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: "
            f"{local_path}"
        )

    if not local_path.is_file():
        raise ValueError(
            f"No es un archivo: {local_path}"
        )

    supabase = get_supabase_client()

    content_type, _ = mimetypes.guess_type(
        local_path.name
    )

    if content_type is None:
        content_type = (
            "application/octet-stream"
        )

    with open(local_path, "rb") as file:

        response = (
            supabase.storage
            .from_(bucket)
            .upload(
                path=remote_path,
                file=file,
                file_options={
                    "content-type": content_type,
                    "upsert": (
                        "true"
                        if upsert
                        else "false"
                    ),
                },
            )
        )

    return response


### FUNCION QUE DESCARGA UN ARCHIVO DESDE SUPABASE 

def download_file_from_storage(
    bucket: str,
    remote_path: str,
    local_path: Path,
) -> Path:

    supabase = get_supabase_client()

    data = (
        supabase.storage
        .from_(bucket)
        .download(remote_path)
    )

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(local_path, "wb") as file:
        file.write(data)

    return local_path