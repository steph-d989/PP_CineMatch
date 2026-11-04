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

## FUNCION QUE BUSCA EL ARCHIVO PAGINA POR PAGINA

def storage_file_exists(
    bucket: str,
    remote_path: str,
) -> bool:

    supabase = get_supabase_client()

    carpeta, nombre = remote_path.rsplit("/", 1)

    storage = supabase.storage.from_(bucket)

    limite = 100
    offset = 0

    while True:
        objetos = storage.list(
            path=carpeta,
            options={
                "limit": limite,
                "offset": offset,
                "sortBy": {
                    "column": "name",
                    "order": "asc"
                }
            }
        )

        if any(
            objeto.get("name") == nombre
            for objeto in objetos
        ):
            return True

        if len(objetos) < limite:
            return False

        offset += limite


def list_storage_filenames(
    bucket: str,
    folder: str,
    page_size: int = 100,
) -> set[str]:
    """
    Obtiene los nombres de todos los archivos de
    una carpeta de Supabase Storage.

    Consulta las páginas y devuelve un conjunto
    para realizar búsquedas eficientes.
    """

    if page_size < 1:
        raise ValueError(
            "page_size debe ser mayor que cero"
        )

    supabase = get_supabase_client()
    storage = supabase.storage.from_(bucket)

    nombres = set()
    offset = 0

    while True:
        objetos = storage.list(
            path=folder,
            options={
                "limit": page_size,
                "offset": offset,
                "sortBy": {
                    "column": "name",
                    "order": "asc",
                },
            },
        )

        for objeto in objetos:
            nombre = objeto.get("name")

            # Excluir entradas de subdirectorios
            if nombre and objeto.get("id") is not None:
                nombres.add(nombre)

        if len(objetos) < page_size:
            break

        offset += page_size

    return nombres


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


### FUNCION QUE CONSTRUYE LA RUTA ESTABLE DE UNA PELICULA  EN SUPABASE STORAGE

def build_tmdb_movie_path(tmdb_id: int) -> str:

    if not isinstance(tmdb_id, int) or isinstance(tmdb_id, bool):
        raise TypeError("tmdb_id debe ser un entero")

    if tmdb_id <= 0:
        raise ValueError("tmdb_id debe ser positivo")

    return f"tmdb/movies/movie_{tmdb_id}.json"



def list_tmdb_legacy_files(
    bucket: str = "Bronze",
) -> list[dict]:
    """
    Identifica archivos TMDb de las carpetas
    antiguas ingestion_date=YYYY-MM-DD.

    No modifica archivos en Supabase.
    """

    supabase = get_supabase_client()
    storage = supabase.storage.from_(bucket)

    carpetas = storage.list(path="tmdb")

    inventario = []

    for carpeta in carpetas:
        nombre = carpeta.get("name", "")

        if not nombre.startswith("ingestion_date="):
            continue

        ruta_carpeta = f"tmdb/{nombre}"

        archivos = list_storage_filenames(
            bucket=bucket,
            folder=ruta_carpeta,
        )

        for archivo in sorted(archivos):
            if (
                archivo.startswith("movie_")
                and archivo.endswith(".json")
            ):
                inventario.append({
                    "archivo": archivo,
                    "ruta_antigua": (
                        f"{ruta_carpeta}/{archivo}"
                    ),
                    "fecha_carpeta": nombre.replace(
                        "ingestion_date=", ""
                    ),
                })

    return inventario



def plan_tmdb_migration() -> dict:
    """
    Construye un plan de migración.
    No copia, elimina ni modifica archivos.
    """

    inventario = list_tmdb_legacy_files()

    archivos_destino = list_storage_filenames(
        bucket="Bronze",
        folder="tmdb/movies",
    )

    agrupados = {}

    for registro in inventario:
        nombre = registro["archivo"]
        agrupados.setdefault(nombre, []).append(registro)

    pendientes = []
    existentes = []
    duplicados = []

    for nombre, versiones in sorted(agrupados.items()):

        versiones = sorted(
            versiones,
            key=lambda x: x["fecha_carpeta"],
            reverse=True,
        )

        if len(versiones) > 1:
            duplicados.append({
                "archivo": nombre,
                "versiones": len(versiones),
                "rutas": [
                    item["ruta_antigua"]
                    for item in versiones
                ],
            })

        registro = {
            "archivo": nombre,
            "origen": versiones[0]["ruta_antigua"],
            "destino": f"tmdb/movies/{nombre}",
        }

        if nombre in archivos_destino:
            existentes.append(registro)
        else:
            pendientes.append(registro)

    return {
        "peliculas_unicas": len(agrupados),
        "pendientes": pendientes,
        "existentes": existentes,
        "duplicados": duplicados,
    }
