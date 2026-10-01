from datetime import date

from src.utils.paths import BRONZE_TMDB_DIR
from src.utils.supabase_client import (
    build_storage_path,
    upload_file_to_storage,
)


# Tomamos uno de los JSON TMDb que ya descargamos
archivos_tmdb = list(
    BRONZE_TMDB_DIR.glob("movie_*.json")
)

if not archivos_tmdb:
    raise FileNotFoundError(
        "No hay archivos movie_*.json en Bronze TMDb"
    )

archivo = archivos_tmdb[0]

print("Archivo local:")
print(archivo)


# Construimos la ruta remota
remote_path = build_storage_path(
    source="tmdb",
    ingestion_date=date.today().isoformat(),
    filename=archivo.name,
)

print("\nRuta en Supabase:")
print(remote_path)


# Subimos el archivo
respuesta = upload_file_to_storage(
    bucket="Bronze",
    local_path=archivo,
    remote_path=remote_path,
)

print("\nArchivo subido correctamente.")
print(respuesta)