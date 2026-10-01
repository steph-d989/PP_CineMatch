from datetime import date
from pathlib import Path

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_TMDB_DIR,
    SILVER_WATCHMODE_DIR,
    SILVER_INTEGRATED_DIR,
)

from src.utils.supabase_client import (
    build_storage_path,
    upload_file_to_storage,
)

### FUNCION QUE SUBE LOS ARCHIVOS DE UNA CARPETA SILVER AL BUCKET SILVER

def upload_silver_directory(
    local_dir: Path,
    source: str,
) -> dict:

    ingestion_date = date.today().isoformat()

    resultados = {
        "source": source,
        "subidos": 0,
        "errores": [],
    }

    archivos = [
        archivo
        for archivo in local_dir.iterdir()
        if archivo.is_file()
    ]

    for archivo in archivos:

        remote_path = build_storage_path(
            source=source,
            ingestion_date=ingestion_date,
            filename=archivo.name,
        )

        try:
            upload_file_to_storage(
                bucket="Silver",
                local_path=archivo,
                remote_path=remote_path,
                upsert=False,
            )

            resultados["subidos"] += 1

            print(
                f"{source} | {archivo.name} -> OK"
            )

        except Exception as error:

            resultados["errores"].append(
                {
                    "archivo": archivo.name,
                    "error": str(error),
                }
            )

            print(
                f"{source} | {archivo.name} -> ERROR"
            )
            
            print(
            "   Detalle:",
            error
        )

    return resultados


### FUNCION QUE PUBLICA TODA LA CAPA SILVER LOCAL EN SUPABASE

def upload_silver_to_supabase() -> dict:

    print("=" * 60)
    print("SUBIDA SILVER A SUPABASE")
    print("=" * 60)

    fuentes = {
        "movielens": SILVER_MOVIELENS_DIR,
        "tmdb": SILVER_TMDB_DIR,
        "watchmode": SILVER_WATCHMODE_DIR,
        "integrated": SILVER_INTEGRATED_DIR,
    }

    resultados = {}

    for source, local_dir in fuentes.items():

        print(
            f"\nProcesando Silver/{source}"
        )

        resultados[source] = (
            upload_silver_directory(
                local_dir=local_dir,
                source=source,
            )
        )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    for source, resultado in resultados.items():

        print(
            source,
            "| subidos:",
            resultado["subidos"],
            "| errores:",
            len(resultado["errores"]),
        )

    print("=" * 60)

    return resultados

if __name__ == "__main__":
    upload_silver_to_supabase()