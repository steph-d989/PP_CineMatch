from datetime import date
from pathlib import Path

from src.utils.paths import (
    GOLD_ANALYTICS_DIR,
    GOLD_ML_DIR,
)

from src.utils.supabase_client import (
    upload_file_to_storage,
)

### FUNCION QUE SUBE LOS ARCHIVOS DE UNA CARPETA GOLD AL BUCKET DE SUPABASE

def upload_gold_directory(
    local_dir: Path,
    source: str,
) -> dict:

    ingestion_date = date.today().isoformat()

    resultados = {
        "source": source,
        "subidos": 0,
        "existentes": 0,
        "errores": [],
    }

    if not local_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta Gold: {local_dir}"
        )

    archivos = [
        archivo
        for archivo in local_dir.rglob("*")
        if archivo.is_file()
    ]

    if not archivos:
        print(
            f"{source} -> no hay archivos para subir."
        )

        return resultados

    for archivo in archivos:

        relative_path = (
            archivo
            .relative_to(local_dir)
            .as_posix()
        )

        remote_path = (
            f"{source}/"
            f"ingestion_date={ingestion_date}/"
            f"{relative_path}"
        )

        try:
            upload_file_to_storage(
                bucket="Gold",
                local_path=archivo,
                remote_path=remote_path,
                upsert=False,
            )

            resultados["subidos"] += 1

            print(
                f"{source} | {relative_path} -> OK"
            )

        except Exception as error:

            mensaje = str(error)
            mensaje_lower = mensaje.lower()

            if (
                "already exists" in mensaje_lower
                or "duplicate" in mensaje_lower
                or "resource already exists" in mensaje_lower
            ):

                resultados["existentes"] += 1

                print(
                    f"{source} | "
                    f"{relative_path} -> YA EXISTE"
                )

            else:

                resultados["errores"].append({
                    "archivo": relative_path,
                    "error": mensaje,
                })

                print(
                    f"{source} | "
                    f"{relative_path} -> ERROR"
                )

                print(
                    "   Detalle:",
                    mensaje
                )

    return resultados


### FUNCION QUE PUBLICA GOLD ANALITICO EN SUPABASE STORAGE

def upload_gold_to_supabase() -> dict:

    print("=" * 60)
    print("SUBIDA GOLD A SUPABASE")
    print("=" * 60)

    fuentes = {
        "analytics": GOLD_ANALYTICS_DIR,
        "ml": GOLD_ML_DIR,
    }

    resultados = {}

    for source, local_dir in fuentes.items():

        print(
            f"\nProcesando Gold/{source}"
        )

        try:
            resultados[source] = (
                upload_gold_directory(
                    local_dir=local_dir,
                    source=source,
                )
            )

        except Exception as error:

            resultados[source] = {
                "source": source,
                "subidos": 0,
                "existentes": 0,
                "errores": [
                    {
                        "archivo": None,
                        "error": str(error),
                    }
                ],
            }

            print(
                f"{source} -> ERROR GENERAL"
            )

            print(
                "   Detalle:",
                error
            )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    total_subidos = 0
    total_existentes = 0
    total_errores = 0

    for source, resultado in resultados.items():

        subidos = resultado["subidos"]
        existentes = resultado["existentes"]
        errores = len(
            resultado["errores"]
        )

        total_subidos += subidos
        total_existentes += existentes
        total_errores += errores

        print(
            source,
            "| subidos:",
            subidos,
            "| existentes:",
            existentes,
            "| errores:",
            errores,
        )

    print("-" * 60)

    print(
        "TOTAL",
        "| subidos:",
        total_subidos,
        "| existentes:",
        total_existentes,
        "| errores:",
        total_errores,
    )

    print("=" * 60)

    return resultados


if __name__ == "__main__":
    upload_gold_to_supabase()