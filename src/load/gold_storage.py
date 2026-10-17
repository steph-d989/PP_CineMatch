from datetime import date
from pathlib import Path

from src.utils.paths import GOLD_DIR
from src.utils.supabase_client import (
    upload_file_to_storage,
)

### FUNCION QUE SUBE LOS AARCHIVOS A GOLD CONSERVANDO LA ESTRUCTURA

def upload_gold_directory(
    local_dir: Path,
    source_root: Path,
) -> dict:

    ingestion_date = date.today().isoformat()

    resultados = {
        "subidos": 0,
        "existentes": 0,
        "errores": [],
    }

    archivos = [
        archivo
        for archivo in local_dir.rglob("*")
        if archivo.is_file()
    ]

    for archivo in archivos:

        relative_path = (
            archivo
            .relative_to(source_root)
            .as_posix()
        )

        remote_path = (
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
                f"{relative_path} -> OK"
            )

        except Exception as error:

            mensaje = str(error)
            mensaje_lower = mensaje.lower()

            if (
                "already exists" in mensaje_lower
                or "duplicate" in mensaje_lower
                or "resource already exists"
                in mensaje_lower
            ):

                resultados["existentes"] += 1

                print(
                    f"{relative_path} -> YA EXISTE"
                )

            else:

                resultados["errores"].append(
                    {
                        "archivo": relative_path,
                        "error": mensaje,
                    }
                )

                print(
                    f"{relative_path} -> ERROR"
                )

                print(
                    "   Detalle:",
                    mensaje
                )

    return resultados

### FUNCION QUE PUBLICA LA CAPA GOLD LOCAL A UPABASE STORAGE

def upload_gold_to_supabase() -> dict:

    print("=" * 60)
    print("SUBIDA GOLD A SUPABASE")
    print("=" * 60)

    if not GOLD_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta Gold: {GOLD_DIR}"
        )

    resultado = upload_gold_directory(
        local_dir=GOLD_DIR,
        source_root=GOLD_DIR,
    )

    print("\n" + "=" * 60)
    print("RESUMEN GOLD")
    print("=" * 60)

    print(
        "Subidos:",
        resultado["subidos"]
    )

    print(
        "Ya existentes:",
        resultado["existentes"]
    )

    print(
        "Errores:",
        len(resultado["errores"])
    )

    print("=" * 60)

    return resultado


if __name__ == "__main__":
    upload_gold_to_supabase()