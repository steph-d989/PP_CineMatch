
import argparse
import hashlib
import json
import re
import shutil

from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.watchmode import (
    inventory_legacy_watchmode_files,
    validate_watchmode_local_file,
)

from src.utils.paths import BRONZE_WATCHMODE_DIR


def calcular_hash(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def migrar(execute: bool = False):

    print("=" * 60)
    print("CINEMATCH - MIGRACIÓN LOCAL WATCHMODE")
    print("=" * 60)

    inventario = inventory_legacy_watchmode_files()

    carpeta_destino = (
        BRONZE_WATCHMODE_DIR / "regions" / "PE"
    )

    resumen = {
        "total": inventario["total"],
        "preparados": 0,
        "copiados": 0,
        "existentes": 0,
        "errores": [],
    }

    registros = []

    for item in inventario["validos"]:
        origen = Path(item["ruta"])
        destino = carpeta_destino / origen.name

        try:
            if not re.fullmatch(
                r"movie_\d+_sources\.json",
                origen.name
            ):
                raise ValueError("Nombre inválido")

            validacion = validate_watchmode_local_file(
                origen
            )

            if not validacion["valid"]:
                raise ValueError(validacion["error"])

            hash_original = calcular_hash(origen)

            registro = {
                "archivo": origen.name,
                "origen": str(origen),
                "destino": str(destino),
                "sha256": hash_original,
                "region_assumed": "PE",
                "region_verified": False,
                "sources_count": validacion["sources_count"],
            }

            if destino.exists():

                if calcular_hash(destino) != hash_original:
                    raise RuntimeError(
                        "El destino existe con contenido diferente"
                    )

                resumen["existentes"] += 1
                registros.append(registro)
                print(f"EXISTENTE: {origen.name}")
                continue

            resumen["preparados"] += 1

            if not execute:
                print(f"SIMULACIÓN: {origen.name}")
                registros.append(registro)
                continue

            carpeta_destino.mkdir(
                parents=True,
                exist_ok=True
            )

            # Copiar sin sobrescribir archivos existentes.
            # Si otro proceso crea el destino simultáneamente,
            # esta operación fallará.
            with open(origen, "rb") as fuente:
                with open(destino, "xb") as copia:
                    shutil.copyfileobj(fuente, copia)

            if calcular_hash(destino) != hash_original:
                raise RuntimeError(
                    "La copia no coincide con el original"
                )

            resumen["copiados"] += 1
            registros.append(registro)
            print(f"VERIFICADO: {origen.name}")

        except Exception as error:
            resumen["errores"].append({
                "archivo": origen.name,
                "error": str(error),
            })
            print(f"ERROR: {origen.name} - {error}")

    # Incluir también los archivos antiguos inválidos
    for item in inventario["invalidos"]:
        resumen["errores"].append({
            "archivo": item["archivo"],
            "error": item["error"],
        })

    if execute:
        metadata = {
            "migration_date": datetime.now(
                timezone.utc
            ).isoformat(),
            "region_assumed": "PE",
            "region_verified": False,
            "records": registros,
            "summary": resumen,
        }

        carpeta_destino.mkdir(
            parents=True,
            exist_ok=True
        )

        ruta_metadata = (
            carpeta_destino / "_migration_metadata.json"
        )

        ruta_metadata.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

    print("\nRESUMEN")
    print(json.dumps(
        resumen,
        ensure_ascii=False,
        indent=2
    ))

    print(
        "\nModo:",
        "EJECUCIÓN" if execute else "SIMULACIÓN"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Realizar la migración local",
    )

    args = parser.parse_args()

    migrar(execute=args.execute)
