
import argparse
import hashlib
import json
import re

from src.utils.supabase_client import (
    get_supabase_client,
    plan_tmdb_migration,
    list_storage_filenames,
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main(execute=False):

    print("=" * 60)
    print("CINEMATCH - MIGRACIÓN TMDB")
    print("=" * 60)

    plan = plan_tmdb_migration()

    storage = get_supabase_client().storage.from_("Bronze")

    pendientes = plan["pendientes"]

    # Inventario de destinos existentes
    destinos = list_storage_filenames(
        bucket="Bronze",
        folder="tmdb/movies",
    )

    resumen = {
        "verificados": 0,
        "copiados": 0,
        "omitidos": 0,
        "errores": [],
    }

    for item in pendientes:

        origen = item["origen"]
        destino = item["destino"]
        nombre = item["archivo"]

        try:
            if nombre in destinos:
                print(f"OMITIDO: {nombre} ya existe")
                resumen["omitidos"] += 1
                continue

            match = re.fullmatch(
                r"movie_(\d+)\.json", nombre
            )

            if not match:
                raise ValueError("Nombre inválido")

            tmdb_id = int(match.group(1))

            # Verificar contenido original
            original = storage.download(origen)
            datos = json.loads(original)

            if not isinstance(datos, dict):
                raise ValueError("JSON no es un objeto")

            if datos.get("id") != tmdb_id:
                raise ValueError("ID incorrecto")

            hash_original = sha256(original)

            resumen["verificados"] += 1

            if not execute:
                print(f"SIMULACIÓN: {origen} -> {destino}")
                continue

            # Copia dentro del mismo bucket
            storage.copy(origen, destino)

            # Verificación posterior
            copia = storage.download(destino)

            if sha256(copia) != hash_original:
                raise RuntimeError(
                    "Los hashes no coinciden"
                )

            destinos.add(nombre)
            resumen["copiados"] += 1

            print(f"VERIFICADO: {nombre}")

        except Exception as error:
            resumen["errores"].append({
                "archivo": nombre,
                "error": str(error),
            })
            print(f"ERROR: {nombre} - {error}")

    print("\nRESUMEN FINAL")
    print(json.dumps(
        resumen,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nArchivos originales conservados.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Realiza las copias en Supabase",
    )

    args = parser.parse_args()

    main(execute=args.execute)
