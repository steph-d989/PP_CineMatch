
import json
import re
import hashlib

from src.utils.supabase_client import (
    get_supabase_client,
    plan_tmdb_migration,
)


def main():
    print("=" * 60)
    print("CINEMATCH - PREPARACIÓN DE MIGRACIÓN TMDB")
    print("=" * 60)

    plan = plan_tmdb_migration()

    storage = get_supabase_client().storage.from_("Bronze")

    pendientes = plan["pendientes"]
    preparados = []
    errores = []

    for item in pendientes:
        origen = item["origen"]
        destino = item["destino"]

        print(f"\nVerificando: {item['archivo']}")

        try:
            coincidencia = re.fullmatch(
                r"movie_(\d+)\.json",
                item["archivo"],
            )

            if not coincidencia:
                raise ValueError(
                    "Nombre de archivo inválido"
                )

            tmdb_id = int(coincidencia.group(1))

            contenido = storage.download(origen)
            datos = json.loads(contenido)

            if not isinstance(datos, dict):
                raise ValueError(
                    "El JSON no contiene un objeto"
                )

            if datos.get("id") != tmdb_id:
                raise ValueError(
                    "El ID interno no coincide"
                )

            sha256 = hashlib.sha256(
                contenido
            ).hexdigest()

            preparados.append({
                "origen": origen,
                "destino": destino,
                "sha256": sha256,
            })

            print("Origen:", origen)
            print("Destino:", destino)
            print("Integridad: OK")

        except Exception as error:
            errores.append({
                "archivo": item["archivo"],
                "error": str(error),
            })

            print("ERROR:", error)

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print("Pendientes:", len(pendientes))
    print("Preparados:", len(preparados))
    print("Errores:", len(errores))

    print("\nMODO SIMULACIÓN")
    print("No se copiaron ni eliminaron archivos.")


if __name__ == "__main__":
    main()
