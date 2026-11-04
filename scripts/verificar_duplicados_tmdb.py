
import hashlib
import json

from src.utils.supabase_client import (
    get_supabase_client,
    plan_tmdb_migration,
)


def main():
    print("=" * 60)
    print("CINEMATCH - VERIFICACIÓN DE DUPLICADOS TMDB")
    print("=" * 60)

    plan = plan_tmdb_migration()
    duplicados = plan["duplicados"]

    if not duplicados:
        print("No existen películas duplicadas.")
        return

    cliente = get_supabase_client()
    storage = cliente.storage.from_("Bronze")

    for pelicula in duplicados:

        print("\nPelícula:", pelicula["archivo"])

        hashes = []

        for ruta in pelicula["rutas"]:

            # Descarga en memoria, sin modificar Storage
            contenido = storage.download(ruta)

            hash_archivo = hashlib.sha256(
                contenido
            ).hexdigest()

            hashes.append(hash_archivo)

            print("\nRuta:", ruta)
            print("SHA-256:", hash_archivo)
            print("Tamaño:", len(contenido), "bytes")

            try:
                datos = json.loads(contenido)
                print("TMDb ID:", datos.get("id"))
                print("Título:", datos.get("title"))
            except (ValueError, AttributeError):
                print("Advertencia: JSON inválido")

        if len(set(hashes)) == 1:
            print("\nRESULTADO: CONTENIDO IDÉNTICO")
        else:
            print("\nRESULTADO: VERSIONES DIFERENTES")

    print("\nVerificación completada.")
    print("No se modificaron archivos en Supabase.")


if __name__ == "__main__":
    main()
