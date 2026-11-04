
from collections import Counter
from datetime import datetime

from src.utils.supabase_client import (
    list_tmdb_legacy_files,
    plan_tmdb_migration,
)


def main():

    print("=" * 60)
    print("CINEMATCH - DIAGNÓSTICO TMDB")
    print("=" * 60)

    fecha_inicio = datetime.now()

    print("\nConsultando inventario de Supabase...")

    inventario = list_tmdb_legacy_files()

    print("Preparando plan de migración...")

    plan = plan_tmdb_migration()

    # Conteo de versiones por película
    conteo = Counter(
        item["archivo"] for item in inventario
    )

    print("\nRESULTADOS")
    print("-" * 60)

    print("Archivos históricos:", len(inventario))

    print(
        "Películas únicas:",
        plan["peliculas_unicas"]
    )

    print(
        "Pendientes de migración:",
        len(plan["pendientes"])
    )

    print(
        "Ya existentes en destino:",
        len(plan["existentes"])
    )

    print(
        "Películas con duplicados:",
        len(plan["duplicados"])
    )

    print("\nDUPLICADOS")

    for nombre, cantidad in conteo.most_common(10):
        if cantidad > 1:
            print(f"{nombre}: {cantidad} versiones")

    duracion = (
        datetime.now() - fecha_inicio
    ).total_seconds()

    print("\nTiempo de diagnóstico:", duracion, "s")

    print("\nNo se modificó ningún archivo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
