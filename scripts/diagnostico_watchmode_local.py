
from src.ingestion.watchmode import (
    inventory_legacy_watchmode_files,
    get_local_watchmode_files,
)


def main():
    print("=" * 60)
    print("CINEMATCH - DIAGNÓSTICO LOCAL WATCHMODE")
    print("=" * 60)

    inventario = inventory_legacy_watchmode_files()

    archivos_pe = get_local_watchmode_files(
        region="PE"
    )

    print("\nARCHIVOS ANTIGUOS")
    print("-" * 60)

    print("Total:", inventario["total"])
    print("Válidos:", len(inventario["validos"]))
    print("Inválidos:", len(inventario["invalidos"]))

    print("\nARCHIVOS REGIONALES")
    print("-" * 60)
    print("Perú (PE):", len(archivos_pe))

    print("\nDETALLE DE ARCHIVOS ANTIGUOS")
    print("-" * 60)

    for item in inventario["validos"]:
        print(
            item["archivo"],
            "| fuentes:",
            item["sources_count"]
        )

    for item in inventario["invalidos"]:
        print(
            item["archivo"],
            "| ERROR:",
            item["error"]
        )

    print("\nDiagnóstico completado.")
    print("No se modificaron archivos.")


if __name__ == "__main__":
    main()
