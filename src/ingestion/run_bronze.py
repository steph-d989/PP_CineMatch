from datetime import datetime

from src.ingestion.movielens import run_movielens_ingestion
from src.ingestion.tmdb import run_tmdb_ingestion
from src.ingestion.watchmode import (
    run_watchmode_ingestion,
    sync_watchmode_bundles,
    diagnose_watchmode_sync
)
from src.utils.paths import BRONZE_MOVIELENS_DIR


def evaluate_bronze_status(resultados: dict) -> str:
    """
    Determina el estado global del pipeline
    a partir del resultado de cada fuente.
    """

    if not resultados:
        return "failed"

    estados = []

    for resultado in resultados.values():

        if not isinstance(resultado, dict):
            return "failed"

        estado = resultado.get("estado")

        if estado not in {
            "success",
            "partial",
            "failed",
            "skipped",
        }:
            return "failed"

        estados.append(estado)

    if all(estado == "success" for estado in estados):
        return "success"

    if all(estado == "failed" for estado in estados):
        return "failed"

    if all(estado == "skipped" for estado in estados):
        return "skipped"

    return "partial"


def execute_bronze_source(
    name: str,
    function,
    **kwargs,
) -> dict:
    """
    Ejecuta una fuente Bronze y captura sus errores.

    Permite que el orquestador reciba un resultado
    estructurado incluso cuando una fuente falla.
    """

    try:
        resultado = function(**kwargs)

        if not isinstance(resultado, dict):
            raise TypeError(
                f"{name} no devolvió un diccionario"
            )

        estado = resultado.get("estado")

        if estado not in {
            "success",
            "partial",
            "failed",
            "skipped",
        }:
            raise ValueError(
                f"{name} devolvió un estado inválido"
            )

        return resultado

    except Exception as error:
        print(
            f"ERROR EN {name}: {error}"
        )

        return {
            "estado": "failed",
            "fuente": name,
            "error": str(error),
        }


def movielens_links_available() -> bool:
    """
    Comprueba que links.csv de MovieLens
    esté disponible y tenga las columnas
    necesarias para las ingestas dependientes.
    """

    import pandas as pd

    ruta = (
        BRONZE_MOVIELENS_DIR
        / "ml-32m"
        / "links.csv"
    )

    if not ruta.is_file():
        return False

    try:
        columnas = pd.read_csv(
            ruta,
            nrows=0,
        ).columns

        return (
            "movieId" in columnas
            and "tmdbId" in columnas
        )

    except (OSError, ValueError, pd.errors.ParserError):
        return False


### FUNCION QUE EJECUTA LA CAPA BROMZE

def run_bronze_pipeline(
    sync_dry_run: bool = True,
    execute_ingestion: bool = False,
) -> dict:

    # 0. Protección contra ingestas no autorizadas
    if not execute_ingestion:
        print(
            "Ejecución bloqueada por seguridad. "
            "No se realizaron ingestas ni sincronizaciones."
        )

        return {
            "estado": "skipped",
            "motivo": "ingestion_not_authorized",
            "resultados": {},
        }

    print("=" * 60)
    print("INICIO PIPELINE BRONZE CINEMATCH")
    print("=" * 60)

    fecha_inicio = datetime.now()
    resultados = {}

    # 1. Ingesta MovieLens
    print("\n1. Ingesta MovieLens")

    resultados["movielens"] = execute_bronze_source(
        "MovieLens",
        run_movielens_ingestion,
    )

    # Verificar disponibilidad de links.csv
    links_disponibles = movielens_links_available()

    if not links_disponibles:

        print(
            "\nNo hay links.csv válido. "
            "Se omiten TMDb y Watchmode."
        )

        resultados["tmdb"] = {
            "estado": "skipped",
            "motivo": "movielens_links_unavailable",
        }

        resultados["watchmode"] = {
            "estado": "skipped",
            "motivo": "movielens_links_unavailable",
        }

    else:

        # 2. Ingesta TMDb
        print("\n2. Ingesta TMDb")

        resultados["tmdb"] = execute_bronze_source(
            "TMDb",
            run_tmdb_ingestion,
            limit=10,
        )

        # 3. Ingesta Watchmode
        print("\n3. Ingesta Watchmode")

        resultados["watchmode"] = execute_bronze_source(
            "Watchmode",
            run_watchmode_ingestion,
            limit=10,
            region="PE",
        )

    # 4. Sincronización independiente de Watchmode
    # Se ejecuta aunque MovieLens no tenga links.csv.

    if sync_dry_run:

        print(
            "\n4. Diagnóstico de sincronización Watchmode"
        )

        resultados["watchmode_sync"] = execute_bronze_source(
            "Watchmode Sync",
            diagnose_watchmode_sync,
            region="PE",
        )

    else:

        print(
            "\n4. Sincronización Watchmode con Supabase"
        )

        resultados["watchmode_sync"] = execute_bronze_source(
            "Watchmode Sync",
            sync_watchmode_bundles,
            region="PE",
        )

    # 5. Evaluar estado global
    estado = evaluate_bronze_status(resultados)

    fecha_fin = datetime.now()

    print("\n" + "=" * 60)
    print("RESUMEN PIPELINE BRONZE")
    print("=" * 60)

    print("Estado:", estado)
    print("Inicio:", fecha_inicio.isoformat())
    print("Fin:", fecha_fin.isoformat())

    for fuente, resultado in resultados.items():
        print(
            f"{fuente}: {resultado.get('estado')}"
        )

    print("=" * 60)

    return {
        "estado": estado,
        "fecha_inicio": fecha_inicio.isoformat(),
        "fecha_fin": fecha_fin.isoformat(),
        "resultados": resultados,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline Bronze de CineMatch"
    )

    parser.add_argument(
        "--execute-ingestion",
        action="store_true",
        help=(
            "Autoriza la ejecución real de las "
            "ingestas MovieLens, TMDb y Watchmode"
        ),
    )

    args = parser.parse_args()

    if not args.execute_ingestion:
        print(
            "Ejecución bloqueada por seguridad.\n"
            "No se ejecutaron ingestas ni "
            "sincronizaciones.\n"
            "Para habilitar la ingesta real, "
            "se requiere --execute-ingestion."
        )
    else:
        run_bronze_pipeline(
            sync_dry_run=True
        )
