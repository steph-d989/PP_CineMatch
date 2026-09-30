from datetime import datetime

from src.ingestion.movielens import run_movielens_ingestion
from src.ingestion.tmdb import run_tmdb_ingestion
from src.ingestion.watchmode import run_watchmode_ingestion


### FUNCION QUE EJECUTA LA CAPA BROMZE

def run_bronze_pipeline():

    print("=" * 60)
    print("INICIO PIPELINE BRONZE CINEMATCH")
    print("=" * 60)

    fecha_inicio = datetime.now()

    resultados = {}

    try:
        print("\n1. Ingesta MovieLens")
        resultados["movielens"] = run_movielens_ingestion()

        print("\n2. Ingesta TMDb")
        resultados["tmdb"] = run_tmdb_ingestion(
            limit=10
        )

        print("\n3. Ingesta Watchmode")
        resultados["watchmode"] = run_watchmode_ingestion(
            limit=10,
            region="PE"
        )

        estado = "success"

    except Exception as error:
        estado = "failed"

        print("\nERROR EN PIPELINE BRONZE")
        print(error)

        resultados["error"] = str(error)

    fecha_fin = datetime.now()

    print("\n" + "=" * 60)
    print("RESUMEN PIPELINE BRONZE")
    print("=" * 60)

    print("Estado:", estado)
    print("Inicio:", fecha_inicio.isoformat())
    print("Fin:", fecha_fin.isoformat())

    print("=" * 60)

    return {
        "estado": estado,
        "fecha_inicio": fecha_inicio.isoformat(),
        "fecha_fin": fecha_fin.isoformat(),
        "resultados": resultados,
    }


if __name__ == "__main__":
    run_bronze_pipeline()