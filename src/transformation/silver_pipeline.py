
"""
Orquestador de la capa Silver de CineMatch.

Prepara el plan y las rutas de una ejecución
sin procesar datos ni realizar conexiones externas.
"""

from pathlib import Path

from src.utils.silver_runs import (
    build_silver_run_directories,
)


SILVER_EXECUTION_ORDER = (
    "movielens",
    "tmdb",
    "watchmode",
    "integrated",
)


def prepare_silver_run(
    silver_root: Path,
    run_id: str,
) -> dict:
    """
    Prepara una ejecución Silver.

    Todas las fuentes comparten el mismo run_id.
    Integrated se ejecutará después de las otras
    tres fuentes.

    Esta función no crea carpetas ni escribe datos.
    """

    run_dirs = build_silver_run_directories(
        silver_root=Path(silver_root),
        run_id=run_id,
    )

    return {
        "run_id": run_dirs["root"].name.removeprefix(
            "run_id="
        ),
        "estado": "planned",
        "orden": SILVER_EXECUTION_ORDER,
        "directorios": run_dirs,
    }


def execute_silver_run(
    plan: dict,
    transformers: dict,
) -> dict:
    """
    Ejecuta las transformaciones Silver en orden.

    Los transformers se proporcionan explícitamente
    para permitir pruebas sin procesar archivos reales.

    No publica resultados en Supabase.
    """

    if plan["estado"] != "planned":
        raise ValueError(
            "La ejecución Silver debe estar en estado planned."
        )

    orden = tuple(plan["orden"])
    directorios = plan["directorios"]

    if orden != SILVER_EXECUTION_ORDER:
        raise ValueError(
            "El orden de ejecución Silver no es válido."
        )

    if set(transformers) != set(orden):
        raise ValueError(
            "Los transformers no coinciden con las fuentes Silver."
        )

    resultados = {}

    for fuente in orden:

        transformador = transformers[fuente]

        try:
            if fuente == "integrated":
                resultado = transformador(
                    run_dirs=directorios,
                )
            else:
                resultado = transformador(
                    output_dir=directorios[fuente],
                )

            resultados[fuente] = resultado

        except Exception as error:
            return {
                "run_id": plan["run_id"],
                "estado": "failed",
                "fuente_fallida": fuente,
                "error": str(error),
                "fuentes_completadas": tuple(
                    resultados.keys()
                ),
            }

    return {
        "run_id": plan["run_id"],
        "estado": "transformed",
        "fuentes_completadas": tuple(
            resultados.keys()
        ),
        "resultados": resultados,
    }
