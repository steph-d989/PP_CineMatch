
"""
Orquestador de la capa Silver de CineMatch.

Prepara el plan y las rutas de una ejecución
sin procesar datos ni realizar conexiones externas.
"""

from pathlib import Path

from src.utils.silver_runs import (
    build_silver_run_directories,
)

from src.transformation.bronze_to_silver import (
    run_movielens_bronze_to_silver,
    run_tmdb_bronze_to_silver,
    run_watchmode_bronze_to_silver,
)

from src.transformation.silver_integration import (
    run_silver_integration,
)

from src.load.silver_storage import (
    verify_silver_local_manifest,
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


            # No aceptar transformaciones sin resultado.
            if resultado is None:
                raise ValueError(
                    f"La transformación {fuente} "
                    "no devolvió ningún resultado."
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


def validate_silver_run(
    plan: dict,
    execution_result: dict,
) -> dict:
    """
    Valida los manifiestos locales de una ejecución Silver.

    No publica archivos ni realiza conexiones externas.
    """

    from src.load.silver_storage import (
        build_silver_manifest,
        validate_silver_manifests,
    )

    run_id = plan["run_id"]

    # No validar resultados de otra ejecución.
    if execution_result.get("run_id") != run_id:
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": "El run_id de la ejecución no coincide.",
        }

    # Solo se validan transformaciones completadas.
    if execution_result.get("estado") != "transformed":
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": "La ejecución Silver no está transformada.",
        }
        
    # Comprobar que finalizaron todas las fuentes.
    fuentes_completadas = execution_result.get(
        "fuentes_completadas"
    )

    if (
        not isinstance(fuentes_completadas, (tuple, list))
        or tuple(fuentes_completadas) != SILVER_EXECUTION_ORDER
    ):
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": (
                "La ejecución Silver no completó "
                "las cuatro fuentes en el orden requerido."
            ),
        }

    try:
        directorios = plan["directorios"]

        # Construir manifiestos de las cuatro fuentes.
        manifests = {}

        for fuente in SILVER_EXECUTION_ORDER:
            manifests[fuente] = build_silver_manifest(
                local_dir=directorios[fuente],
                source=fuente,
                run_id=run_id,
            )

        # Aplicar las validaciones existentes.
        evaluacion = validate_silver_manifests(
            manifests=manifests,
            run_id=run_id,
        )

        if evaluacion.get("estado") != "ready":
            return {
                "run_id": run_id,
                "estado": "blocked",
                "evaluacion": evaluacion,
            }

        return {
            "run_id": run_id,
            "estado": "validated",
            "evaluacion": evaluacion,
            "manifests": manifests,
        }

    except Exception as error:
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": str(error),
        }


def reverify_silver_run(
    plan: dict,
    validation_result: dict,
) -> dict:
    """
    Recomprueba la integridad local de Silver
    utilizando los manifiestos ya validados.

    No reconstruye los manifiestos.
    No realiza conexiones externas.
    No publica archivos en Supabase.
    """

    run_id = plan["run_id"]

    if validation_result.get("run_id") != run_id:
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": "El run_id no coincide.",
        }

    if validation_result.get("estado") != "validated":
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": "La ejecución no está validada.",
        }

    manifests = validation_result.get("manifests")

    if (
        not isinstance(manifests, dict)
        or set(manifests) != set(SILVER_EXECUTION_ORDER)
    ):
        return {
            "run_id": run_id,
            "estado": "blocked",
            "error": "Los manifiestos Silver están incompletos.",
        }

    verificaciones = {}

    for fuente in SILVER_EXECUTION_ORDER:

        try:
            resultado = verify_silver_local_manifest(
                manifest=manifests[fuente],
                local_dir=plan["directorios"][fuente],
            )

            verificaciones[fuente] = resultado

            if resultado.get("estado") != "verified":
                return {
                    "run_id": run_id,
                    "estado": "blocked",
                    "fuente_fallida": fuente,
                    "verificaciones": verificaciones,
                }

        except Exception as error:
            return {
                "run_id": run_id,
                "estado": "blocked",
                "fuente_fallida": fuente,
                "error": str(error),
                "verificaciones": verificaciones,
            }

    return {
        "run_id": run_id,
        "estado": "locally_verified",
        "verificaciones": verificaciones,
        "manifests": manifests,
    }


def build_silver_transformers() -> dict:
    """
    Construye el registro de transformaciones Silver.

    Conecta las funciones reales de CineMatch
    sin ejecutarlas.

    No procesa archivos, no consulta APIs
    y no realiza operaciones en Supabase.
    """

    return {
        "movielens": run_movielens_bronze_to_silver,
        "tmdb": run_tmdb_bronze_to_silver,
        "watchmode": run_watchmode_bronze_to_silver,
        "integrated": run_silver_integration,
    }
