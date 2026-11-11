
from pathlib import Path
from uuid import UUID


def build_silver_run_directories(
    silver_root: Path,
    run_id: str,
) -> dict[str, Path]:
    """
    Construye las carpetas locales de una
    ejecución Silver.

    No crea carpetas ni modifica archivos.
    """

    # Validar identificador de ejecución.
    try:
        run_id = str(UUID(run_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("run_id inválido") from None

    # Carpeta principal de la ejecución.
    run_root = (
        Path(silver_root)
        / "runs"
        / f"run_id={run_id}"
    )

    return {
        "root": run_root,
        "movielens": run_root / "movielens",
        "tmdb": run_root / "tmdb",
        "watchmode": run_root / "watchmode",
        "integrated": run_root / "integrated",
    }
