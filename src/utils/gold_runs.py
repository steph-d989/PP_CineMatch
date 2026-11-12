from pathlib import Path
from uuid import UUID
import json
import pandas as pd
import hashlib
import os
import tempfile

GOLD_REQUIRED_DATASETS = frozenset({
    "dim_user",
    "dim_movie",
    "dim_platform",
    "dim_date",
    "fact_rating",
    "fact_availability",
    "movie_metrics",
    "user_metrics",
    "availability_unlinked",
})




def build_gold_run_directories(
    gold_root: Path,
    run_id: str,
) -> dict[str, Path]:
    """
    Prepara las rutas de una ejecución Gold.

    Todas las carpetas comparten el mismo run_id
    utilizado por la ejecución Silver de origen.
    """

    version = str(UUID(run_id))

    run_root = (
        Path(gold_root)
        / "runs"
        / f"run_id={version}"
    )

    return {
        "root": run_root,
        "dimensions": run_root / "dimensions",
        "facts": run_root / "facts",
        "metrics": run_root / "metrics",
        "ml_content": run_root / "ml" / "content",
        "ml_collaborative": (
            run_root / "ml" / "collaborative"
        ),
        "ml_hybrid": run_root / "ml" / "hybrid",
        "quality": run_root / "quality",
    }

def validate_gold_run_paths(
    gold_dirs: dict[str, Path],
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> None:
    """
    Verifica que todos los archivos Gold pertenezcan
    al directorio raíz de la ejecución seleccionada.

    No lee, modifica ni publica archivos.
    """

    if "root" not in gold_dirs:
        raise ValueError(
            "Gold no contiene un directorio raíz."
        )

    raiz = Path(gold_dirs["root"]).resolve()

    # Comprobar que la carpeta representa un run_id.
    if not raiz.name.startswith("run_id="):
        raise ValueError(
            "Gold no contiene una raíz versionada válida."
        )

    try:
        UUID(raiz.name.removeprefix("run_id="))
    except (ValueError, AttributeError):
        raise ValueError(
            "Gold contiene un run_id inválido."
        ) from None

    # Validar todas las rutas, incluidas las particiones.
    archivos = []

    for nombre, referencia in rutas.items():

        if isinstance(referencia, list):
            if not referencia:
                raise ValueError(
                    f"Gold no contiene archivos para {nombre}."
                )
            archivos.extend(referencia)

        elif isinstance(referencia, Path):
            archivos.append(referencia)

        else:
            raise ValueError(
                f"Gold contiene una ruta inválida: {nombre}."
            )

    archivos.append(Path(metadata_path))

    for archivo in archivos:

        ruta = Path(archivo).resolve()

        try:
            relativa = ruta.relative_to(raiz)
        except ValueError:
            raise ValueError(
                "Gold contiene archivos fuera "
                "del run_id seleccionado."
            ) from None

        if not relativa.parts or ruta == raiz:
            raise ValueError(
                "Gold contiene una ruta de archivo inválida."
            )

def validate_gold_run_directories(
    gold_dirs: dict[str, Path],
) -> None:
    """
    Verifica que los directorios Gold pertenezcan
    a una única ejecución versionada.

    No crea carpetas ni modifica archivos.
    """

    obligatorios = {
        "root",
        "dimensions",
        "facts",
        "metrics",
        "quality",
    }

    faltantes = obligatorios - set(gold_dirs)

    if faltantes:
        raise ValueError(
            "Gold tiene directorios obligatorios faltantes: "
            + ", ".join(sorted(faltantes))
        )

    raiz = Path(gold_dirs["root"]).resolve()

    if not raiz.name.startswith("run_id="):
        raise ValueError(
            "Gold no tiene una raíz versionada válida."
        )

    try:
        version = str(
            UUID(raiz.name.removeprefix("run_id="))
        )
    except ValueError:
        raise ValueError(
            "Gold contiene un run_id inválido."
        ) from None

    raiz_esperada = (
        raiz.parent / f"run_id={version}"
    ).resolve()

    if raiz != raiz_esperada:
        raise ValueError(
            "Gold contiene una raíz versionada inconsistente."
        )

    ubicaciones = {
        "dimensions": raiz / "dimensions",
        "facts": raiz / "facts",
        "metrics": raiz / "metrics",
        "quality": raiz / "quality",
    }

    for nombre, esperado in ubicaciones.items():

        actual = Path(gold_dirs[nombre]).resolve()

        if actual != esperado.resolve():
            raise ValueError(
                f"El directorio {nombre} no pertenece "
                "al run_id seleccionado."
            )

def finalize_gold_run_paths(
    gold_dirs: dict[str, Path],
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> dict:
    """
    Comprueba la coherencia de las rutas declaradas
    para una ejecución Gold versionada.

    No verifica aún existencia, esquemas ni hashes.
    No publica archivos.
    """

    validate_gold_run_directories(
        gold_dirs=gold_dirs,
    )

    validate_gold_run_paths(
        gold_dirs=gold_dirs,
        rutas=rutas,
        metadata_path=metadata_path,
    )

    run_id = str(
        UUID(
            Path(gold_dirs["root"])
            .name
            .removeprefix("run_id=")
        )
    )

    return {
        "run_id": run_id,
        "estado": "paths_verified",
        "root": Path(gold_dirs["root"]),
        "datasets": tuple(rutas.keys()),
        "metadata": Path(metadata_path),
    }

def verify_gold_run_files(
    gold_dirs: dict[str, Path],
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> dict:
    """
    Verifica los archivos físicos de Gold.

    Comprueba:
    - Que pertenezcan a la misma ejecución.
    - Que existan.
    - Que sean archivos regulares.
    - Que no estén vacíos.

    No calcula hashes ni publica archivos.
    """


    # Exigir todos los datasets obligatorios
    # antes de comprobar los archivos físicos.
    validate_gold_required_datasets(
        rutas=rutas,
    )

    # Comprobar que todas las rutas declaradas
    # pertenecen a la misma ejecución Gold.
    cierre = finalize_gold_run_paths(
        gold_dirs=gold_dirs,
        rutas=rutas,
        metadata_path=metadata_path,
    )


    archivos = []

    for nombre, referencia in rutas.items():

        if isinstance(referencia, list):
            archivos.extend(
                (nombre, Path(ruta))
                for ruta in referencia
            )
        else:
            archivos.append(
                (nombre, Path(referencia))
            )

    archivos.append(
        ("metadata", Path(metadata_path))
    )

    verificados = []

    for nombre, ruta in archivos:

        if not ruta.is_file():
            raise FileNotFoundError(
                f"Gold: archivo obligatorio inexistente: {ruta}"
            )

        tamanio = ruta.stat().st_size

        if tamanio <= 0:
            raise ValueError(
                f"Gold: archivo vacío: {ruta}"
            )

        verificados.append({
            "dataset": nombre,
            "ruta": ruta,
            "bytes": tamanio,
        })

    return {
        "run_id": cierre["run_id"],
        "estado": "files_verified",
        "archivos": verificados,
        "total_archivos": len(verificados),
    }

def validate_gold_required_datasets(
    rutas: dict[str, Path | list[Path]],
) -> None:
    """
    Comprueba que una ejecución Gold declare
    todos los datasets obligatorios.

    No lee ni modifica archivos.
    """

    if not isinstance(rutas, dict):
        raise ValueError(
            "Gold requiere un diccionario de rutas."
        )

    faltantes = GOLD_REQUIRED_DATASETS - set(rutas)

    if faltantes:
        raise ValueError(
            "Gold contiene datasets obligatorios faltantes: "
            + ", ".join(sorted(faltantes))
        )

    for nombre in GOLD_REQUIRED_DATASETS:

        referencia = rutas[nombre]

        if isinstance(referencia, list):
            if not referencia:
                raise ValueError(
                    f"Gold no contiene archivos para {nombre}."
                )

            if not all(
                isinstance(ruta, Path)
                for ruta in referencia
            ):
                raise ValueError(
                    f"Gold contiene rutas inválidas para {nombre}."
                )

        elif not isinstance(referencia, Path):
            raise ValueError(
                f"Gold contiene una ruta inválida: {nombre}."
            )

def validate_gold_file_formats(
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> None:
    """
    Comprueba que los archivos declarados sean
    Parquet legibles y que los metadatos contengan
    un documento JSON válido.

    No modifica archivos.
    """

    validate_gold_required_datasets(rutas)

    for nombre, referencia in rutas.items():

        archivos = (
            referencia
            if isinstance(referencia, list)
            else [referencia]
        )

        for archivo in archivos:

            ruta = Path(archivo)

            try:
                # Leer únicamente los metadatos del
                # esquema, sin cargar las filas completas.
                pd.read_parquet(ruta, columns=[])

            except (OSError, ValueError, ImportError) as exc:
                raise ValueError(
                    f"Gold contiene un Parquet inválido "
                    f"en {nombre}: {ruta}"
                ) from exc

    try:
        with Path(metadata_path).open(
            "r",
            encoding="utf-8",
        ) as archivo:
            metadata = json.load(archivo)

    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(
            "Gold contiene metadatos JSON inválidos."
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(
            "Gold requiere metadatos JSON tipo objeto."
        )

def verify_gold_run_artifacts(
    gold_dirs: dict[str, Path],
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> dict:
    """
    Ejecuta la verificación integral básica de
    los archivos Gold de una ejecución.

    No modifica archivos ni publica resultados.
    """

    # Paso 1: datasets, rutas, existencia y tamaño.
    verificacion = verify_gold_run_files(
        gold_dirs=gold_dirs,
        rutas=rutas,
        metadata_path=metadata_path,
    )

    # Paso 2: legibilidad básica de Parquet y JSON.
    validate_gold_file_formats(
        rutas=rutas,
        metadata_path=metadata_path,
    )

    return {
        **verificacion,
        "estado": "artifacts_verified",
        "formatos_validos": True,
    }

def build_gold_manifest(
    gold_dirs: dict[str, Path],
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> dict:
    """
    Construye el manifiesto de una ejecución Gold.

    Requiere artefactos físicos y formatos válidos.
    Calcula SHA-256 leyendo los archivos por bloques.

    No crea archivos ni realiza publicaciones.
    """

    verificacion = verify_gold_run_artifacts(
        gold_dirs=gold_dirs,
        rutas=rutas,
        metadata_path=metadata_path,
    )

    raiz = Path(gold_dirs["root"]).resolve()
    entradas = []
    rutas_registradas = set()

    for archivo in verificacion["archivos"]:

        ruta = Path(archivo["ruta"]).resolve()
        relativa = ruta.relative_to(raiz).as_posix()

        if relativa in rutas_registradas:
            raise ValueError(
                f"Gold contiene una ruta duplicada: {relativa}"
            )

        rutas_registradas.add(relativa)

        hash_archivo = hashlib.sha256()

        with ruta.open("rb") as origen:
            while True:
                bloque = origen.read(1024 * 1024)

                if not bloque:
                    break

                hash_archivo.update(bloque)

        entradas.append({
            "dataset": archivo["dataset"],
            "path": relativa,
            "bytes": ruta.stat().st_size,
            "sha256": hash_archivo.hexdigest(),
        })

    entradas.sort(key=lambda item: item["path"])

    return {
        "run_id": verificacion["run_id"],
        "estado": "manifest_built",
        "algoritmo": "sha256",
        "total_archivos": len(entradas),
        "archivos": entradas,
    }


def verify_gold_manifest(
    gold_dirs: dict[str, Path],
    manifest: dict,
) -> dict:
    """
    Verifica los archivos contra un manifiesto
    SHA-256 previamente generado.

    No modifica ni publica archivos.
    """

    validate_gold_run_directories(gold_dirs)

    raiz = Path(gold_dirs["root"]).resolve()
    run_id = str(
        UUID(raiz.name.removeprefix("run_id="))
    )

    if manifest.get("run_id") != run_id:
        raise ValueError(
            "El manifiesto Gold pertenece a otro run_id."
        )

    if manifest.get("algoritmo") != "sha256":
        raise ValueError(
            "El manifiesto Gold requiere SHA-256."
        )

    entradas = manifest.get("archivos")

    if not isinstance(entradas, list) or not entradas:
        raise ValueError(
            "El manifiesto Gold no contiene archivos válidos."
        )

    if manifest.get("total_archivos") != len(entradas):
        raise ValueError(
            "El manifiesto Gold tiene un conteo inconsistente."
        )

    rutas_vistas = set()

    for entrada in entradas:

        if not isinstance(entrada, dict):
            raise ValueError(
                "El manifiesto Gold contiene una entrada inválida."
            )

        relativa = entrada.get("path")
        esperado = entrada.get("sha256")
        bytes_esperados = entrada.get("bytes")

        if (
            not isinstance(relativa, str)
            or not relativa
            or not isinstance(esperado, str)
            or len(esperado) != 64
            or any(c not in "0123456789abcdef" for c in esperado)
            or type(bytes_esperados) is not int
            or bytes_esperados <= 0
        ):
            raise ValueError(
                "El manifiesto Gold contiene datos inválidos."
            )

        if relativa in rutas_vistas:
            raise ValueError(
                "El manifiesto Gold contiene rutas duplicadas."
            )

        rutas_vistas.add(relativa)

        # Las rutas del manifiesto son relativas al run.
        ruta = (raiz / relativa).resolve()

        if not ruta.is_relative_to(raiz) or ruta == raiz:
            raise ValueError(
                "El manifiesto Gold contiene una ruta externa."
            )

        if not ruta.is_file():
            raise FileNotFoundError(
                f"Gold no encuentra el archivo: {relativa}"
            )

        if ruta.stat().st_size != bytes_esperados:
            raise ValueError(
                f"Gold detectó un tamaño diferente: {relativa}"
            )

        hash_actual = hashlib.sha256()

        with ruta.open("rb") as archivo:
            while True:
                bloque = archivo.read(1024 * 1024)

                if not bloque:
                    break

                hash_actual.update(bloque)

        if hash_actual.hexdigest() != esperado:
            raise ValueError(
                f"Gold detectó una diferencia SHA-256: {relativa}"
            )

    return {
        "run_id": run_id,
        "estado": "manifest_verified",
        "total_archivos": len(entradas),
    }

def save_gold_manifest(
    gold_dirs: dict[str, Path],
    manifest: dict,
) -> Path:
    """
    Guarda el manifiesto Gold de forma atómica,
    sin sobrescribir un manifiesto existente.

    Utiliza un archivo temporal y un enlace físico.
    """

    validate_gold_run_directories(gold_dirs)

    # Verificar los archivos antes de registrar
    # sus huellas de integridad.
    verify_gold_manifest(
        gold_dirs=gold_dirs,
        manifest=manifest,
    )

    destino = (
        Path(gold_dirs["root"]) / "_manifest.json"
    )

    contenido = (
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )

    temporal = None

    try:
        # Crear el temporal en el mismo volumen
        # que el archivo definitivo.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destino.parent,
            prefix="._manifest_",
            suffix=".tmp",
            delete=False,
        ) as archivo:

            temporal = Path(archivo.name)

            archivo.write(contenido)
            archivo.flush()
            os.fsync(archivo.fileno())

        # Publicación atómica sin sobrescritura.
        # Si el destino existe, falla.
        os.link(temporal, destino)

    finally:
        # Limpiar el temporal incluso si falla
        # la escritura o la publicación.
        if temporal is not None:
            temporal.unlink(missing_ok=True)

    return destino


def seal_gold_run(
    gold_dirs: dict[str, Path],
    rutas: dict[str, Path | list[Path]],
    metadata_path: Path,
) -> dict:
    """
    Cierra una ejecución Gold con archivos físicos.

    Valida los artefactos, genera el manifiesto,
    lo publica sin sobrescribir y verifica los
    hashes registrados.

    No realiza publicaciones remotas.
    """

    validate_gold_run_directories(gold_dirs)

    manifest_path = (
        Path(gold_dirs["root"]) / "_manifest.json"
    )

    if manifest_path.exists():
        raise FileExistsError(
            "La ejecución Gold ya tiene un manifiesto."
        )

    # Verifica datasets obligatorios, rutas,
    # existencia y legibilidad básica.
    manifest = build_gold_manifest(
        gold_dirs=gold_dirs,
        rutas=rutas,
        metadata_path=metadata_path,
    )

    # Persistencia sin sobrescritura.
    manifest_path = save_gold_manifest(
        gold_dirs=gold_dirs,
        manifest=manifest,
    )

    # Recuperar exactamente lo que quedó guardado.
    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as archivo:
        manifest_guardado = json.load(archivo)

    resultado = verify_gold_manifest(
        gold_dirs=gold_dirs,
        manifest=manifest_guardado,
    )

    return {
        "run_id": resultado["run_id"],
        "estado": "sealed",
        "manifest_path": manifest_path,
        "total_archivos": resultado["total_archivos"],
    }
