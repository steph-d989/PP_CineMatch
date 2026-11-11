import hashlib
import re

from datetime import date
from pathlib import Path
from uuid import UUID, uuid4

from src.utils.paths import (
    SILVER_MOVIELENS_DIR,
    SILVER_TMDB_DIR,
    SILVER_WATCHMODE_DIR,
    SILVER_INTEGRATED_DIR,
)

from src.utils.supabase_client import (
    upload_file_to_storage,
    get_supabase_client,
)


def build_silver_versioned_path(
    source: str,
    relative_path: str,
    run_id: str,
) -> str:
    """
    Construye una ruta Silver asociada
    a una ejecución específica.

    No crea carpetas ni realiza conexiones.
    """

    if source not in {
        "movielens",
        "tmdb",
        "watchmode",
        "integrated",
    }:
        raise ValueError("Fuente Silver inválida")

    # Validar que run_id sea un UUID válido.
    try:
        version = str(UUID(run_id))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("run_id inválido") from None

    # Impedir rutas absolutas o relativas peligrosas.
    ruta = Path(relative_path.replace("\\", "/"))

    partes = relative_path.replace("\\", "/").split("/")

    if (
        ruta.is_absolute()
        or not partes
        or any(
            parte in ("", ".", "..")
            for parte in partes
        )
        or ":" in relative_path
    ):
        raise ValueError("Ruta relativa inválida")

    archivo = "/".join(partes)

    return (
        f"{source}/"
        f"run_id={version}/"
        f"{archivo}"
    )


def verify_silver_local_manifest(
    manifest: dict,
    local_dir: Path,
) -> dict:
    """
    Comprueba que los archivos locales conservan
    el contenido registrado en el manifiesto.

    Solo realiza lecturas.
    No modifica archivos ni consulta Supabase.
    """

    local_dir = Path(local_dir)

    verificados = []
    conflictos = []
    errores = []

    for item in manifest["archivos"]:

        nombre = item["archivo"]
        ruta = local_dir / nombre

        try:
            # Evitar rutas que salgan de la carpeta.
            ruta.resolve().relative_to(
                local_dir.resolve()
            )

            if not ruta.is_file():
                errores.append({
                    "archivo": nombre,
                    "motivo": "file_missing",
                })
                continue

            calculo_hash = hashlib.sha256()

            with ruta.open("rb") as contenido:
                for bloque in iter(
                    lambda: contenido.read(1024 * 1024),
                    b"",
                ):
                    calculo_hash.update(bloque)

            hash_actual = calculo_hash.hexdigest()

            if (
                hash_actual == item["sha256"]
                and ruta.stat().st_size
                == item["size_bytes"]
            ):
                verificados.append(nombre)

            else:
                conflictos.append({
                    "archivo": nombre,
                    "motivo": "local_file_changed",
                })

        except (OSError, ValueError, KeyError) as error:
            errores.append({
                "archivo": nombre,
                "motivo": str(error),
            })

    completo = (
        manifest.get("estado") == "complete"
        and len(verificados) == len(manifest["archivos"])
        and not conflictos
        and not errores
    )

    return {
        "source": manifest["source"],
        "run_id": manifest["run_id"],
        "estado": "verified" if completo else "failed",
        "verificados": verificados,
        "conflictos": conflictos,
        "errores": errores,
    }


def build_silver_manifest(
    local_dir: Path,
    source: str,
    run_id: str,
) -> dict:
    """
    Construye un inventario verificable de
    los archivos de una fuente Silver.

    Solo utiliza archivos locales.
    No modifica archivos ni consulta Supabase.
    """

    local_dir = Path(local_dir)

    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"No existe la carpeta Silver: {local_dir}"
        )

    archivos_obligatorios = {
        "movielens": {
            "movies.parquet",
            "links.parquet",
            "tags.parquet",
            "_metadata.json",
        },
        "tmdb": {
            "movies_tmdb.parquet",
            "_metadata.json",
        },
        "watchmode": {
            "availability.parquet",
            "availability_status.parquet",
            "_metadata.json",
        },
        "integrated": {
            "movies_integrated.parquet",
            "_metadata.json",
        },
    }

    if source not in archivos_obligatorios:
        raise ValueError("Fuente Silver inválida")

    # Validar y normalizar el run_id.
    run_id = str(UUID(run_id))

    archivos = sorted(
        (
            archivo for archivo in local_dir.rglob("*")
            if archivo.is_file()
        ),
        key=lambda archivo: archivo.relative_to(
            local_dir
        ).as_posix(),
    )

    nombres = {
        archivo.relative_to(local_dir).as_posix()
        for archivo in archivos
    }

    faltantes = sorted(
        archivos_obligatorios[source] - nombres
    )

    # MovieLens debe contener al menos
    # una partición de calificaciones.
    if source == "movielens":
        particiones = [
            nombre for nombre in nombres
            if nombre.startswith("ratings/")
            and nombre.endswith(".parquet")
        ]

        if not particiones:
            faltantes.append("ratings/*.parquet")

    registros = []

    for archivo in archivos:

        ruta_relativa = (
            archivo.relative_to(local_dir).as_posix()
        )

        # Reutilizar la validación de rutas.
        ruta_remota = build_silver_versioned_path(
            source=source,
            relative_path=ruta_relativa,
            run_id=run_id,
        )

        # Calcular SHA-256 por bloques para
        # no cargar archivos grandes en memoria.
        calculo_hash = hashlib.sha256()

        with archivo.open("rb") as contenido:
            for bloque in iter(
                lambda: contenido.read(1024 * 1024),
                b"",
            ):
                calculo_hash.update(bloque)

        registros.append({
            "archivo": ruta_relativa,
            "remote_path": ruta_remota,
            "size_bytes": archivo.stat().st_size,
            "sha256": calculo_hash.hexdigest(),
        })

    completo = len(faltantes) == 0

    return {
        "source": source,
        "run_id": run_id,
        "estado": (
            "complete" if completo else "incomplete"
        ),
        "archivos_esperados": len(registros),
        "archivos": registros,
        "faltantes": sorted(faltantes),
    }


def validate_silver_manifests(
    manifests: dict,
    run_id: str,
) -> dict:
    """
    Valida los manifiestos de las cuatro fuentes
    antes de autorizar una publicación Silver.

    No realiza conexiones ni modifica archivos.
    """

    run_id = str(UUID(run_id))

    fuentes = {
        "movielens",
        "tmdb",
        "watchmode",
        "integrated",
    }

    faltantes = sorted(
        fuentes - set(manifests.keys())
    )

    bloqueados = {}
    validados = []

    for fuente in sorted(fuentes):

        manifiesto = manifests.get(fuente)

        if not isinstance(manifiesto, dict):
            bloqueados[fuente] = "missing_manifest"
            continue

        if manifiesto.get("source") != fuente:
            bloqueados[fuente] = "invalid_source"
            continue

        if manifiesto.get("run_id") != run_id:
            bloqueados[fuente] = "run_id_mismatch"
            continue

        if (
            manifiesto.get("estado") != "complete"
            or manifiesto.get("faltantes")
        ):
            bloqueados[fuente] = "incomplete_manifest"
            continue

        archivos = manifiesto.get("archivos")

        if (
            not isinstance(archivos, list)
            or not archivos
            or manifiesto.get("archivos_esperados")
            != len(archivos)
        ):
            bloqueados[fuente] = "invalid_inventory"
            continue

        # Comprobar que los registros tengan hashes
        # y rutas remotas coherentes.
        registros_validos = all(
            isinstance(item, dict)
            and isinstance(item.get("archivo"), str)
            and isinstance(item.get("sha256"), str)
            and bool(
                re.fullmatch(
                    r"[0-9a-fA-F]{64}",
                    item["sha256"],
                )
            )
            and item.get("remote_path")
            == build_silver_versioned_path(
                source=fuente,
                relative_path=item["archivo"],
                run_id=run_id,
            )
            for item in archivos
        )

        if not registros_validos:
            bloqueados[fuente] = "invalid_file_record"
            continue

        validados.append(fuente)

    autorizado = (
        not faltantes
        and not bloqueados
        and len(validados) == len(fuentes)
    )

    return {
        "run_id": run_id,
        "estado": (
            "ready" if autorizado else "blocked"
        ),
        "autorizado": autorizado,
        "fuentes_validadas": validados,
        "fuentes_faltantes": faltantes,
        "bloqueados": bloqueados,
    }


def verify_silver_remote_manifest(
    manifest: dict,
    bucket: str = "Silver",
) -> dict:
    """
    Verifica los archivos remotos de un manifiesto
    comparando sus SHA-256.

    Solo descarga y lee archivos.
    No realiza subidas ni modificaciones.
    """

    if manifest.get("estado") != "complete":
        raise ValueError(
            "No se puede verificar un manifiesto incompleto"
        )

    verificados = []
    conflictos = []
    errores = []

    archivos = manifest.get("archivos", [])

    if not archivos:
        raise ValueError("El manifiesto no contiene archivos")

    storage = get_supabase_client().storage.from_(bucket)

    for item in archivos:
        ruta = item["remote_path"]
        hash_esperado = item["sha256"]

        try:
            contenido = storage.download(ruta)

            hash_remoto = hashlib.sha256(
                contenido
            ).hexdigest()

            if hash_remoto == hash_esperado:
                verificados.append(ruta)

            else:
                conflictos.append({
                    "archivo": ruta,
                    "motivo": "sha256_mismatch",
                })

        except Exception as error:
            errores.append({
                "archivo": ruta,
                "error": str(error),
            })

    completo = (
        len(verificados) == len(archivos)
        and not conflictos
        and not errores
    )

    return {
        "source": manifest["source"],
        "run_id": manifest["run_id"],
        "estado": "verified" if completo else "failed",
        "verificados": verificados,
        "conflictos": conflictos,
        "errores": errores,
    }


### FUNCIÓN QUE SUBE LOS ARCHIVOS DE UNA CARPETA SILVER AL BUCKET SILVER

def upload_silver_directory(
    local_dir: Path,
    source: str,
    run_id: str | None = None,
) -> dict:

    # Si no recibimos un identificador,
    # generamos uno nuevo para esta ejecución.
    if run_id is None:
        run_id = str(uuid4())

    # Validar el identificador antes de subir archivos.
    run_id = str(UUID(run_id))

    resultados = {
        "source": source,
        "run_id": run_id,
        "subidos": 0,
        "existentes": 0,
        "errores": [],
    }


    # Verificar que la carpeta exista
    if not local_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta Silver: {local_dir}"
        )

    # Buscar archivos también dentro de subcarpetas
    archivos = [
        archivo
        for archivo in local_dir.rglob("*")
        if archivo.is_file()
    ]

    if not archivos:
        print(
            f"{source} -> no se encontraron archivos para subir."
        )

        return resultados

    for archivo in archivos:

        # Mantener estructura interna.
        # Ejemplo:
        # ratings/part_000.parquet
        relative_path = (
            archivo
            .relative_to(local_dir)
            .as_posix()
        )


        remote_path = build_silver_versioned_path(
            source=source,
            relative_path=relative_path,
            run_id=run_id,
        )


        try:
            upload_file_to_storage(
                bucket="Silver",
                local_path=archivo,
                remote_path=remote_path,
                upsert=False,
            )

            resultados["subidos"] += 1

            print(
                f"{source} | {relative_path} -> OK"
            )

        except Exception as error:

            mensaje = str(error)

            # Supabase puede devolver distintos mensajes
            # cuando el objeto ya existe.
            mensaje_lower = mensaje.lower()

            if (
                "already exists" in mensaje_lower
                or "duplicate" in mensaje_lower
                or "resource already exists" in mensaje_lower
            ):

                # El objeto existe: comprobar su contenido.
                try:
                    comparacion = compare_silver_storage_file(
                        local_path=archivo,
                        remote_path=remote_path,
                        bucket="Silver",
                    )

                    if comparacion["estado"] == "identical":

                        resultados["existentes"] += 1

                        print(
                            f"{source} | {relative_path} -> "
                            "YA EXISTE Y ES IDÉNTICO"
                        )

                    elif comparacion["estado"] == "conflict":

                        resultados["errores"].append({
                            "archivo": relative_path,
                            "error": "remote_object_exists_unverified",
                            "detalle": "SHA-256 diferente",
                        })

                        print(
                            f"{source} | {relative_path} -> "
                            "CONFLICTO: contenido diferente"
                        )

                    else:
                        raise ValueError(
                            "Resultado de comparación desconocido"
                        )

                except Exception as error_comparacion:

                    resultados["errores"].append({
                        "archivo": relative_path,
                        "error": "remote_comparison_failed",
                        "detalle": str(error_comparacion),
                    })

                    print(
                        f"{source} | {relative_path} -> "
                        "ERROR AL COMPARAR"
                    )
            
            else:
                # Error de subida distinto de objeto existente.
                # No debe ignorarse ni contabilizarse como éxito.
                resultados["errores"].append({
                    "archivo": relative_path,
                    "error": "storage_upload_failed",
                    "detalle": mensaje,
                })

                print(
                    f"{source} | {relative_path} -> "
                    f"ERROR DE SUBIDA: {mensaje}"
                )


    return resultados



def compare_silver_storage_file(
    local_path: Path,
    remote_path: str,
    bucket: str = "Silver",
) -> dict:
    """
    Compara un archivo Silver local con uno remoto
    utilizando SHA-256.

    Solo realiza lecturas.
    No sube, modifica ni elimina archivos.
    """

    local_path = Path(local_path)

    if not local_path.is_file():
        raise FileNotFoundError(
            f"No existe el archivo local: {local_path}"
        )

    # 1. Calcular el hash del archivo local.
    hash_local = hashlib.sha256(
        local_path.read_bytes()
    ).hexdigest()

    # 2. Descargar el contenido remoto.
    cliente = get_supabase_client()

    storage = cliente.storage.from_(bucket)

    contenido_remoto = storage.download(
        remote_path
    )

    # 3. Calcular el hash del archivo remoto.
    hash_remoto = hashlib.sha256(
        contenido_remoto
    ).hexdigest()

    # 4. Comparar ambos hashes.
    iguales = hash_local == hash_remoto

    return {
        "archivo": local_path.name,
        "remote_path": remote_path,
        "hash_local": hash_local,
        "hash_remoto": hash_remoto,
        "iguales": iguales,
        "estado": (
            "identical"
            if iguales
            else "conflict"
        ),
    }


def evaluate_silver_publication(
    resultados: dict,
) -> dict:
    """
    Evalúa si una publicación Silver terminó
    correctamente en todas sus fuentes.

    No realiza conexiones ni modifica archivos.
    """

    fuentes_esperadas = {
        "movielens",
        "tmdb",
        "watchmode",
        "integrated",
    }

    fuentes_recibidas = set(resultados.keys())

    faltantes = sorted(
        fuentes_esperadas - fuentes_recibidas
    )

    errores = 0
    archivos_procesados = 0
    detalle = {}

    for fuente in fuentes_esperadas:

        resultado = resultados.get(fuente)

        if not isinstance(resultado, dict):
            detalle[fuente] = "missing"
            continue

        subidos = resultado.get("subidos", 0)
        existentes = resultado.get("existentes", 0)
        errores_fuente = resultado.get("errores", [])

        if (
            not isinstance(subidos, int)
            or isinstance(subidos, bool)
            or subidos < 0
            or not isinstance(existentes, int)
            or isinstance(existentes, bool)
            or existentes < 0
            or not isinstance(errores_fuente, list)
        ):
            detalle[fuente] = "invalid_result"
            errores += 1
            continue

        errores += len(errores_fuente)
        archivos_procesados += subidos + existentes

        if errores_fuente:
            detalle[fuente] = "failed"

        elif subidos + existentes == 0:
            detalle[fuente] = "empty"

        else:
            detalle[fuente] = "success"

    if all(
        detalle.get(fuente) == "success"
        for fuente in fuentes_esperadas
    ):
        estado = "success"

    elif archivos_procesados == 0:
        estado = "failed"

    else:
        estado = "partial"

    return {
        "estado": estado,
        "fuentes_faltantes": faltantes,
        "errores": errores,
        "archivos_procesados": archivos_procesados,
        "detalle": detalle,
    }


### FUNCIÓN QUE PUBLICA TODA LA CAPA SILVER LOCAL EN SUPABASE


def upload_silver_to_supabase(
    run_id: str | None = None,
) -> dict:

    # Crear un único identificador para las
    # cuatro fuentes de esta publicación.
    if run_id is None:
        run_id = str(uuid4())

    # Validar y normalizar el UUID.
    run_id = str(UUID(run_id))

    print("=" * 60)
    print("SUBIDA SILVER A SUPABASE")
    print("Run ID:", run_id)
    print("=" * 60)

    fuentes = {
        "movielens": SILVER_MOVIELENS_DIR,
        "tmdb": SILVER_TMDB_DIR,
        "watchmode": SILVER_WATCHMODE_DIR,
        "integrated": SILVER_INTEGRATED_DIR,
    }

    
    # 1. Construir manifiestos antes de publicar.
    manifests = {}
    errores_preflight = {}

    for source, local_dir in fuentes.items():
        try:
            manifests[source] = build_silver_manifest(
                local_dir=local_dir,
                source=source,
                run_id=run_id,
            )

        except Exception as error:
            errores_preflight[source] = str(error)

    # 2. Validar los manifiestos encontrados.
    validacion = validate_silver_manifests(
        manifests=manifests,
        run_id=run_id,
    )

    # 3. Bloquear toda la publicación si hay problemas.
    if errores_preflight or not validacion["autorizado"]:

        print(
            "PUBLICACIÓN SILVER BLOQUEADA: "
            "la validación previa no fue satisfactoria."
        )

        return {
            "run_id": run_id,
            "estado": "blocked",
            "preflight": validacion,
            "errores_preflight": errores_preflight,
            "evaluacion": None,
            "fuentes": {},
        }

    print(
        "Validación previa satisfactoria. "
        "Los archivos están listos para publicación."
    )

    resultados = {}

    for source, local_dir in fuentes.items():

        print(
            f"\nProcesando Silver/{source}"
        )

        try:
            resultados[source] = (
                upload_silver_directory(
                    local_dir=local_dir,
                    source=source,
                    run_id=run_id,
                )
            )

        except Exception as error:

            resultados[source] = {
                "source": source,
                "subidos": 0,
                "existentes": 0,
                "errores": [
                    {
                        "archivo": None,
                        "error": str(error),
                    }
                ],
            }

            print(
                f"{source} -> ERROR GENERAL"
            )

            print(
                "   Detalle:",
                error
            )

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    total_subidos = 0
    total_existentes = 0
    total_errores = 0

    for source, resultado in resultados.items():

        subidos = resultado["subidos"]
        existentes = resultado["existentes"]
        errores = len(
            resultado["errores"]
        )

        total_subidos += subidos
        total_existentes += existentes
        total_errores += errores

        print(
            source,
            "| subidos:",
            subidos,
            "| existentes:",
            existentes,
            "| errores:",
            errores,
        )

    print("-" * 60)

    print(
        "TOTAL",
        "| subidos:",
        total_subidos,
        "| existentes:",
        total_existentes,
        "| errores:",
        total_errores,
    )

    print("=" * 60)


    # Evaluar los resultados de las cuatro fuentes.
    evaluacion = evaluate_silver_publication(
        resultados
    )

    # Devolver un resultado estructurado.
    return {
        "run_id": run_id,
        "estado": evaluacion["estado"],
        "evaluacion": evaluacion,
        "fuentes": resultados,
    }



if __name__ == "__main__":
    upload_silver_to_supabase()