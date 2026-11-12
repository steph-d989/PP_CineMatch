import pandas as pd
import unittest
import hashlib
import json

from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.gold_runs import (
    build_gold_run_directories,
    validate_gold_run_paths,
    finalize_gold_run_paths,
    validate_gold_required_datasets,
    verify_gold_run_files,
    verify_gold_run_artifacts,
    build_gold_manifest,
    verify_gold_manifest,
    save_gold_manifest,
    verify_gold_manifest,
    seal_gold_run
)

from src.transformation.silver_to_gold import (
    run_silver_to_gold,
)

class GoldRunsTests(unittest.TestCase):

    
    def test_pipeline_gold_escribe_archivos_reales(self):

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        # Datos Silver mínimos para ejecutar
        # las transformaciones reales de Gold.
        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series(
                    [550], dtype="Int64"
                ),
                "canonical_title": ["Pelicula A"],
            }),
            "ratings": pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "rating": [4.5],
                "rated_at": pd.to_datetime(
                    ["2024-01-01"],
                    utc=True,
                ),
            }),
            "tags": pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "tag": ["Drama"],
            }),
            "availability": pd.DataFrame({
                "tmdb_id": [550],
                "source_id": [10],
                "provider_name": ["Plataforma A"],
                "region": ["PE"],
                "type": ["sub"],
                "format": ["HD"],
                "price": [0.0],
                "web_url": [None],
                "ios_url": [None],
                "android_url": [None],
            }),
        }

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            # Únicamente se simula la lectura
            # de Silver. No se simulan escrituras.
            with patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ) as mock_carga:

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            mock_carga.assert_called_once()

            # El orquestador debe haber creado
            # físicamente su directorio Gold.
            self.assertTrue(
                gold_dirs["root"].is_dir()
            )

            # Comprobar los nueve datasets.
            esperados = {
                "dim_user",
                "dim_movie",
                "dim_platform",
                "dim_date",
                "fact_rating",
                "fact_availability",
                "movie_metrics",
                "user_metrics",
                "availability_unlinked",
            }

            rutas = resultado["rutas"]

            self.assertTrue(
                esperados.issubset(rutas.keys())
            )

            # Cerrar la ejecución con los
            # archivos realmente generados.
            cierre = seal_gold_run(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=resultado["metadata"],
            )

            self.assertEqual(
                cierre["estado"],
                "sealed",
            )

            self.assertEqual(
                cierre["run_id"],
                run_id,
            )

            self.assertEqual(
                cierre["total_archivos"],
                10,
            )

            # Recuperar el manifiesto físico
            # y verificar nuevamente sus hashes.
            manifest_path = cierre["manifest_path"]

            self.assertTrue(
                manifest_path.is_file()
            )

            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as archivo:
                manifest = json.load(archivo)

            verificacion = verify_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifest,
            )

            self.assertEqual(
                verificacion["estado"],
                "manifest_verified",
            )

            # Comprobar que los resultados
            # no son solamente archivos vacíos.
            dim_movie = pd.read_parquet(
                rutas["dim_movie"]
            )

            self.assertEqual(
                len(dim_movie),
                1,
            )

            self.assertEqual(
                int(dim_movie.iloc[0]["movie_id"]),
                1,
            )

            particiones = rutas["fact_rating"]

            self.assertGreaterEqual(
                len(particiones),
                1,
            )

            rating = pd.concat(
                [
                    pd.read_parquet(ruta)
                    for ruta in particiones
                ],
                ignore_index=True,
            )

            self.assertEqual(
                len(rating),
                1,
            )

            self.assertEqual(
                int(rating.iloc[0]["user_id"]),
                10,
            )

            self.assertEqual(
                float(rating.iloc[0]["rating"]),
                4.5,
            )

    def test_gold_cierra_ejecucion_con_archivos_reales(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            rutas = {
                "dim_user": gold_dirs["dimensions"] / "dim_user.parquet",
                "dim_movie": gold_dirs["dimensions"] / "dim_movie.parquet",
                "dim_platform": gold_dirs["dimensions"] / "dim_platform.parquet",
                "dim_date": gold_dirs["dimensions"] / "dim_date.parquet",
                "fact_rating": [
                    gold_dirs["facts"] / "fact_rating"
                    / "part_000.parquet",
                ],
                "fact_availability": (
                    gold_dirs["facts"] / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"] / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"] / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"] / "availability_unlinked.parquet"
                ),
            }

            archivos = [
                archivo
                for referencia in rutas.values()
                for archivo in (
                    referencia
                    if isinstance(referencia, list)
                    else [referencia]
                )
            ]

            # Crear nueve archivos Parquet físicos.
            for ruta in archivos:
                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pd.DataFrame({
                    "id": [1],
                }).to_parquet(
                    ruta,
                    index=False,
                )

            metadata = (
                gold_dirs["metrics"] / "_metadata.json"
            )

            metadata.write_text(
                json.dumps({"capa": "gold"}),
                encoding="utf-8",
            )

            # Cerrar la ejecución.
            resultado = seal_gold_run(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=metadata,
            )

            self.assertEqual(
                resultado["estado"],
                "sealed",
            )

            self.assertEqual(
                resultado["total_archivos"],
                10,
            )

            manifest_path = resultado["manifest_path"]

            self.assertTrue(
                manifest_path.is_file()
            )

            # El manifiesto puede verificarse
            # después de recuperarlo del disco.
            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as archivo:
                manifest = json.load(archivo)

            verificacion = verify_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifest,
            )

            self.assertEqual(
                verificacion["estado"],
                "manifest_verified",
            )

            # No se permite cerrar nuevamente
            # la misma ejecución.
            with self.assertRaises(FileExistsError):
                seal_gold_run(
                    gold_dirs=gold_dirs,
                    rutas=rutas,
                    metadata_path=metadata,
                )

            # Una modificación posterior debe
            # invalidar el manifiesto guardado.
            rutas["dim_movie"].write_bytes(
                b"contenido modificado"
            )

            with self.assertRaises(ValueError):
                verify_gold_manifest(
                    gold_dirs=gold_dirs,
                    manifest=manifest,
                )

    def test_gold_manifiesto_atomico_sin_temporales(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as directorio:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(directorio) / "gold",
                run_id=run_id,
            )

            archivo = (
                gold_dirs["dimensions"]
                / "dim_movie.parquet"
            )

            archivo.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            contenido = b"datos de prueba"
            archivo.write_bytes(contenido)

            manifest = {
                "run_id": run_id,
                "algoritmo": "sha256",
                "total_archivos": 1,
                "archivos": [
                    {
                        "dataset": "dim_movie",
                        "path": "dimensions/dim_movie.parquet",
                        "bytes": len(contenido),
                        "sha256": hashlib.sha256(
                            contenido
                        ).hexdigest(),
                    }
                ],
            }

            # Primera publicación.
            destino = save_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifest,
            )

            self.assertTrue(destino.is_file())

            with destino.open(
                "r", encoding="utf-8"
            ) as origen:
                recuperado = json.load(origen)

            self.assertEqual(recuperado, manifest)

            resultado = verify_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=recuperado,
            )

            self.assertEqual(
                resultado["estado"],
                "manifest_verified",
            )

            # La segunda publicación no puede
            # sobrescribir el manifiesto.
            with self.assertRaises(FileExistsError):
                save_gold_manifest(
                    gold_dirs=gold_dirs,
                    manifest=manifest,
                )

            # El contenido original permanece.
            with destino.open(
                "r", encoding="utf-8"
            ) as origen:
                self.assertEqual(
                    json.load(origen),
                    manifest,
                )

            # No deben quedar temporales.
            temporales = list(
                gold_dirs["root"].glob(
                    "._manifest_*.tmp"
                )
            )

            self.assertEqual(temporales, [])

    def test_gold_guarda_manifiesto_json_sin_sobrescribir(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            archivo = (
                gold_dirs["dimensions"]
                / "dim_movie.parquet"
            )

            archivo.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            contenido = b"contenido de prueba"
            archivo.write_bytes(contenido)

            manifiesto = {
                "run_id": run_id,
                "estado": "manifest_built",
                "algoritmo": "sha256",
                "total_archivos": 1,
                "archivos": [
                    {
                        "dataset": "dim_movie",
                        "path": (
                            "dimensions/dim_movie.parquet"
                        ),
                        "bytes": len(contenido),
                        "sha256": hashlib.sha256(
                            contenido
                        ).hexdigest(),
                    }
                ],
            }

            # Guardar el manifiesto.
            ruta_manifest = save_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifiesto,
            )

            self.assertEqual(
                ruta_manifest,
                gold_dirs["root"] / "_manifest.json",
            )

            self.assertTrue(
                ruta_manifest.is_file()
            )

            # Recuperar el contenido desde JSON.
            with ruta_manifest.open(
                "r",
                encoding="utf-8",
            ) as origen:
                recuperado = json.load(origen)

            self.assertEqual(
                recuperado,
                manifiesto,
            )

            # Verificar los hashes del manifiesto
            # recuperado desde el disco.
            resultado = verify_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=recuperado,
            )

            self.assertEqual(
                resultado["estado"],
                "manifest_verified",
            )

            # Impedir que se sobrescriba.
            with self.assertRaises(FileExistsError):
                save_gold_manifest(
                    gold_dirs=gold_dirs,
                    manifest=manifiesto,
                )

            # Confirmar que el JSON original
            # permanece intacto.
            with ruta_manifest.open(
                "r",
                encoding="utf-8",
            ) as origen:
                self.assertEqual(
                    json.load(origen),
                    manifiesto,
                )

    def test_gold_manifest_detecta_modificaciones(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            archivo = (
                gold_dirs["dimensions"]
                / "dim_movie.parquet"
            )

            archivo.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            contenido_original = b"abc"
            archivo.write_bytes(contenido_original)

            manifiesto = {
                "run_id": run_id,
                "algoritmo": "sha256",
                "total_archivos": 1,
                "archivos": [
                    {
                        "dataset": "dim_movie",
                        "path": "dimensions/dim_movie.parquet",
                        "bytes": len(contenido_original),
                        "sha256": hashlib.sha256(
                            contenido_original
                        ).hexdigest(),
                    }
                ],
            }

            # Archivo sin modificaciones.
            resultado = verify_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifiesto,
            )

            self.assertEqual(
                resultado["estado"],
                "manifest_verified",
            )

            # Alterar el contenido sin cambiar
            # el tamaño del archivo.
            archivo.write_bytes(b"xyz")

            with self.assertRaisesRegex(
                ValueError,
                "diferencia SHA-256",
            ):
                verify_gold_manifest(
                    gold_dirs=gold_dirs,
                    manifest=manifiesto,
                )

            # Restaurar y después eliminar el archivo.
            archivo.write_bytes(contenido_original)
            archivo.unlink()

            with self.assertRaises(
                FileNotFoundError
            ):
                verify_gold_manifest(
                    gold_dirs=gold_dirs,
                    manifest=manifiesto,
                )

            # Impedir rutas que salgan del run_id.
            manifiesto["archivos"][0]["path"] = (
                "../archivo_externo.parquet"
            )

            with self.assertRaisesRegex(
                ValueError,
                "ruta externa",
            ):
                verify_gold_manifest(
                    gold_dirs=gold_dirs,
                    manifest=manifiesto,
                )

    def test_gold_manifest_registra_sha256_archivos(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            rutas = {
                "dim_user": gold_dirs["dimensions"] / "dim_user.parquet",
                "dim_movie": gold_dirs["dimensions"] / "dim_movie.parquet",
                "dim_platform": gold_dirs["dimensions"] / "dim_platform.parquet",
                "dim_date": gold_dirs["dimensions"] / "dim_date.parquet",
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
                "fact_availability": (
                    gold_dirs["facts"] / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"] / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"] / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"] / "availability_unlinked.parquet"
                ),
            }

            metadata = (
                gold_dirs["metrics"] / "_metadata.json"
            )

            archivos = [
                ruta
                for referencia in rutas.values()
                for ruta in (
                    referencia
                    if isinstance(referencia, list)
                    else [referencia]
                )
            ]

            for ruta in archivos:

                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pd.DataFrame({
                    "id": [1],
                }).to_parquet(
                    ruta,
                    index=False,
                )

            metadata.write_text(
                json.dumps({"capa": "gold"}),
                encoding="utf-8",
            )

            manifiesto = build_gold_manifest(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=metadata,
            )

            self.assertEqual(
                manifiesto["estado"],
                "manifest_built",
            )

            self.assertEqual(
                manifiesto["run_id"],
                run_id,
            )

            self.assertEqual(
                manifiesto["total_archivos"],
                10,
            )

            entradas = {
                item["path"]: item
                for item in manifiesto["archivos"]
            }

            self.assertIn(
                "dimensions/dim_movie.parquet",
                entradas,
            )

            self.assertIn(
                "metrics/_metadata.json",
                entradas,
            )

            # Verificar que el hash almacenado
            # corresponde al contenido real.
            ruta_dim = rutas["dim_movie"]

            hash_esperado = hashlib.sha256(
                ruta_dim.read_bytes()
            ).hexdigest()

            self.assertEqual(
                entradas["dimensions/dim_movie.parquet"]["sha256"],
                hash_esperado,
            )

            self.assertTrue(
                all(
                    len(item["sha256"]) == 64
                    for item in manifiesto["archivos"]
                )
            )

            # El manifiesto debe ser serializable a JSON.
            json.dumps(manifiesto)

            # Todavía no se escribe un manifiesto.
            self.assertFalse(
                (gold_dirs["root"] / "_manifest.json").exists()
            )

    def test_gold_verifica_artefactos_integrados(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            rutas = {
                "dim_user": gold_dirs["dimensions"] / "dim_user.parquet",
                "dim_movie": gold_dirs["dimensions"] / "dim_movie.parquet",
                "dim_platform": gold_dirs["dimensions"] / "dim_platform.parquet",
                "dim_date": gold_dirs["dimensions"] / "dim_date.parquet",
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
                "fact_availability": (
                    gold_dirs["facts"] / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"] / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"] / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"] / "availability_unlinked.parquet"
                ),
            }

            metadata = (
                gold_dirs["metrics"] / "_metadata.json"
            )

            archivos = [
                ruta
                for referencia in rutas.values()
                for ruta in (
                    referencia
                    if isinstance(referencia, list)
                    else [referencia]
                )
            ]

            # Crear Parquet físicos pequeños.
            for ruta in archivos:

                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pd.DataFrame({
                    "id": [1],
                }).to_parquet(
                    ruta,
                    index=False,
                )

            metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata.write_text(
                json.dumps({"capa": "gold"}),
                encoding="utf-8",
            )

            # Escenario correcto: diez archivos
            # existentes y formatos legibles.
            resultado = verify_gold_run_artifacts(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=metadata,
            )

            self.assertEqual(
                resultado["estado"],
                "artifacts_verified",
            )

            self.assertEqual(
                resultado["total_archivos"],
                10,
            )

            self.assertTrue(
                resultado["formatos_validos"]
            )

            # Un archivo corrupto debe impedir
            # la verificación integral.
            rutas["dim_movie"].write_bytes(
                b"parquet corrupto"
            )

            with self.assertRaisesRegex(
                ValueError,
                "Parquet inválido",
            ):
                verify_gold_run_artifacts(
                    gold_dirs=gold_dirs,
                    rutas=rutas,
                    metadata_path=metadata,
                )

    def test_gold_exige_datasets_obligatorios(self):

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            rutas = {
                "dim_user": raiz / "dim_user.parquet",
                "dim_movie": raiz / "dim_movie.parquet",
                "dim_platform": raiz / "dim_platform.parquet",
                "dim_date": raiz / "dim_date.parquet",
                "fact_rating": [
                    raiz / "part_000.parquet",
                ],
                "fact_availability": (
                    raiz / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    raiz / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    raiz / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    raiz / "availability_unlinked.parquet"
                ),
            }

            # Todos los datasets están declarados.
            validate_gold_required_datasets(rutas)

            # Falta fact_availability.
            incompletas = rutas.copy()
            del incompletas["fact_availability"]

            with self.assertRaisesRegex(
                ValueError,
                "datasets obligatorios faltantes",
            ):
                validate_gold_required_datasets(
                    incompletas
                )

            # Ratings sin particiones.
            sin_particiones = rutas.copy()
            sin_particiones["fact_rating"] = []

            with self.assertRaisesRegex(
                ValueError,
                "no contiene archivos para fact_rating",
            ):
                validate_gold_required_datasets(
                    sin_particiones
                )

            # La validación no genera archivos.
            self.assertFalse(
                (raiz / "dim_user.parquet").exists()
            )

    def test_gold_prepara_directorios_mismo_run_id(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_root = Path(temporal) / "gold"

            directorios = build_gold_run_directories(
                gold_root=gold_root,
                run_id=run_id,
            )

            raiz_esperada = (
                gold_root / "runs" / f"run_id={run_id}"
            )

            self.assertEqual(
                directorios["root"],
                raiz_esperada,
            )

            self.assertEqual(
                directorios["dimensions"],
                raiz_esperada / "dimensions",
            )

            self.assertEqual(
                directorios["facts"],
                raiz_esperada / "facts",
            )

            self.assertEqual(
                directorios["metrics"],
                raiz_esperada / "metrics",
            )

            self.assertEqual(
                directorios["ml_content"],
                raiz_esperada / "ml" / "content",
            )

            self.assertEqual(
                directorios["ml_collaborative"],
                raiz_esperada / "ml" / "collaborative",
            )

            self.assertEqual(
                directorios["ml_hybrid"],
                raiz_esperada / "ml" / "hybrid",
            )

            self.assertEqual(
                directorios["quality"],
                raiz_esperada / "quality",
            )

            # Preparar rutas no debe crear archivos.
            self.assertFalse(
                raiz_esperada.exists()
            )

            # No aceptar un run_id inválido.
            with self.assertRaises(ValueError):
                build_gold_run_directories(
                    gold_root=gold_root,
                    run_id="identificador-invalido",
                )

    def test_gold_rechaza_archivos_fuera_del_run_id(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        otro_run_id = (
            "87654321-4321-4321-8321-cba987654321"
        )

        with TemporaryDirectory() as temporal:

            gold_root = Path(temporal) / "gold"

            directorios = build_gold_run_directories(
                gold_root=gold_root,
                run_id=run_id,
            )

            otros_directorios = build_gold_run_directories(
                gold_root=gold_root,
                run_id=otro_run_id,
            )

            rutas_validas = {
                "dim_movie": (
                    directorios["dimensions"]
                    / "dim_movie.parquet"
                ),
                "fact_rating": [
                    directorios["facts"]
                    / "fact_rating"
                    / "part_000.parquet",
                    directorios["facts"]
                    / "fact_rating"
                    / "part_001.parquet",
                ],
            }

            metadata = (
                directorios["metrics"] / "_metadata.json"
            )

            # Caso correcto: todos los archivos
            # pertenecen a la misma ejecución.
            validate_gold_run_paths(
                gold_dirs=directorios,
                rutas=rutas_validas,
                metadata_path=metadata,
            )

            # Caso incorrecto: fact_rating pertenece
            # a una ejecución diferente.
            rutas_incorrectas = {
                "dim_movie": rutas_validas["dim_movie"],
                "fact_rating": [
                    otros_directorios["facts"]
                    / "fact_rating"
                    / "part_000.parquet",
                ],
            }

            with self.assertRaisesRegex(
                ValueError,
                "fuera del run_id seleccionado",
            ):
                validate_gold_run_paths(
                    gold_dirs=directorios,
                    rutas=rutas_incorrectas,
                    metadata_path=metadata,
                )

            # Caso incorrecto: metadata externa.
            with self.assertRaisesRegex(
                ValueError,
                "fuera del run_id seleccionado",
            ):
                validate_gold_run_paths(
                    gold_dirs=directorios,
                    rutas=rutas_validas,
                    metadata_path=(
                        Path(temporal) / "metadata_externa.json"
                    ),
                )

            # La validación no crea carpetas.
            self.assertFalse(
                directorios["root"].exists()
            )

    def test_gold_finaliza_rutas_solo_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        otro_run_id = (
            "87654321-4321-4321-8321-cba987654321"
        )

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal) / "gold"

            gold_dirs = build_gold_run_directories(
                gold_root=raiz,
                run_id=run_id,
            )

            otros_dirs = build_gold_run_directories(
                gold_root=raiz,
                run_id=otro_run_id,
            )

            rutas = {
                "dim_user": (
                    gold_dirs["dimensions"]
                    / "dim_user.parquet"
                ),
                "dim_movie": (
                    gold_dirs["dimensions"]
                    / "dim_movie.parquet"
                ),
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet",
                ],
                "fact_availability": (
                    gold_dirs["facts"]
                    / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"]
                    / "movie_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"]
                    / "availability_unlinked.parquet"
                ),
            }

            metadata = (
                gold_dirs["metrics"]
                / "_metadata.json"
            )

            # Escenario correcto.
            resultado = finalize_gold_run_paths(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=metadata,
            )

            self.assertEqual(
                resultado["estado"],
                "paths_verified",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertEqual(
                set(resultado["datasets"]),
                set(rutas),
            )

            # Introducir una partición de otra ejecución.
            rutas_incorrectas = {
                **rutas,
                "fact_rating": [
                    otros_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet",
                ],
            }

            with self.assertRaisesRegex(
                ValueError,
                "fuera del run_id seleccionado",
            ):
                finalize_gold_run_paths(
                    gold_dirs=gold_dirs,
                    rutas=rutas_incorrectas,
                    metadata_path=metadata,
                )

            # La función no debe crear archivos.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_gold_verifica_archivos_fisicos(self):

        from src.utils.gold_runs import (
            verify_gold_run_files,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            rutas = {
                "dim_user": gold_dirs["dimensions"] / "dim_user.parquet",
                "dim_movie": gold_dirs["dimensions"] / "dim_movie.parquet",
                "dim_platform": gold_dirs["dimensions"] / "dim_platform.parquet",
                "dim_date": gold_dirs["dimensions"] / "dim_date.parquet",
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
                "fact_availability": (
                    gold_dirs["facts"] / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"] / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"] / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"] / "availability_unlinked.parquet"
                ),
            }

            metadata = gold_dirs["metrics"] / "_metadata.json"

            archivos = [
                referencia
                for valor in rutas.values()
                for referencia in (
                    valor if isinstance(valor, list) else [valor]
                )
            ]

            # Crear los directorios, pero no los archivos.
            for ruta in [*archivos, metadata]:
                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # Los datasets están declarados,
            # pero todavía falta un archivo.
            with self.assertRaisesRegex(
                FileNotFoundError,
                "archivo obligatorio inexistente",
            ):
                verify_gold_run_files(
                    gold_dirs=gold_dirs,
                    rutas=rutas,
                    metadata_path=metadata,
                )

            # Crear todos los archivos con contenido
            # mínimo para probar existencia y tamaño.
            # Esta prueba NO evalúa el formato Parquet.
            for ruta in [*archivos, metadata]:
                ruta.write_bytes(b"contenido de prueba")

            # Vaciar deliberadamente una partición.
            particion = rutas["fact_rating"][0]
            particion.write_bytes(b"")

            with self.assertRaisesRegex(
                ValueError,
                "archivo vacío",
            ):
                verify_gold_run_files(
                    gold_dirs=gold_dirs,
                    rutas=rutas,
                    metadata_path=metadata,
                )

            # Restaurar la partición.
            particion.write_bytes(b"contenido de prueba")

            resultado = verify_gold_run_files(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=metadata,
            )

            self.assertEqual(
                resultado["estado"],
                "files_verified",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            # Nueve datasets y un archivo de metadatos.
            self.assertEqual(
                resultado["total_archivos"],
                10,
            )

            self.assertTrue(
                all(
                    archivo["bytes"] > 0
                    for archivo in resultado["archivos"]
                )
            )

    def test_gold_verificacion_fisica_exige_datasets_completos(self):

        from src.utils.gold_runs import (
            verify_gold_run_files,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            rutas_incompletas = {
                "dim_movie": (
                    gold_dirs["dimensions"]
                    / "dim_movie.parquet"
                ),
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
            }

            metadata = (
                gold_dirs["metrics"]
                / "_metadata.json"
            )

            with self.assertRaisesRegex(
                ValueError,
                "datasets obligatorios faltantes",
            ):
                verify_gold_run_files(
                    gold_dirs=gold_dirs,
                    rutas=rutas_incompletas,
                    metadata_path=metadata,
                )

            # No debe crear directorios ni archivos.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    
    def test_gold_rechaza_formatos_invalidos(self):

        import json
        import pandas as pd

        from src.utils.gold_runs import (
            validate_gold_file_formats,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            rutas = {
                "dim_user": gold_dirs["dimensions"] / "dim_user.parquet",
                "dim_movie": gold_dirs["dimensions"] / "dim_movie.parquet",
                "dim_platform": gold_dirs["dimensions"] / "dim_platform.parquet",
                "dim_date": gold_dirs["dimensions"] / "dim_date.parquet",
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
                "fact_availability": (
                    gold_dirs["facts"] / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"] / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"] / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"] / "availability_unlinked.parquet"
                ),
            }

            metadata = gold_dirs["metrics"] / "_metadata.json"

            archivos = [
                ruta
                for referencia in rutas.values()
                for ruta in (
                    referencia
                    if isinstance(referencia, list)
                    else [referencia]
                )
            ]

            # Crear Parquet reales y pequeños.
            for ruta in archivos:

                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pd.DataFrame({
                    "id": [1],
                }).to_parquet(
                    ruta,
                    index=False,
                )

            metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata.write_text(
                json.dumps({"capa": "gold"}),
                encoding="utf-8",
            )

            # Escenario correcto.
            validate_gold_file_formats(
                rutas=rutas,
                metadata_path=metadata,
            )

            # Sustituir una dimensión por bytes
            # que no representan un Parquet válido.
            ruta_dim = rutas["dim_movie"]

            ruta_dim.write_bytes(
                b"archivo corrupto"
            )

            with self.assertRaisesRegex(
                ValueError,
                "Parquet inválido",
            ):
                validate_gold_file_formats(
                    rutas=rutas,
                    metadata_path=metadata,
                )

            # Restaurar el Parquet.
            pd.DataFrame({
                "id": [1],
            }).to_parquet(
                ruta_dim,
                index=False,
            )

            # Corromper los metadatos.
            metadata.write_text(
                "{json incompleto",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "JSON inválidos",
            ):
                validate_gold_file_formats(
                    rutas=rutas,
                    metadata_path=metadata,
                )


if __name__ == "__main__":
    unittest.main()
