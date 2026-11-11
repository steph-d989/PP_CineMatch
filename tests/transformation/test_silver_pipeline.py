
import unittest
import json 
import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, Mock

from src.transformation.silver_pipeline import (
    prepare_silver_run,
    validate_silver_run,
    execute_silver_run,
    reverify_silver_run,
    build_silver_transformers,
    SILVER_EXECUTION_ORDER,
)

from src.load.silver_storage import (
    build_silver_manifest,
    verify_silver_local_manifest,
)

from src.transformation.silver_integration import (
    run_silver_integration,
)


from src.transformation.bronze_to_silver import (
    run_movielens_bronze_to_silver,
    run_tmdb_bronze_to_silver,
    run_watchmode_bronze_to_silver,
)

class SilverPipelineTests(unittest.TestCase):
    
    
    def test_pipeline_silver_integracion_local_completa(self):

        run_id = "12345678-1234-4234-8234-123456789abc"

        # Tres películas con estados diferentes:
        # 1: disponible
        # 2: consultada sin disponibilidad
        # 3: disponibilidad desconocida

        bronze_movielens = {
            "movies": pd.DataFrame({
                "movieId": [1, 2, 3],
                "title": [
                    "Titulo MovieLens A",
                    "Titulo MovieLens B",
                    "Titulo MovieLens C",
                ],
                "genres": [
                    "Drama",
                    "Comedy",
                    "Action",
                ],
            }),
            "links": pd.DataFrame({
                "movieId": [1, 2, 3],
                "imdbId": [
                    "0000550",
                    "0000600",
                    pd.NA,
                ],
                "tmdbId": [550, 600, None],
            }),
            "ratings": pd.DataFrame({
                "userId": [10, 11, 12],
                "movieId": [1, 2, 3],
                "rating": [4.5, 3.0, 5.0],
                "timestamp": [
                    1700000000,
                    1700000100,
                    1700000200,
                ],
            }),
            "tags": pd.DataFrame({
                "userId": [10],
                "movieId": [1],
                "tag": ["interesante"],
                "timestamp": [1700000000],
            }),
        }

        bronze_tmdb = [
            {
                "id": 550,
                "imdb_id": "tt0000550",
                "title": "Titulo TMDb A",
                "genres": [{"name": "Drama"}],
            },
            {
                "id": 600,
                "imdb_id": "tt0000600",
                "title": "Titulo TMDb B",
                "genres": [{"name": "Comedy"}],
            },
        ]

        bronze_watchmode = [
            {
                "tmdb_id": 550,
                "sources": [
                    {
                        "source_id": 1,
                        "name": "Plataforma A",
                        "type": "sub",
                        "region": "PE",
                    },
                ],
            },
            {
                "tmdb_id": 600,
                "sources": [],
            },
        ]

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # Solo simulamos las entradas Bronze.
            # Las transformaciones, guardados
            # e integración son reales.
            with (
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_movielens_bronze",
                    return_value=bronze_movielens,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_tmdb_bronze",
                    return_value=bronze_tmdb,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_watchmode_bronze",
                    return_value=bronze_watchmode,
                ),
            ):

                ejecucion = execute_silver_run(
                    plan=plan,
                    transformers=build_silver_transformers(),
                )

            self.assertEqual(
                ejecucion["estado"],
                "transformed",
                msg=str(ejecucion.get("error")),
            )

            self.assertEqual(
                ejecucion["fuentes_completadas"],
                plan["orden"],
            )

            # Comprobar los archivos generados.
            dirs = plan["directorios"]

            ruta_integrated = (
                dirs["integrated"]
                / "movies_integrated.parquet"
            )

            self.assertTrue(ruta_integrated.is_file())

            for fuente in plan["orden"]:
                self.assertTrue(
                    (dirs[fuente] / "_metadata.json").is_file()
                )

            # Leer el catálogo integrado REAL.
            peliculas = pd.read_parquet(
                ruta_integrated
            ).set_index("movie_id")

            self.assertEqual(
                len(peliculas),
                3,
            )

            # Película 1: disponible.
            self.assertEqual(
                peliculas.loc[1, "canonical_title"],
                "Titulo TMDb A",
            )

            self.assertTrue(
                bool(peliculas.loc[1, "has_availability"])
            )

            self.assertTrue(
                bool(peliculas.loc[1, "watchmode_checked"])
            )

            # Película 2: consultada sin plataformas.
            self.assertFalse(
                bool(peliculas.loc[2, "has_availability"])
            )

            self.assertTrue(
                bool(peliculas.loc[2, "watchmode_checked"])
            )

            self.assertEqual(
                int(peliculas.loc[2, "sources_count"]),
                0,
            )

            # Película 3: sin identificador TMDb.
            # Conserva el título MovieLens.
            self.assertEqual(
                peliculas.loc[3, "canonical_title"],
                "Titulo MovieLens C",
            )

            self.assertTrue(
                pd.isna(peliculas.loc[3, "has_availability"])
            )

            self.assertFalse(
                bool(peliculas.loc[3, "watchmode_checked"])
            )

            # Validación real de manifiestos.
            validacion = validate_silver_run(
                plan=plan,
                execution_result=ejecucion,
            )

            self.assertEqual(
                validacion["estado"],
                "validated",
                msg=str(validacion),
            )

            # Verificación real SHA-256.
            reverificacion = reverify_silver_run(
                plan=plan,
                validation_result=validacion,
            )

            self.assertEqual(
                reverificacion["estado"],
                "locally_verified",
                msg=str(reverificacion),
            )

    
    def test_orquestador_rechaza_fuentes_incompletas(self):

        with TemporaryDirectory() as temporal:

            run_id = (
                "12345678-1234-4234-8234-123456789abc"
            )

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # Aunque se declare "transformed",
            # Integrated no figura como completada.
            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": (
                    "movielens",
                    "tmdb",
                    "watchmode",
                ),
            }

            # No debe intentar construir manifiestos.
            with patch(
                "src.load.silver_storage."
                "build_silver_manifest"
            ) as mock_manifest:

                resultado = validate_silver_run(
                    plan=plan,
                    execution_result=execution_result,
                )

            self.assertEqual(
                resultado["estado"],
                "blocked",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertIn(
                "no completó",
                resultado["error"],
            )

            mock_manifest.assert_not_called()

            # No se crean archivos.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )


    
    def test_orquestador_conecta_transformaciones_reales(self):

        transformers = build_silver_transformers()

        # Deben existir exactamente cuatro fuentes.
        self.assertEqual(
            tuple(transformers.keys()),
            SILVER_EXECUTION_ORDER,
        )

        # Comprobar que se conectaron las funciones
        # reales y no otras implementaciones.
        self.assertIs(
            transformers["movielens"],
            run_movielens_bronze_to_silver,
        )

        self.assertIs(
            transformers["tmdb"],
            run_tmdb_bronze_to_silver,
        )

        self.assertIs(
            transformers["watchmode"],
            run_watchmode_bronze_to_silver,
        )

        self.assertIs(
            transformers["integrated"],
            run_silver_integration,
        )

        # Todas las entradas deben ser ejecutables.
        for fuente in SILVER_EXECUTION_ORDER:
            self.assertTrue(
                callable(transformers[fuente])
            )

    
    def test_orquestador_bloquea_tmdb_modificado_tras_validacion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            dirs = plan["directorios"]

            # Crear las cuatro carpetas.
            for fuente in plan["orden"]:
                dirs[fuente].mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # MovieLens: archivos obligatorios.
            movielens = dirs["movielens"]

            for nombre in (
                "movies.parquet",
                "links.parquet",
                "tags.parquet",
            ):
                pd.DataFrame({
                    "dato": [1],
                }).to_parquet(
                    movielens / nombre,
                    index=False,
                )

            ratings_dir = movielens / "ratings"

            ratings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "rating": [4.5],
            }).to_parquet(
                ratings_dir / "part_000.parquet",
                index=False,
            )

            # TMDb: archivo original.
            ruta_tmdb = (
                dirs["tmdb"] / "movies_tmdb.parquet"
            )

            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula Original"],
            }).to_parquet(
                ruta_tmdb,
                index=False,
            )

            # Watchmode: archivos obligatorios.
            pd.DataFrame({
                "tmdb_id": [550],
            }).to_parquet(
                dirs["watchmode"] / "availability.parquet",
                index=False,
            )

            pd.DataFrame({
                "tmdb_id": [550],
                "has_availability": [True],
            }).to_parquet(
                dirs["watchmode"]
                / "availability_status.parquet",
                index=False,
            )

            # Integrated: archivo obligatorio.
            pd.DataFrame({
                "movie_id": [1],
            }).to_parquet(
                dirs["integrated"]
                / "movies_integrated.parquet",
                index=False,
            )

            # Metadatos para las cuatro fuentes.
            for fuente in plan["orden"]:

                (dirs[fuente] / "_metadata.json").write_text(
                    json.dumps({
                        "fuente": fuente,
                        "run_id": run_id,
                        "estado": "success",
                    }),
                    encoding="utf-8",
                )

            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": plan["orden"],
            }

            # Primera validación: todo está correcto.
            validacion = validate_silver_run(
                plan=plan,
                execution_result=execution_result,
            )

            self.assertEqual(
                validacion["estado"],
                "validated",
                msg=str(validacion),
            )

            # Modificar TMDb después de la validación.
            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula Modificada"],
            }).to_parquet(
                ruta_tmdb,
                index=False,
            )

            # Reverificar con los manifiestos originales.
            resultado = reverify_silver_run(
                plan=plan,
                validation_result=validacion,
            )

            self.assertEqual(
                resultado["estado"],
                "blocked",
                msg=str(resultado),
            )

            self.assertEqual(
                resultado["fuente_fallida"],
                "tmdb",
            )

            # MovieLens estaba intacto.
            self.assertEqual(
                resultado["verificaciones"]
                ["movielens"]["estado"],
                "verified",
            )

            # TMDb fue rechazado.
            self.assertEqual(
                resultado["verificaciones"]
                ["tmdb"]["estado"],
                "failed",
            )

            # No continuar con las fuentes posteriores.
            self.assertNotIn(
                "watchmode",
                resultado["verificaciones"],
            )

            self.assertNotIn(
                "integrated",
                resultado["verificaciones"],
            )

    
    def test_orquestador_reverifica_cuatro_manifiestos_reales(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            dirs = plan["directorios"]

            for fuente in plan["orden"]:
                dirs[fuente].mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # Archivos obligatorios de MovieLens.
            movielens = dirs["movielens"]

            for nombre in (
                "movies.parquet",
                "links.parquet",
                "tags.parquet",
            ):
                pd.DataFrame({
                    "dato": [1],
                }).to_parquet(
                    movielens / nombre,
                    index=False,
                )

            ratings_dir = movielens / "ratings"
            ratings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "rating": [4.5],
            }).to_parquet(
                ratings_dir / "part_000.parquet",
                index=False,
            )

            # Archivo obligatorio de TMDb.
            pd.DataFrame({
                "tmdb_id": [550],
            }).to_parquet(
                dirs["tmdb"] / "movies_tmdb.parquet",
                index=False,
            )

            # Archivos obligatorios de Watchmode.
            pd.DataFrame({
                "tmdb_id": [550],
            }).to_parquet(
                dirs["watchmode"] / "availability.parquet",
                index=False,
            )

            pd.DataFrame({
                "tmdb_id": [550],
                "has_availability": [True],
            }).to_parquet(
                dirs["watchmode"]
                / "availability_status.parquet",
                index=False,
            )

            # Archivo obligatorio de Integrated.
            pd.DataFrame({
                "movie_id": [1],
            }).to_parquet(
                dirs["integrated"]
                / "movies_integrated.parquet",
                index=False,
            )

            # Metadatos de las cuatro fuentes.
            for fuente in plan["orden"]:

                (dirs[fuente] / "_metadata.json").write_text(
                    json.dumps({
                        "fuente": fuente,
                        "run_id": run_id,
                        "estado": "success",
                    }),
                    encoding="utf-8",
                )

            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": plan["orden"],
            }

            # Primera validación real.
            validacion = validate_silver_run(
                plan=plan,
                execution_result=execution_result,
            )

            self.assertEqual(
                validacion["estado"],
                "validated",
                msg=str(validacion),
            )

            # Segunda verificación real con
            # los mismos manifiestos.
            reverificacion = reverify_silver_run(
                plan=plan,
                validation_result=validacion,
            )

            self.assertEqual(
                reverificacion["estado"],
                "locally_verified",
                msg=str(reverificacion),
            )

            self.assertEqual(
                set(reverificacion["verificaciones"]),
                set(plan["orden"]),
            )

            for fuente in plan["orden"]:

                self.assertEqual(
                    reverificacion["verificaciones"]
                    [fuente]["estado"],
                    "verified",
                )

            # Se conservan los manifiestos originales.
            self.assertIs(
                reverificacion["manifests"],
                validacion["manifests"],
            )

    
    def test_orquestador_bloquea_hash_modificado_antes_publicacion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # Manifiestos simulados que representan
            # una validación anterior.
            manifests = {
                fuente: {
                    "source": fuente,
                    "run_id": run_id,
                }
                for fuente in plan["orden"]
            }

            validation_result = {
                "run_id": run_id,
                "estado": "validated",
                "manifests": manifests,
            }

            llamadas = []

            def simular_verificacion(
                manifest,
                local_dir,
            ):
                fuente = manifest["source"]
                llamadas.append(fuente)

                if fuente == "tmdb":
                    return {
                        "estado": "failed",
                        "error": "Hash SHA-256 diferente",
                    }

                return {
                    "estado": "verified",
                }


            # Utilizar el mismo mock en las dos posibles
            # referencias de la función de verificación.
            mock_verificacion = Mock(
                side_effect=simular_verificacion
            )

            with (
                patch(
                    "src.load.silver_storage."
                    "verify_silver_local_manifest",
                    mock_verificacion,
                ),
                patch(
                    "src.transformation.silver_pipeline."
                    "verify_silver_local_manifest",
                    mock_verificacion,
                    create=True,
                ),
            ):

                resultado = reverify_silver_run(
                    plan=plan,
                    validation_result=validation_result,
                )

            self.assertEqual(
                resultado["estado"],
                "blocked",
                msg=str(resultado),
            )

            self.assertEqual(
                resultado["fuente_fallida"],
                "tmdb",
            )

            self.assertEqual(
                llamadas,
                ["movielens", "tmdb"],
            )

            self.assertEqual(
                mock_verificacion.call_count,
                2,
            )

            self.assertNotIn(
                "watchmode",
                llamadas,
            )

            self.assertNotIn(
                "integrated",
                llamadas,
            )

            self.assertFalse(
                plan["directorios"]["root"].exists()
            )


    
    def test_silver_rechaza_hash_local_modificado(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            tmdb_dir = plan["directorios"]["tmdb"]

            tmdb_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            ruta_parquet = (
                tmdb_dir / "movies_tmdb.parquet"
            )

            ruta_metadata = tmdb_dir / "_metadata.json"

            # Crear el archivo original.
            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula Original"],
            }).to_parquet(
                ruta_parquet,
                index=False,
            )

            ruta_metadata.write_text(
                json.dumps({
                    "fuente": "tmdb",
                    "run_id": run_id,
                    "estado": "success",
                }),
                encoding="utf-8",
            )

            # Construir la referencia original.
            manifest = build_silver_manifest(
                local_dir=tmdb_dir,
                source="tmdb",
                run_id=run_id,
            )

            # Verificar el archivo sin modificaciones.
            original = verify_silver_local_manifest(
                manifest=manifest,
                local_dir=tmdb_dir,
            )

            # Sobrescribir el archivo con datos distintos.
            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula Modificada"],
            }).to_parquet(
                ruta_parquet,
                index=False,
            )

            # Verificar contra el manifiesto original.
            modificado = verify_silver_local_manifest(
                manifest=manifest,
                local_dir=tmdb_dir,
            )

            # El resultado original debe ser exitoso.
            self.assertEqual(
                original["estado"],
                "verified",
                msg=str(original),
            )

            # El archivo modificado debe rechazarse.
            self.assertEqual(
                modificado["estado"],
                "failed",
                msg=str(modificado),
            )

    
    def test_silver_detecta_archivo_modificado_tras_manifest(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            tmdb_dir = plan["directorios"]["tmdb"]

            tmdb_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            ruta_parquet = (
                tmdb_dir / "movies_tmdb.parquet"
            )

            ruta_metadata = (
                tmdb_dir / "_metadata.json"
            )

            # 1. Crear archivo TMDb original.
            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula Original"],
            }).to_parquet(
                ruta_parquet,
                index=False,
            )

            ruta_metadata.write_text(
                json.dumps({
                    "fuente": "tmdb",
                    "run_id": run_id,
                    "estado": "success",
                }),
                encoding="utf-8",
            )

            # 2. Construir el manifiesto original.
            manifest = build_silver_manifest(
                local_dir=tmdb_dir,
                source="tmdb",
                run_id=run_id,
            )

            # 3. Verificar antes de modificar.
            verificacion_original = (
                verify_silver_local_manifest(
                    manifest=manifest,
                    local_dir=tmdb_dir,
                )
            )

            # 4. Modificar el contenido del Parquet.
            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula Modificada"],
            }).to_parquet(
                ruta_parquet,
                index=False,
            )

            # 5. Verificar usando el manifiesto
            # ORIGINAL, sin reconstruirlo.
            verificacion_modificada = (
                verify_silver_local_manifest(
                    manifest=manifest,
                    local_dir=tmdb_dir,
                )
            )

            # La verificación debe cambiar de
            # resultado al detectar la modificación.
            self.assertNotEqual(
                verificacion_original,
                verificacion_modificada,
                msg=(
                    "La verificación no detectó "
                    "el cambio en el archivo TMDb."
                ),
            )

            # Confirmar el contenido nuevo.
            recuperado = pd.read_parquet(
                ruta_parquet
            )

            self.assertEqual(
                recuperado.iloc[0]["title"],
                "Pelicula Modificada",
            )

    
    def test_orquestador_valida_archivos_reales_completos(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            directorios = plan["directorios"]

            # Crear las cuatro carpetas Silver.
            for fuente in plan["orden"]:
                directorios[fuente].mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # 1. MovieLens
            movielens = directorios["movielens"]

            archivos_movielens = {
                "movies.parquet": {
                    "movie_id": [1],
                    "title": ["Pelicula A"],
                },
                "links.parquet": {
                    "movie_id": [1],
                    "tmdb_id": [550],
                },
                "tags.parquet": {
                    "movie_id": [1],
                    "tag": ["Drama"],
                },
            }

            for nombre, datos in archivos_movielens.items():
                pd.DataFrame(datos).to_parquet(
                    movielens / nombre,
                    index=False,
                )

            ratings_dir = movielens / "ratings"
            ratings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "user_id": [1],
                "movie_id": [1],
                "rating": [4.5],
            }).to_parquet(
                ratings_dir / "part_000.parquet",
                index=False,
            )

            # 2. TMDb
            pd.DataFrame({
                "tmdb_id": [550],
                "title": ["Pelicula A"],
            }).to_parquet(
                directorios["tmdb"] / "movies_tmdb.parquet",
                index=False,
            )

            # 3. Watchmode
            pd.DataFrame({
                "tmdb_id": [550],
                "provider_name": ["Plataforma A"],
            }).to_parquet(
                directorios["watchmode"]
                / "availability.parquet",
                index=False,
            )

            pd.DataFrame({
                "tmdb_id": [550],
                "has_availability": [True],
                "sources_count": [1],
            }).to_parquet(
                directorios["watchmode"]
                / "availability_status.parquet",
                index=False,
            )

            # 4. Integrated
            pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": [550],
                "canonical_title": ["Pelicula A"],
            }).to_parquet(
                directorios["integrated"]
                / "movies_integrated.parquet",
                index=False,
            )

            # Metadatos para las cuatro fuentes.
            for fuente in plan["orden"]:

                metadata = {
                    "fuente": fuente,
                    "capa": "silver",
                    "run_id": run_id,
                    "estado": "success",
                }

                ruta_metadata = (
                    directorios[fuente] / "_metadata.json"
                )

                ruta_metadata.write_text(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            # Las cuatro transformaciones
            # se consideran completadas.
            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": plan["orden"],
            }

            # Validación REAL, sin mocks.
            resultado = validate_silver_run(
                plan=plan,
                execution_result=execution_result,
            )

            self.assertEqual(
                resultado["estado"],
                "validated",
                msg=str(resultado),
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertEqual(
                resultado["evaluacion"]["estado"],
                "ready",
            )

            # Confirmar los cuatro manifiestos.
            self.assertEqual(
                set(resultado["manifests"]),
                set(plan["orden"]),
            )

    
    def test_orquestador_valida_silver_completa(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": plan["orden"],
            }

            def crear_manifest(**kwargs):
                return {
                    "source": kwargs["source"],
                    "run_id": kwargs["run_id"],
                }

            with (
                patch(
                    "src.load.silver_storage."
                    "build_silver_manifest",
                    side_effect=crear_manifest,
                ) as mock_manifest,
                patch(
                    "src.load.silver_storage."
                    "validate_silver_manifests",
                    return_value={
                        "estado": "ready",
                        "errores": [],
                    },
                ) as mock_validacion,
            ):

                resultado = validate_silver_run(
                    plan=plan,
                    execution_result=execution_result,
                )

            # Se reconoce una ejecución validada.
            self.assertEqual(
                resultado["estado"],
                "validated",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            # Se construyeron los cuatro manifiestos.
            self.assertEqual(
                mock_manifest.call_count,
                4,
            )

            self.assertEqual(
                set(resultado["manifests"]),
                {
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                },
            )

            # Cada manifiesto pertenece al mismo run_id.
            for fuente, manifest in resultado["manifests"].items():

                self.assertEqual(
                    manifest["source"],
                    fuente,
                )

                self.assertEqual(
                    manifest["run_id"],
                    run_id,
                )

            mock_validacion.assert_called_once()

            # La evaluación confirma el estado ready.
            self.assertEqual(
                resultado["evaluacion"]["estado"],
                "ready",
            )

            # No se generan archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )

    
    def test_orquestador_bloquea_silver_incompleta(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # Simular que las transformaciones
            # finalizaron correctamente.
            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": plan["orden"],
            }

            # Los manifiestos se construyen,
            # pero la evaluación detecta que
            # faltan archivos obligatorios.
            with (
                patch(
                    "src.load.silver_storage."
                    "build_silver_manifest",
                    side_effect=lambda **kwargs: {
                        "source": kwargs["source"],
                    },
                ) as mock_manifest,
                patch(
                    "src.load.silver_storage."
                    "validate_silver_manifests",
                    return_value={
                        "estado": "blocked",
                        "errores": [
                            "Faltan archivos Silver"
                        ],
                    },
                ) as mock_validacion,
            ):

                resultado = validate_silver_run(
                    plan=plan,
                    execution_result=execution_result,
                )

            # La ejecución no se considera validada.
            self.assertEqual(
                resultado["estado"],
                "blocked",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            # Se revisaron las cuatro fuentes.
            self.assertEqual(
                mock_manifest.call_count,
                4,
            )

            mock_validacion.assert_called_once()

            # No se crean archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )

    
    def test_orquestador_prepara_cuatro_fuentes_mismo_run_id(
        self,
    ):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            silver_root = Path(temporal) / "silver"

            plan = prepare_silver_run(
                silver_root=silver_root,
                run_id=run_id,
            )

            self.assertEqual(
                plan["run_id"],
                run_id,
            )

            self.assertEqual(
                plan["estado"],
                "planned",
            )

            self.assertEqual(
                plan["orden"],
                (
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                ),
            )

            run_root = (
                silver_root
                / "runs"
                / f"run_id={run_id}"
            )

            self.assertEqual(
                plan["directorios"]["root"],
                run_root,
            )

            for fuente in plan["orden"]:

                self.assertEqual(
                    plan["directorios"][fuente],
                    run_root / fuente,
                )

            # La preparación no debe escribir archivos
            # ni crear carpetas.
            self.assertFalse(
                run_root.exists()
            )


    def test_orquestador_ejecuta_cuatro_fuentes_en_orden(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            llamadas = []

            def simular_transformacion(fuente):

                def ejecutar(**kwargs):
                    llamadas.append((fuente, kwargs))
                    return {"estado": "simulado"}

                return ejecutar

            transformers = {
                fuente: Mock(
                    side_effect=simular_transformacion(fuente)
                )
                for fuente in plan["orden"]
            }

            resultado = execute_silver_run(
                plan=plan,
                transformers=transformers,
            )

            self.assertEqual(
                resultado["estado"],
                "transformed",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertEqual(
                resultado["fuentes_completadas"],
                plan["orden"],
            )

            self.assertEqual(
                [fuente for fuente, _ in llamadas],
                [
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                ],
            )

            for fuente in (
                "movielens",
                "tmdb",
                "watchmode",
            ):

                transformers[fuente].assert_called_once_with(
                    output_dir=plan["directorios"][fuente],
                )

            transformers["integrated"].assert_called_once_with(
                run_dirs=plan["directorios"],
            )

            # No se crean carpetas ni archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )
    
    
    def test_orquestador_bloquea_archivos_reales_incompletos(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            directorios = plan["directorios"]

            # Crear las carpetas de las cuatro fuentes.
            for fuente in plan["orden"]:
                directorios[fuente].mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # Crear archivos Parquet locales reales
            # para MovieLens.
            movielens = directorios["movielens"]

            for nombre in (
                "movies.parquet",
                "links.parquet",
                "tags.parquet",
            ):
                pd.DataFrame({
                    "dato_prueba": [1]
                }).to_parquet(
                    movielens / nombre,
                    index=False,
                )

            ratings_dir = movielens / "ratings"

            ratings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "rating": [4.5]
            }).to_parquet(
                ratings_dir / "part_000.parquet",
                index=False,
            )

            # Crear el archivo de TMDb.
            pd.DataFrame({
                "tmdb_id": [550]
            }).to_parquet(
                directorios["tmdb"] / "movies_tmdb.parquet",
                index=False,
            )

            # Watchmode: crear solo availability.
            # Dejamos ausente availability_status.
            pd.DataFrame({
                "tmdb_id": [550]
            }).to_parquet(
                directorios["watchmode"]
                / "availability.parquet",
                index=False,
            )

            # Crear Integrated.
            pd.DataFrame({
                "movie_id": [1]
            }).to_parquet(
                directorios["integrated"]
                / "movies_integrated.parquet",
                index=False,
            )

            # Crear metadatos locales en cada fuente.
            for fuente in plan["orden"]:

                ruta_metadata = (
                    directorios[fuente] / "_metadata.json"
                )

                ruta_metadata.write_text(
                    json.dumps({
                        "fuente": fuente,
                        "run_id": run_id,
                        "estado": "success",
                    }),
                    encoding="utf-8",
                )

            execution_result = {
                "run_id": run_id,
                "estado": "transformed",
                "fuentes_completadas": plan["orden"],
            }

            # Utilizar la validación REAL.
            # No hay mocks en esta prueba.
            resultado = validate_silver_run(
                plan=plan,
                execution_result=execution_result,
            )

            self.assertEqual(
                resultado["estado"],
                "blocked",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            # Confirmar que realmente falta el
            # archivo obligatorio de Watchmode.
            self.assertFalse(
                (
                    directorios["watchmode"]
                    / "availability_status.parquet"
                ).exists()
            )

    
    def test_orquestador_detiene_ejecucion_si_tmdb_falla(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # MovieLens termina correctamente.
            mock_movielens = Mock(
                return_value={"estado": "success"}
            )

            # TMDb simula un error de transformación.
            mock_tmdb = Mock(
                side_effect=RuntimeError(
                    "Error simulado en TMDb"
                )
            )

            # Estas funciones nunca deben ejecutarse.
            mock_watchmode = Mock()
            mock_integrated = Mock()

            transformers = {
                "movielens": mock_movielens,
                "tmdb": mock_tmdb,
                "watchmode": mock_watchmode,
                "integrated": mock_integrated,
            }

            resultado = execute_silver_run(
                plan=plan,
                transformers=transformers,
            )

            # El orquestador reconoce el error.
            self.assertEqual(
                resultado["estado"],
                "failed",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertEqual(
                resultado["fuente_fallida"],
                "tmdb",
            )

            self.assertIn(
                "Error simulado en TMDb",
                resultado["error"],
            )

            # Solo MovieLens completó su ejecución.
            self.assertEqual(
                resultado["fuentes_completadas"],
                ("movielens",),
            )

            mock_movielens.assert_called_once_with(
                output_dir=plan["directorios"]["movielens"],
            )

            mock_tmdb.assert_called_once_with(
                output_dir=plan["directorios"]["tmdb"],
            )

            # No debe continuar después del fallo.
            mock_watchmode.assert_not_called()
            mock_integrated.assert_not_called()

            # No se generan archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )

    
    def test_orquestador_rechaza_transformacion_sin_resultado(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            mock_movielens = Mock(
                return_value={"estado": "success"}
            )

            # TMDb termina sin excepción,
            # pero no devuelve datos.
            mock_tmdb = Mock(
                return_value=None
            )

            mock_watchmode = Mock()
            mock_integrated = Mock()

            resultado = execute_silver_run(
                plan=plan,
                transformers={
                    "movielens": mock_movielens,
                    "tmdb": mock_tmdb,
                    "watchmode": mock_watchmode,
                    "integrated": mock_integrated,
                },
            )

            self.assertEqual(
                resultado["estado"],
                "failed",
            )

            self.assertEqual(
                resultado["fuente_fallida"],
                "tmdb",
            )

            self.assertIn(
                "no devolvió ningún resultado",
                resultado["error"],
            )

            self.assertEqual(
                resultado["fuentes_completadas"],
                ("movielens",),
            )

            mock_movielens.assert_called_once()
            mock_tmdb.assert_called_once()

            # No continuar con fuentes posteriores.
            mock_watchmode.assert_not_called()
            mock_integrated.assert_not_called()

            # No se escriben archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )


if __name__ == "__main__":
    unittest.main()
