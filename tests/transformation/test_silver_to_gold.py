import json
import unittest
import pandas as pd
import inspect
import hashlib

from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.silver_runs import (
    build_silver_run_directories,
)

from src.transformation import (
    bronze_to_silver as b,
)

from src.transformation.silver_to_gold import (
    load_silver_for_gold,
    run_silver_to_gold,
    save_gold_metadata,
    build_dim_user,
    build_fact_availability,
    classify_availability_movie_links,
    build_dim_platform,
    build_dim_date,
    build_dim_movie,
    save_gold_dataset,
)

from src.utils.gold_runs import (
    build_gold_run_directories,
    verify_gold_manifest,
)

from src.validation.gold_integrity import (
    validate_gold_dim_movie_file,
    validate_gold_fact_rating_users_files,
    validate_gold_dim_fact_files,
)

from src.transformation.bronze_to_silver import (
    transform_watchmode_sources,
    transform_watchmode_status,
)

class SilverToGoldTests(unittest.TestCase):
    
    
    def test_watchmode_conserva_estado_sin_ofertas(self):

        registros = [
            {
                "tmdb_id": 550,
                "sources": [
                    {
                        "source_id": 20,
                        "name": "Plataforma A",
                        "type": "sub",
                        "region": "PE",
                        "format": "HD",
                        "price": 0.0,
                        "web_url": None,
                        "ios_url": None,
                        "android_url": None,
                    }
                ],
            },
            {
                "tmdb_id": 551,
                "sources": [],
            },
        ]

        # Ejecutar las transformaciones reales.
        availability = transform_watchmode_sources(
            registros
        )

        status = transform_watchmode_status(
            registros
        )

        # Solo existe una oferta.
        self.assertEqual(
            len(availability),
            1,
        )

        self.assertEqual(
            int(availability.iloc[0]["tmdb_id"]),
            550,
        )

        # Se conserva el estado de ambas películas.
        self.assertEqual(
            len(status),
            2,
        )

        estados = status.set_index("tmdb_id")

        # Película con disponibilidad.
        self.assertTrue(
            bool(estados.loc[550, "has_availability"])
        )

        self.assertEqual(
            int(estados.loc[550, "sources_count"]),
            1,
        )

        # Película sin ofertas registradas.
        self.assertFalse(
            bool(estados.loc[551, "has_availability"])
        )

        self.assertEqual(
            int(estados.loc[551, "sources_count"]),
            0,
        )

        # No debe inventarse una plataforma
        # para la película sin disponibilidad.
        self.assertNotIn(
            551,
            availability["tmdb_id"].tolist(),
        )

    def test_watchmode_bronze_lee_procedencia_verificada(self):

        # Localizar el módulo donde realmente
        # se implementó get_watchmode_provenance.
        modulo_procedencia = inspect.getmodule(
            b.get_watchmode_provenance
        )

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            archivo = (
                raiz / "movie_550_sources.json"
            )

            metadata_path = (
                raiz / "movie_550_sources_metadata.json"
            )

            # Archivo Bronze local con formato válido.
            fuentes = []

            archivo.write_text(
                json.dumps(fuentes),
                encoding="utf-8",
            )

            hash_original = hashlib.sha256(
                archivo.read_bytes()
            ).hexdigest()

            metadata = {
                "tmdb_id": 550,
                "region": "PE",
                "region_verified": True,
                "retrieved_at": "2026-11-10T12:00:00Z",
                "sha256": hash_original,
            }

            metadata_path.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            # Redirigir solamente las rutas locales.
            # Las funciones de validación son reales.
            with (
                patch.object(
                    b,
                    "get_local_watchmode_files",
                    return_value=[archivo],
                ),
                patch.object(
                    modulo_procedencia,
                    "build_local_watchmode_path",
                    return_value=archivo,
                ),
            ):

                # 1. El archivo físico es válido.
                validacion = (
                    b.validate_watchmode_local_file(
                        archivo
                    )
                )

                self.assertTrue(
                    validacion["valid"]
                )

                self.assertTrue(
                    validacion["empty"]
                )

                # 2. La procedencia pasa SHA-256.
                procedencia = (
                    b.get_watchmode_provenance(
                        tmdb_id=550,
                        region="PE",
                    )
                )

                self.assertEqual(
                    procedencia["status"],
                    "verified",
                )

                # 3. El lector real acepta el archivo.
                registros = b.load_watchmode_bronze(
                    region="PE"
                )

                self.assertEqual(
                    len(registros),
                    1,
                )

                self.assertEqual(
                    registros[0]["tmdb_id"],
                    550,
                )

                self.assertEqual(
                    registros[0]["sources"],
                    [],
                )

                # 4. Modificar los bytes sin actualizar
                # la huella registrada.
                archivo.write_text(
                    json.dumps([{}]),
                    encoding="utf-8",
                )

                procedencia_alterada = (
                    b.get_watchmode_provenance(
                        tmdb_id=550,
                        region="PE",
                    )
                )

                self.assertNotEqual(
                    procedencia_alterada["status"],
                    "verified",
                )

                # La lectura Bronze debe rechazar
                # todos los registros sin procedencia.
                with self.assertRaisesRegex(
                    ValueError,
                    "procedencia verificada",
                ):
                    b.load_watchmode_bronze(
                        region="PE"
                    )


    def test_pipeline_gold_no_sella_si_falla_integridad(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
                "canonical_title": ["Pelicula A"],
            }),
            "ratings": pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "rating": [4.5],
                "rated_at": pd.to_datetime(
                    ["2024-01-01"], utc=True
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

            # Se simula una falla en el control de
            # integridad, no en las transformaciones.
            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "validate_gold_dim_fact_files",
                    side_effect=ValueError(
                        "Integridad DIM/FACT inválida"
                    ),
                ) as mock_integridad,
                patch(
                    "src.transformation.silver_to_gold."
                    "seal_gold_run",
                ) as mock_sellado,
            ):

                with self.assertRaisesRegex(
                    ValueError,
                    "Integridad DIM/FACT inválida",
                ):
                    run_silver_to_gold(
                        gold_dirs=gold_dirs,
                        processing_date="2026-11-10",
                        seal_run=True,
                    )

            # El control de integridad se ejecutó.
            mock_integridad.assert_called_once()

            # Ante un fallo, no debe sellar Gold.
            mock_sellado.assert_not_called()

            # El manifiesto no debe existir.
            self.assertFalse(
                (
                    gold_dirs["root"]
                    / "_manifest.json"
                ).exists()
            )

    def test_gold_valida_seis_relaciones_fisicas(self):

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            rutas = {
                "dim_user": raiz / "dim_user.parquet",
                "dim_movie": raiz / "dim_movie.parquet",
                "dim_platform": raiz / "dim_platform.parquet",
                "dim_date": raiz / "dim_date.parquet",
                "fact_rating": [
                    raiz / "fact_rating_001.parquet",
                ],
                "fact_availability": (
                    raiz / "fact_availability.parquet"
                ),
            }

            pd.DataFrame({
                "user_id": [10],
            }).to_parquet(rutas["dim_user"])

            pd.DataFrame({
                "movie_id": [1],
            }).to_parquet(rutas["dim_movie"])

            pd.DataFrame({
                "provider_id": [20],
            }).to_parquet(rutas["dim_platform"])

            pd.DataFrame({
                "date_key": [20261110],
            }).to_parquet(rutas["dim_date"])

            pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "date_key": [20261110],
                "rating": [4.5],
            }).to_parquet(rutas["fact_rating"][0])

            pd.DataFrame({
                "movie_id": [1],
                "provider_id": [20],
                "date_key": [20261110],
            }).to_parquet(rutas["fact_availability"])

            # Primer escenario: las seis
            # relaciones son correctas.
            resultado = validate_gold_dim_fact_files(
                rutas
            )

            self.assertEqual(
                resultado["estado"],
                "validated",
            )

            self.assertEqual(
                resultado["total_relaciones"],
                6,
            )

            self.assertTrue(
                all(
                    relacion["referencias_huerfanas"] == 0
                    for relacion in resultado["relaciones"]
                )
            )

            # Segundo escenario: una plataforma
            # no existe en dim_platform.
            pd.DataFrame({
                "movie_id": [1],
                "provider_id": [99],
                "date_key": [20261110],
            }).to_parquet(rutas["fact_availability"])

            with self.assertRaisesRegex(
                ValueError,
                "referencias inexistentes",
            ):
                validate_gold_dim_fact_files(
                    rutas
                )

    def test_gold_fact_rating_valida_usuarios_fisicos(self):

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            ruta_usuarios = raiz / "dim_user.parquet"
            ruta_ratings = raiz / "part_000.parquet"

            pd.DataFrame({
                "user_id": [10, 20],
            }).to_parquet(
                ruta_usuarios,
                index=False,
            )

            pd.DataFrame({
                "user_id": [10, 20, 10],
                "rating": [4.5, 3.0, 5.0],
            }).to_parquet(
                ruta_ratings,
                index=False,
            )

            # Caso correcto.
            resultado = (
                validate_gold_fact_rating_users_files(
                    dim_user_path=ruta_usuarios,
                    fact_rating_paths=[ruta_ratings],
                )
            )

            self.assertEqual(
                resultado["estado"],
                "validated",
            )

            self.assertEqual(
                resultado["total_ratings"],
                3,
            )

            self.assertEqual(
                resultado["usuarios_huerfanos"],
                0,
            )

            # Introducir un usuario inexistente.
            pd.DataFrame({
                "user_id": [10, 99],
                "rating": [4.5, 3.0],
            }).to_parquet(
                ruta_ratings,
                index=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "sin correspondencia",
            ):
                validate_gold_fact_rating_users_files(
                    dim_user_path=ruta_usuarios,
                    fact_rating_paths=[ruta_ratings],
                )

    def test_gold_dim_movie_fisica_valida_clave_primaria(self):

        with TemporaryDirectory() as temporal:

            ruta = (
                Path(temporal)
                / "dim_movie.parquet"
            )

            # Caso 1: dimensión válida.
            pd.DataFrame({
                "movie_id": [1, 2, 3],
                "canonical_title": [
                    "Pelicula A",
                    "Pelicula B",
                    "Pelicula C",
                ],
            }).to_parquet(
                ruta,
                index=False,
            )

            resultado = validate_gold_dim_movie_file(
                ruta
            )

            self.assertEqual(
                resultado["estado"],
                "validated",
            )

            self.assertEqual(
                resultado["total_peliculas"],
                3,
            )

            self.assertEqual(
                resultado["duplicados"],
                0,
            )

            # Caso 2: identificadores duplicados.
            pd.DataFrame({
                "movie_id": [1, 2, 2],
            }).to_parquet(
                ruta,
                index=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicados",
            ):
                validate_gold_dim_movie_file(ruta)

            # Caso 3: identificadores nulos.
            pd.DataFrame({
                "movie_id": pd.Series(
                    [1, None],
                    dtype="Int64",
                ),
            }).to_parquet(
                ruta,
                index=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "nulos",
            ):
                validate_gold_dim_movie_file(ruta)

            # Caso 4: identificador no positivo.
            pd.DataFrame({
                "movie_id": [1, 0],
            }).to_parquet(
                ruta,
                index=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "no positivos",
            ):
                validate_gold_dim_movie_file(ruta)

    def test_pipeline_gold_exige_directorios_para_sellado(self):

        with patch(
            "src.transformation.silver_to_gold."
            "load_silver_for_gold"
        ) as mock_carga:

            with self.assertRaisesRegex(
                ValueError,
                "requiere gold_dirs versionados",
            ):
                run_silver_to_gold(
                    seal_run=True,
                )

        # Debe rechazar la configuración antes
        # de cargar o procesar datos Silver.
        mock_carga.assert_not_called()

    def test_pipeline_gold_escribe_archivos_reales(self):

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
                    seal_run=True,
                )


            # Confirmar que la integridad semántica
            # fue validada por el propio orquestador.
            integridad = resultado["integridad_dim_fact"]

            self.assertIsNotNone(
                integridad
            )

            self.assertEqual(
                integridad["estado"],
                "validated",
            )

            self.assertEqual(
                integridad["total_relaciones"],
                6,
            )

            # Confirmar que el sellado se completó
            # después de superar las validaciones.
            self.assertEqual(
                resultado["sellado"]["estado"],
                "sealed",
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
            cierre = resultado["sellado"]       

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
            
            
            # Validar físicamente la clave primaria
            # de la dimensión generada por Gold.
            validacion_dim_movie = (
                validate_gold_dim_movie_file(
                    rutas["dim_movie"]
                )
            )

            self.assertEqual(
                validacion_dim_movie["estado"],
                "validated",
            )

            self.assertEqual(
                validacion_dim_movie["total_peliculas"],
                1,
            )

            self.assertEqual(
                validacion_dim_movie["movie_ids_unicos"],
                1,
            )

            self.assertEqual(
                validacion_dim_movie["nulos"],
                0,
            )

            self.assertEqual(
                validacion_dim_movie["duplicados"],
                0,
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

            
            # Comprobar la integridad referencial
            # entre las calificaciones y películas
            # almacenadas físicamente en Gold.

            peliculas_validas = set(
                dim_movie["movie_id"].tolist()
            )

            peliculas_referenciadas = set(
                rating["movie_id"].dropna().tolist()
            )

            peliculas_huerfanas = (
                peliculas_referenciadas
                - peliculas_validas
            )

            # Ninguna calificación debe tener
            # un identificador de película nulo.
            self.assertFalse(
                rating["movie_id"].isna().any(),
                "fact_rating contiene movie_id nulos.",
            )

            # Todas las películas calificadas
            # deben existir en dim_movie.
            self.assertEqual(
                peliculas_huerfanas,
                set(),
                "Existen calificaciones con "
                "películas inexistentes en dim_movie.",
            )

            # Nuestro conjunto de prueba tiene
            # una película calificada: movie_id=1.
            self.assertEqual(
                peliculas_referenciadas,
                {1},
            )

            
            # Validar la integridad referencial
            # usando los Parquet reales de Gold.
            validacion_usuarios = (
                validate_gold_fact_rating_users_files(
                    dim_user_path=rutas["dim_user"],
                    fact_rating_paths=rutas["fact_rating"],
                )
            )

            
            # Verificar la integridad referencial
            # de las películas disponibles en plataformas.

            fact_availability = pd.read_parquet(
                rutas["fact_availability"]
            )

            
            # Validación semántica consolidada:
            # comprobar las seis relaciones DIM/FACT
            # usando los archivos físicos de Gold.

            integridad_gold = validate_gold_dim_fact_files(
                rutas=resultado["rutas"],
            )

            self.assertEqual(
                integridad_gold["estado"],
                "validated",
            )

            self.assertEqual(
                integridad_gold["total_relaciones"],
                6,
            )

            self.assertTrue(
                all(
                    relacion["referencias_huerfanas"] == 0
                    for relacion in integridad_gold["relaciones"]
                )
            )

            # Los seis controles deben corresponder
            # exactamente a las relaciones previstas.
            relaciones_obtenidas = {
                (
                    relacion["hecho"],
                    relacion["clave"],
                    relacion["dimension"],
                )
                for relacion in integridad_gold["relaciones"]
            }

            relaciones_esperadas = {
                ("fact_rating", "user_id", "dim_user"),
                ("fact_rating", "movie_id", "dim_movie"),
                ("fact_rating", "date_key", "dim_date"),
                ("fact_availability", "movie_id", "dim_movie"),
                ("fact_availability", "provider_id", "dim_platform"),
                ("fact_availability", "date_key", "dim_date"),
            }

            self.assertEqual(
                relaciones_obtenidas,
                relaciones_esperadas,
            )

            
            self.assertIn(
                "movie_id",
                fact_availability.columns,
            )

            self.assertFalse(
                fact_availability["movie_id"].isna().any(),
                "Gold contiene ofertas vinculadas "
                "sin movie_id.",
            )

            peliculas_disponibles = set(
                fact_availability["movie_id"].tolist()
            )

            peliculas_catalogo = set(
                dim_movie["movie_id"].tolist()
            )

            peliculas_huerfanas = (
                peliculas_disponibles
                - peliculas_catalogo
            )

            self.assertEqual(
                peliculas_huerfanas,
                set(),
                "fact_availability contiene películas "
                "inexistentes en dim_movie.",
            )

            # El ejemplo Silver contiene una oferta
            # para la película con movie_id=1.
            self.assertEqual(
                peliculas_disponibles,
                {1},
            )

            
            self.assertEqual(
                validacion_usuarios["estado"],
                "validated",
            )

            self.assertEqual(
                validacion_usuarios["total_usuarios"],
                1,
            )

            self.assertEqual(
                validacion_usuarios["total_ratings"],
                1,
            )

            self.assertEqual(
                validacion_usuarios["usuarios_huerfanos"],
                0,
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
        

    
    def test_pipeline_gold_finaliza_rutas_versionadas(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

            def simular_guardado(df, directory, filename):
                return Path(directory) / filename

            def simular_particiones(
                df, directory, folder_name,
                **kwargs
            ):
                return [
                    Path(directory)
                    / folder_name
                    / "part_000.parquet"
                ]

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                    side_effect=simular_guardado,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                    side_effect=simular_particiones,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_metadata",
                    return_value=(
                        gold_dirs["metrics"]
                        / "_metadata.json"
                    ),
                ),
            ):

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            validacion = resultado["validacion_rutas"]

            self.assertIsNotNone(validacion)

            self.assertEqual(
                validacion["estado"],
                "paths_verified",
            )

            self.assertEqual(
                validacion["run_id"],
                run_id,
            )

            self.assertEqual(
                validacion["root"],
                gold_dirs["root"],
            )

            self.assertIn(
                "fact_rating",
                validacion["datasets"],
            )

            self.assertIn(
                "availability_unlinked",
                validacion["datasets"],
            )

            # Todos los guardados se simularon.
            self.assertFalse(
                gold_dirs["root"].exists()
            )
    
    def test_pipeline_gold_rechaza_run_id_ya_existente(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            # Simular una ejecución anterior
            # que dejó archivos incompletos.
            run_root = gold_dirs["root"]

            run_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            archivo_anterior = (
                run_root / "archivo_incompleto.txt"
            )

            archivo_anterior.write_text(
                "Ejecución anterior",
                encoding="utf-8",
            )

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                ) as mock_carga,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                ) as mock_guardado,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                ) as mock_particionado,
            ):

                with self.assertRaisesRegex(
                    FileExistsError,
                    "ejecución Gold ya existe",
                ):
                    run_silver_to_gold(
                        gold_dirs=gold_dirs,
                    )

            # No debe cargar información de Silver.
            mock_carga.assert_not_called()

            # No debe escribir dimensiones ni hechos.
            mock_guardado.assert_not_called()
            mock_particionado.assert_not_called()

            # Debe conservar intacto el archivo anterior.
            self.assertEqual(
                archivo_anterior.read_text(
                    encoding="utf-8"
                ),
                "Ejecución anterior",
            )
    
    def test_pipeline_gold_rechaza_run_id_silver_diferente(self):

        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch

        from src.utils.silver_runs import (
            build_silver_run_directories,
        )

        from src.utils.gold_runs import (
            build_gold_run_directories,
        )

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        run_id_silver = (
            "12345678-1234-4234-8234-123456789abc"
        )

        run_id_gold = (
            "87654321-4321-4321-8321-cba987654321"
        )

        with TemporaryDirectory() as temporal:

            silver_dirs = build_silver_run_directories(
                silver_root=Path(temporal) / "silver",
                run_id=run_id_silver,
            )

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id_gold,
            )

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                ) as mock_carga,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                ) as mock_guardado,
            ):

                with self.assertRaisesRegex(
                    ValueError,
                    "diferentes run_id",
                ):
                    run_silver_to_gold(
                        run_dirs=silver_dirs,
                        gold_dirs=gold_dirs,
                    )

            # No se debe cargar Silver.
            mock_carga.assert_not_called()

            # No se deben guardar archivos Gold.
            mock_guardado.assert_not_called()

            self.assertFalse(
                gold_dirs["root"].exists()
            )

            self.assertFalse(
                silver_dirs["root"].exists()
            )

    def test_pipeline_gold_rechaza_directorios_otro_run_id(self):

        run_id_a = (
            "12345678-1234-4234-8234-123456789abc"
        )

        run_id_b = (
            "87654321-4321-4321-8321-cba987654321"
        )

        with TemporaryDirectory() as temporal:

            gold_root = Path(temporal) / "gold"

            directorios_a = build_gold_run_directories(
                gold_root=gold_root,
                run_id=run_id_a,
            )

            directorios_b = build_gold_run_directories(
                gold_root=gold_root,
                run_id=run_id_b,
            )

            # Simular un error de configuración:
            # metrics pertenece a otra ejecución.
            directorios_incorrectos = {
                **directorios_a,
                "metrics": directorios_b["metrics"],
            }

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                ) as mock_carga,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                ) as mock_guardado,
            ):

                with self.assertRaisesRegex(
                    ValueError,
                    "no pertenece al run_id seleccionado",
                ):
                    run_silver_to_gold(
                        gold_dirs=directorios_incorrectos,
                    )

            # Debe bloquearse antes de cargar Silver.
            mock_carga.assert_not_called()

            # No debe escribir ningún archivo.
            mock_guardado.assert_not_called()

            self.assertFalse(
                directorios_a["root"].exists()
            )

            self.assertFalse(
                directorios_b["root"].exists()
            )
    
    def test_pipeline_gold_guarda_metadata_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                    side_effect=lambda df, directory, filename:
                        Path(directory) / filename,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                    return_value=[
                        gold_dirs["facts"]
                        / "fact_rating"
                        / "part_000.parquet"
                    ],
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_metadata",
                    return_value=(
                        gold_dirs["metrics"] / "_metadata.json"
                    ),
                ) as mock_metadata,
            ):

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            # Debe guardar los metadatos una sola vez.
            mock_metadata.assert_called_once()

            # El destino debe ser metrics de esta ejecución.
            self.assertEqual(
                mock_metadata.call_args.kwargs["output_dir"],
                gold_dirs["metrics"],
            )

            # Debe recibir las rutas de todos los datasets.
            self.assertIn(
                "dim_movie",
                mock_metadata.call_args.kwargs["rutas"],
            )

            self.assertIn(
                "fact_rating",
                mock_metadata.call_args.kwargs["rutas"],
            )

            self.assertIn(
                "availability_unlinked",
                mock_metadata.call_args.kwargs["rutas"],
            )

            # La ruta devuelta pertenece a esta ejecución.
            self.assertEqual(
                resultado["metadata"],
                gold_dirs["metrics"] / "_metadata.json",
            )

            # No se crean archivos reales.
            self.assertFalse(
                gold_dirs["root"].exists()
            )
    
    def test_gold_metadata_guarda_en_directorio_versionado(self):

        import json

        from src.utils.gold_runs import (
            build_gold_run_directories,
        )

        from src.transformation.silver_to_gold import (
            save_gold_metadata,
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
                "dim_movie": (
                    gold_dirs["dimensions"]
                    / "dim_movie.parquet"
                ),
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet",
                ],
            }

            datasets = {
                "dim_movie": pd.DataFrame({
                    "movie_id": [1, 2],
                }),
                "fact_rating": pd.DataFrame({
                    "user_id": [10, 20],
                    "movie_id": [1, 2],
                    "rating": [4.5, 5.0],
                }),
            }

            ruta = save_gold_metadata(
                rutas=rutas,
                datasets=datasets,
                output_dir=gold_dirs["metrics"],
            )

            # Los metadatos pertenecen a esta ejecución.
            self.assertEqual(
                ruta,
                gold_dirs["metrics"] / "_metadata.json",
            )

            self.assertTrue(ruta.is_file())

            metadata = json.loads(
                ruta.read_text(encoding="utf-8")
            )

            self.assertEqual(
                metadata["capa"],
                "gold",
            )

            self.assertEqual(
                metadata["estado"],
                "success",
            )

            self.assertEqual(
                metadata["datasets"]["dim_movie"]["filas"],
                2,
            )

            # Las particiones siguen registradas
            # mediante una lista JSON.
            self.assertEqual(
                metadata["datasets"]["fact_rating"]["rutas"],
                [str(ruta_parquet) for ruta_parquet
                in rutas["fact_rating"]],
            )

            # No debe crear otras carpetas Gold.
            self.assertFalse(
                gold_dirs["quality"].exists()
            )

            self.assertFalse(
                gold_dirs["ml_content"].exists()
            )

    def test_pipeline_gold_guarda_quality_misma_ejecucion(self):
        
        run_id = "12345678-1234-4234-8234-123456789abc"

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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
                "tmdb_id": [550, 700],
                "source_id": [10, 20],
                "provider_name": [
                    "Plataforma A",
                    "Plataforma B",
                ],
                "region": ["PE", "PE"],
                "type": ["sub", "rent"],
                "format": ["HD", "4K"],
                "price": [0.0, 5.0],
                "web_url": [None, None],
                "ios_url": [None, None],
                "android_url": [None, None],
            }),
        }

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            guardados = {}

            def simular_guardado(df, directory, filename):
                guardados[filename] = {
                    "datos": df.copy(),
                    "directorio": Path(directory),
                }
                return Path(directory) / filename

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                    side_effect=simular_guardado,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                    return_value=[
                        gold_dirs["facts"]
                        / "fact_rating"
                        / "part_000.parquet"
                    ],
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_metadata",
                    return_value=(
                        gold_dirs["metrics"] / "_metadata.json"
                    ),
                ),
            ):

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            # Comprobar el destino versionado.
            self.assertEqual(
                guardados["availability_unlinked.parquet"][
                    "directorio"
                ],
                gold_dirs["quality"],
            )

            self.assertEqual(
                resultado["rutas"]["availability_unlinked"],
                gold_dirs["quality"]
                / "availability_unlinked.parquet",
            )

            # TMDb 700 queda sin correspondencia.
            no_vinculadas = guardados[
                "availability_unlinked.parquet"
            ]["datos"]

            self.assertEqual(len(no_vinculadas), 1)

            self.assertEqual(
                int(no_vinculadas.iloc[0]["tmdb_id"]),
                700,
            )

            self.assertTrue(
                pd.isna(no_vinculadas.iloc[0]["movie_id"])
            )

            # La oferta vinculada permanece en FACT.
            vinculadas = guardados[
                "fact_availability.parquet"
            ]["datos"]

            self.assertEqual(len(vinculadas), 1)

            self.assertEqual(
                int(vinculadas.iloc[0]["movie_id"]),
                1,
            )

            # No se escriben archivos reales.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_pipeline_gold_guarda_metricas_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                    side_effect=lambda df, directory, filename:
                        Path(directory) / filename,
                ) as mock_guardado,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                    return_value=[
                        gold_dirs["facts"]
                        / "fact_rating"
                        / "part_000.parquet"
                    ],
                ),

                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_metadata",
                    return_value=(
                        gold_dirs["metrics"] / "_metadata.json"
                    ),
                ),

            ):

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            nombres_metricas = {
                "movie_metrics.parquet",
                "user_metrics.parquet",
            }

            llamadas_metricas = [
                llamada
                for llamada in mock_guardado.call_args_list
                if llamada.args[2] in nombres_metricas
            ]

            # Deben guardarse exactamente dos métricas.
            self.assertEqual(
                len(llamadas_metricas),
                2,
            )

            # Ambas deben utilizar el directorio
            # metrics de esta ejecución.
            for llamada in llamadas_metricas:
                self.assertEqual(
                    Path(llamada.args[1]),
                    gold_dirs["metrics"],
                )

            # Verificar las rutas devueltas.
            self.assertEqual(
                resultado["rutas"]["movie_metrics"],
                gold_dirs["metrics"]
                / "movie_metrics.parquet",
            )

            self.assertEqual(
                resultado["rutas"]["user_metrics"],
                gold_dirs["metrics"]
                / "user_metrics.parquet",
            )

            # No se crean archivos reales.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_pipeline_gold_guarda_hechos_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                    side_effect=lambda df, directory, filename:
                        Path(directory) / filename,
                ) as mock_guardado,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                    return_value=[
                        gold_dirs["facts"]
                        / "fact_rating"
                        / "part_000.parquet"
                    ],
                ) as mock_particionado,

                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_metadata",
                    return_value=(
                        gold_dirs["metrics"] / "_metadata.json"
                    ),
                ),

            ):

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            # Comprobar fact_availability.
            llamadas_availability = [
                llamada
                for llamada in mock_guardado.call_args_list
                if llamada.args[2] == "fact_availability.parquet"
            ]

            self.assertEqual(
                len(llamadas_availability),
                1,
            )

            self.assertEqual(
                Path(llamadas_availability[0].args[1]),
                gold_dirs["facts"],
            )

            # Comprobar fact_rating particionado.
            mock_particionado.assert_called_once()

            self.assertEqual(
                Path(mock_particionado.call_args.args[1]),
                gold_dirs["facts"],
            )

            self.assertEqual(
                mock_particionado.call_args.args[2],
                "fact_rating",
            )

            # Comprobar las rutas devueltas.
            self.assertEqual(
                resultado["rutas"]["fact_availability"],
                gold_dirs["facts"]
                / "fact_availability.parquet",
            )

            self.assertEqual(
                resultado["rutas"]["fact_rating"],
                [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
            )

            # Las escrituras fueron simuladas.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_pipeline_gold_guarda_dimensiones_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

            with (
                patch(
                    "src.transformation.silver_to_gold."
                    "load_silver_for_gold",
                    return_value=silver,
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_dataset",
                    side_effect=lambda df, directory, filename:
                        Path(directory) / filename,
                ) as mock_guardado,
                patch(
                    "src.transformation.silver_to_gold."
                    "save_partitioned_parquet",
                    return_value=[
                        gold_dirs["facts"]
                        / "fact_rating"
                        / "part_000.parquet"
                    ],
                ),
                patch(
                    "src.transformation.silver_to_gold."
                    "save_gold_metadata",
                    return_value=(
                        gold_dirs["metrics"] / "_metadata.json"
                    ),
                ),
            ):

                resultado = run_silver_to_gold(
                    gold_dirs=gold_dirs,
                    processing_date="2026-11-10",
                )

            nombres_dimensiones = {
                "dim_user.parquet",
                "dim_movie.parquet",
                "dim_platform.parquet",
                "dim_date.parquet",
            }

            llamadas_dimensiones = [
                llamada
                for llamada in mock_guardado.call_args_list
                if llamada.args[2] in nombres_dimensiones
            ]
            
            # Se guardan exactamente las cuatro DIM.
            self.assertEqual(
                len(llamadas_dimensiones),
                4,
            )

            # Todas utilizan el mismo directorio.
            for llamada in llamadas_dimensiones:

                self.assertEqual(
                    Path(llamada.args[1]),
                    gold_dirs["dimensions"],
                )

            # Las rutas devueltas corresponden
            # a la ejecución seleccionada.
            for nombre in (
                "dim_user",
                "dim_movie",
                "dim_platform",
                "dim_date",
            ):

                self.assertEqual(
                    resultado["rutas"][nombre],
                    gold_dirs["dimensions"]
                    / f"{nombre}.parquet",
                )

            # Las escrituras están simuladas.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_gold_guarda_dimension_en_directorio_versionado(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        dim_movie = pd.DataFrame({
            "movie_id": [1, 2],
            "title": [
                "Pelicula A",
                "Pelicula B",
            ],
        })

        with TemporaryDirectory() as temporal:

            directorios = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            destino = directorios["dimensions"]

            ruta = save_gold_dataset(
                df=dim_movie,
                directory=destino,
                filename="dim_movie.parquet",
            )

            # El archivo debe pertenecer
            # a la ejecución seleccionada.
            self.assertEqual(
                ruta,
                destino / "dim_movie.parquet",
            )

            self.assertTrue(
                ruta.is_file()
            )

            # Comprobar la lectura real del Parquet.
            recuperado = pd.read_parquet(
                ruta
            )

            self.assertEqual(
                recuperado["movie_id"].tolist(),
                [1, 2],
            )

            self.assertEqual(
                recuperado["title"].tolist(),
                [
                    "Pelicula A",
                    "Pelicula B",
                ],
            )

            # No debe crear carpetas de otras
            # categorías de Gold.
            self.assertFalse(
                directorios["facts"].exists()
            )

            self.assertFalse(
                directorios["metrics"].exists()
            )

    def test_dim_movie_rechaza_movie_id_invalidos(self):

        casos_invalidos = [0, -5, 2.5, "ABC"]

        for identificador in casos_invalidos:

            with self.subTest(movie_id=identificador):

                peliculas = pd.DataFrame({
                    "movie_id": [1, identificador],
                    "canonical_title": [
                        "Pelicula A",
                        "Pelicula B",
                    ],
                })

                with self.assertRaisesRegex(
                    ValueError,
                    "movie_id inválidos",
                ):
                    build_dim_movie(peliculas)

        # Comprobar que los identificadores
        # correctos continúan siendo aceptados.
        peliculas_validas = pd.DataFrame({
            "movie_id": [1, 250],
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
            ],
        })

        resultado = build_dim_movie(
            peliculas_validas
        )

        self.assertEqual(
            resultado["movie_id"].tolist(),
            [1, 250],
        )

        self.assertEqual(
            str(resultado["movie_id"].dtype),
            "int64",
        )

    def test_dim_movie_rechaza_identificadores_duplicados(self):

        # Caso válido: películas únicas.
        movies_validas = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": pd.Series(
                [550, 600],
                dtype="Int64",
            ),
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
            ],
        })

        dim_movie = build_dim_movie(
            movies_validas
        )

        self.assertEqual(
            len(dim_movie),
            2,
        )

        self.assertFalse(
            dim_movie["movie_id"].duplicated().any()
        )

        self.assertEqual(
            dim_movie["title"].tolist(),
            ["Pelicula A", "Pelicula B"],
        )

        # Caso inválido: un movie_id aparece
        # con dos identificadores TMDb diferentes.
        movies_duplicadas = pd.DataFrame({
            "movie_id": [1, 2, 2],
            "tmdb_id": [550, 600, 700],
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "movie_id duplicados",
        ):
            build_dim_movie(
                movies_duplicadas
            )

        # Caso inválido: película sin movie_id.
        movies_nulas = pd.DataFrame({
            "movie_id": pd.Series(
                [1, pd.NA],
                dtype="Int64",
            ),
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "movie_id nulos",
        ):
            build_dim_movie(
                movies_nulas
            )

    def test_pipeline_gold_sincroniza_fecha_dim_y_fact(self):

        from unittest.mock import patch

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        fecha = "2033-06-15"

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

        # Guardar copias en memoria sin crear archivos.
        guardados = {}

        def simular_guardado(df, directory, filename):
            guardados[filename] = df.copy()
            return Path(filename)

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
                side_effect=simular_guardado,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
                return_value=[
                    Path("fact_rating/part_000.parquet")
                ],
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
                return_value=Path("_metadata.json"),
            ),
        ):

            run_silver_to_gold(
                processing_date=fecha,
            )

        dim_date = guardados["dim_date.parquet"]
        fact_availability = guardados[
            "fact_availability.parquet"
        ]

        # La fecha debe existir en el calendario.
        self.assertIn(
            20330615,
            dim_date["date_key"].tolist(),
        )

        # La disponibilidad debe utilizar esa fecha.
        self.assertEqual(
            fact_availability["date_key"].tolist(),
            [20330615],
        )

        # La clave del hecho debe existir en la DIM.
        self.assertTrue(
            fact_availability["date_key"]
            .isin(dim_date["date_key"])
            .all()
        )

    def test_dim_date_incluye_fecha_procesamiento(self):

        ratings = pd.DataFrame({
            "rated_at": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                ],
                utc=True,
            ),
        })

        # Fecha futura fuera del horizonte habitual.
        dim_date = build_dim_date(
            ratings=ratings,
            required_date="2033-06-15",
        )

        # Debe conservar las fechas de ratings.
        self.assertIn(
            20200101,
            dim_date["date_key"].tolist(),
        )

        self.assertIn(
            20200102,
            dim_date["date_key"].tolist(),
        )

        # Debe incluir la fecha de procesamiento.
        self.assertIn(
            20330615,
            dim_date["date_key"].tolist(),
        )

        # La dimensión no debe repetir fechas.
        self.assertFalse(
            dim_date["date_key"].duplicated().any()
        )

        # Caso adicional: fecha anterior al primer rating.
        dim_date_anterior = build_dim_date(
            ratings=ratings,
            required_date="2019-12-31",
        )

        self.assertIn(
            20191231,
            dim_date_anterior["date_key"].tolist(),
        )

        # Rechazar fechas inválidas.
        with self.assertRaises(ValueError):
            build_dim_date(
                ratings=ratings,
                required_date="fecha-invalida",
            )

    def test_pipeline_gold_bloquea_fechas_disponibilidad_huerfanas(self):

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

        # La dimensión contiene la fecha de ratings,
        # pero no la fecha de disponibilidad.
        dim_date_invalida = pd.DataFrame({
            "date_key": [20240101],
        })

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "build_dim_date",
                return_value=dim_date_invalida,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
            ) as mock_particionado,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
            ) as mock_metadata,
        ):

            with self.assertRaisesRegex(
                ValueError,
                "referencias a fechas inexistentes",
            ):
                run_silver_to_gold(
                    processing_date="2026-11-10",
                )

        # El pipeline debe bloquear todas las escrituras.
        mock_guardado.assert_not_called()
        mock_particionado.assert_not_called()
        mock_metadata.assert_not_called()

    def test_pipeline_gold_transmite_fecha_procesamiento(self):

        from unittest.mock import patch

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        fecha = "2026-11-10"

        # No necesitamos crear datasets completos:
        # detenemos el pipeline cuando alcance la
        # construcción de fact_availability.

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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
            }),
        }

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "build_fact_availability",
                side_effect=RuntimeError(
                    "Fecha recibida en Gold"
                ),
            ) as mock_disponibilidad,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
        ):

            with self.assertRaisesRegex(
                RuntimeError,
                "Fecha recibida en Gold",
            ):
                run_silver_to_gold(
                    processing_date=fecha,
                )

        # La función recibe la fecha seleccionada.
        mock_disponibilidad.assert_called_once()

        self.assertEqual(
            mock_disponibilidad.call_args.kwargs[
                "processing_date"
            ],
            fecha,
        )

        # La excepción simulada impide escribir Gold.
        mock_guardado.assert_not_called()

    def test_fact_availability_fecha_procesamiento_controlada(self):

        from src.transformation.silver_to_gold import (
            build_fact_availability,
        )

        dim_movie = pd.DataFrame({
            "movie_id": [1],
            "tmdb_id": pd.Series(
                [550],
                dtype="Int64",
            ),
        })

        dim_platform = pd.DataFrame({
            "provider_id": [10],
            "provider_name": ["Plataforma A"],
        })

        availability = pd.DataFrame({
            "tmdb_id": [550],
            "source_id": [10],
            "provider_name": ["Plataforma A"],
            "region": ["PE"],
            "type": ["sub"],
            "format": ["HD"],
            "price": [0.0],
        })

        # Primera ejecución: fecha controlada.
        resultado_1 = build_fact_availability(
            availability=availability,
            dim_movie=dim_movie,
            dim_platform=dim_platform,
            processing_date="2026-11-10",
        )

        self.assertEqual(
            resultado_1["date_key"].tolist(),
            [20261110],
        )

        # Segunda ejecución: otra fecha de proceso.
        resultado_2 = build_fact_availability(
            availability=availability,
            dim_movie=dim_movie,
            dim_platform=dim_platform,
            processing_date="2026-11-12",
        )

        self.assertEqual(
            resultado_2["date_key"].tolist(),
            [20261112],
        )

        # La película y plataforma no deben cambiar.
        self.assertEqual(
            resultado_1["movie_id"].tolist(),
            resultado_2["movie_id"].tolist(),
        )

        self.assertEqual(
            resultado_1["provider_id"].tolist(),
            resultado_2["provider_id"].tolist(),
        )

        # Una fecha con formato inválido se rechaza.
        with self.assertRaises(ValueError):
            build_fact_availability(
                availability=availability,
                dim_movie=dim_movie,
                dim_platform=dim_platform,
                processing_date="fecha-invalida",
            )

    def test_pipeline_gold_bloquea_disponibilidad_pelicula_huerfana(self):

        from unittest.mock import patch

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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

        # Simular un error después de construir
        # la tabla de hechos: movie_id inexistente.
        fact_invalida = pd.DataFrame({
            "movie_id": [999],
            "tmdb_id": [550],
            "provider_id": [10],
            "provider_name": ["Plataforma A"],
            "region": ["PE"],
            "access_type": ["sub"],
        })

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "build_fact_availability",
                return_value=fact_invalida,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
            ) as mock_particionado,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
            ) as mock_metadata,
        ):

            with self.assertRaisesRegex(
                ValueError,
                "referencias a películas inexistentes",
            ):
                run_silver_to_gold()

        # El error se detecta antes de guardar.
        mock_guardado.assert_not_called()
        mock_particionado.assert_not_called()
        mock_metadata.assert_not_called()

    def test_pipeline_gold_bloquea_plataformas_huerfanas(self):

        from pathlib import Path
        from unittest.mock import patch

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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
                "source_id": [999],
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

        # Simular una dimensión que contiene
        # únicamente el proveedor 10.
        dim_platform_invalida = pd.DataFrame({
            "provider_id": [10],
            "provider_name": ["Plataforma B"],
        })

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "build_dim_platform",
                return_value=dim_platform_invalida,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
            ) as mock_particionado,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
            ) as mock_metadata,
        ):

            with self.assertRaisesRegex(
                ValueError,
                "plataformas inexistentes",
            ):
                run_silver_to_gold()

        # El error debe impedir cualquier escritura.
        mock_guardado.assert_not_called()
        mock_particionado.assert_not_called()
        mock_metadata.assert_not_called()

    def test_fact_availability_conserva_ofertas_diferentes(self):

        from src.transformation.silver_to_gold import (
            build_fact_availability,
        )

        dim_movie = pd.DataFrame({
            "movie_id": [1],
            "tmdb_id": pd.Series(
                [550],
                dtype="Int64",
            ),
        })

        dim_platform = pd.DataFrame({
            "provider_id": [10],
            "provider_name": ["Plataforma A"],
        })

        availability = pd.DataFrame({
            "tmdb_id": [550, 550, 550],
            "source_id": [10, 10, 10],
            "provider_name": [
                "Plataforma A",
                "Plataforma A",
                "Plataforma A",
            ],
            "region": ["PE", "PE", "PE"],
            "type": ["rent", "rent", "rent"],
            "format": ["HD", "4K", "HD"],
            "price": [3.99, 5.99, 3.99],
            "web_url": [
                "https://example.com/hd",
                "https://example.com/4k",
                "https://example.com/hd",
            ],
            "ios_url": [None, None, None],
            "android_url": [None, None, None],
        })

        resultado = build_fact_availability(
            availability=availability,
            dim_movie=dim_movie,
            dim_platform=dim_platform,
        )

        # Dos ofertas distintas deben conservarse.
        self.assertEqual(
            len(resultado),
            2,
        )

        self.assertEqual(
            set(resultado["format"]),
            {"HD", "4K"},
        )

        self.assertEqual(
            set(resultado["price"]),
            {3.99, 5.99},
        )

        # Ambas corresponden a la misma película.
        self.assertEqual(
            set(resultado["movie_id"]),
            {1},
        )

        # Ambas corresponden a la misma plataforma.
        self.assertEqual(
            set(resultado["provider_id"]),
            {10},
        )

        # El registro idéntico no debe duplicarse.
        self.assertFalse(
            resultado.duplicated().any()
        )
    
    def test_pipeline_gold_separa_disponibilidad_no_vinculada(self):

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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
                "tmdb_id": [550, 700],
                "source_id": [10, 20],
                "provider_name": [
                    "Plataforma A",
                    "Plataforma B",
                ],
                "region": ["PE", "PE"],
                "type": ["sub", "rent"],
                "format": ["HD", "HD"],
                "price": [0.0, 4.0],
                "web_url": [None, None],
                "ios_url": [None, None],
                "android_url": [None, None],
            }),
        }

        guardados = {}
        datasets_metadata = {}

        def simular_guardado(df, directory, filename):
            guardados[filename] = {
                "datos": df.copy(),
                "directorio": Path(directory),
            }
            return Path(directory) / filename


        def simular_metadata(
            rutas,
            datasets,
            output_dir=None,
        ):
            # Conservar los datasets recibidos.
            datasets_metadata.update(datasets)

            # Aceptar el nuevo directorio opcional.
            # No escribir archivos reales.
            if output_dir is not None:
                return Path(output_dir) / "_metadata.json"

            return Path("_metadata.json")


        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
                side_effect=simular_guardado,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
                return_value=[
                    Path("fact_rating/part_000.parquet")
                ],
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
                side_effect=simular_metadata,
            ),
        ):
            resultado = run_silver_to_gold()

        # La tabla de hechos conserva solo TMDb 550.
        vinculadas = guardados[
            "fact_availability.parquet"
        ]["datos"]

        self.assertEqual(len(vinculadas), 1)
        self.assertEqual(
            int(vinculadas.iloc[0]["movie_id"]),
            1,
        )
        self.assertEqual(
            int(vinculadas.iloc[0]["tmdb_id"]),
            550,
        )

        # TMDb 700 queda conservada por separado.
        no_vinculadas = guardados[
            "availability_unlinked.parquet"
        ]["datos"]

        self.assertEqual(len(no_vinculadas), 1)
        self.assertEqual(
            int(no_vinculadas.iloc[0]["tmdb_id"]),
            700,
        )
        self.assertTrue(
            pd.isna(no_vinculadas.iloc[0]["movie_id"])
        )

        # El registro no vinculado no está en FACT.
        self.assertFalse(
            vinculadas["movie_id"].isna().any()
        )

        # Comprobar que ambos datasets se registran.
        self.assertIn(
            "fact_availability",
            resultado["rutas"],
        )
        self.assertIn(
            "availability_unlinked",
            resultado["rutas"],
        )

        self.assertEqual(
            len(datasets_metadata["availability_unlinked"]),
            1,
        )

        # No se perdió ninguna oferta.
        self.assertEqual(
            len(vinculadas) + len(no_vinculadas),
            2,
        )

    def test_gold_clasifica_disponibilidad_sin_movielens(self):

        from src.transformation.silver_to_gold import (
            build_fact_availability,
            classify_availability_movie_links,
        )

        # MovieLens solo conoce TMDb 550 y 600.
        dim_movie = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": pd.Series(
                [550, 600],
                dtype="Int64",
            ),
        })

        dim_platform = pd.DataFrame({
            "provider_id": [10, 20, 30],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
                "Plataforma C",
            ],
        })

        # Watchmode también conoce TMDb 700.
        availability = pd.DataFrame({
            "tmdb_id": [550, 600, 700],
            "source_id": [10, 20, 30],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
                "Plataforma C",
            ],
            "region": ["PE", "PE", "PE"],
            "type": ["sub", "sub", "rent"],
            "format": ["HD", "HD", "HD"],
        })

        # Ejecutar la construcción real.
        ofertas = build_fact_availability(
            availability=availability,
            dim_movie=dim_movie,
            dim_platform=dim_platform,
        )

        # Clasificar los resultados.
        resultado = classify_availability_movie_links(
            fact_availability=ofertas,
        )

        vinculadas = resultado["linked"]
        no_vinculadas = resultado["unlinked"]

        self.assertEqual(
            len(vinculadas),
            2,
        )

        self.assertEqual(
            len(no_vinculadas),
            1,
        )

        # Se conservan todas las ofertas.
        self.assertEqual(
            len(vinculadas) + len(no_vinculadas),
            len(ofertas),
        )

        # Las vinculadas tienen película MovieLens.
        self.assertEqual(
            set(vinculadas["movie_id"].astype(int)),
            {1, 2},
        )

        # TMDb 700 permanece identificable.
        self.assertEqual(
            int(no_vinculadas.iloc[0]["tmdb_id"]),
            700,
        )

        self.assertTrue(
            pd.isna(
                no_vinculadas.iloc[0]["movie_id"]
            )
        )

        # Ninguna vinculada contiene movie_id nulo.
        self.assertFalse(
            vinculadas["movie_id"].isna().any()
        )

    def test_gold_clasifica_disponibilidad_sin_movielens(self):

        # MovieLens solo conoce TMDb 550 y 600.
        dim_movie = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": pd.Series(
                [550, 600],
                dtype="Int64",
            ),
        })

        dim_platform = pd.DataFrame({
            "provider_id": [10, 20, 30],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
                "Plataforma C",
            ],
        })

        # Watchmode también conoce TMDb 700.
        availability = pd.DataFrame({
            "tmdb_id": [550, 600, 700],
            "source_id": [10, 20, 30],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
                "Plataforma C",
            ],
            "region": ["PE", "PE", "PE"],
            "type": ["sub", "sub", "rent"],
            "format": ["HD", "HD", "HD"],
        })

        # Ejecutar la construcción real.
        ofertas = build_fact_availability(
            availability=availability,
            dim_movie=dim_movie,
            dim_platform=dim_platform,
        )

        # Clasificar los resultados.
        resultado = classify_availability_movie_links(
            fact_availability=ofertas,
        )

        vinculadas = resultado["linked"]
        no_vinculadas = resultado["unlinked"]

        self.assertEqual(
            len(vinculadas),
            2,
        )

        self.assertEqual(
            len(no_vinculadas),
            1,
        )

        # Se conservan todas las ofertas.
        self.assertEqual(
            len(vinculadas) + len(no_vinculadas),
            len(ofertas),
        )

        # Las vinculadas tienen película MovieLens.
        self.assertEqual(
            set(vinculadas["movie_id"].astype(int)),
            {1, 2},
        )

        # TMDb 700 permanece identificable.
        self.assertEqual(
            int(no_vinculadas.iloc[0]["tmdb_id"]),
            700,
        )

        self.assertTrue(
            pd.isna(
                no_vinculadas.iloc[0]["movie_id"]
            )
        )

        # Ninguna vinculada contiene movie_id nulo.
        self.assertFalse(
            vinculadas["movie_id"].isna().any()
        )

    def test_fact_availability_rechaza_tmdb_ambiguos(self):

        availability = pd.DataFrame({
            "tmdb_id": [550],
            "source_id": [10],
            "provider_name": ["Plataforma A"],
            "region": ["PE"],
            "type": ["sub"],
            "format": ["HD"],
        })

        dim_platform = pd.DataFrame({
            "provider_id": [10],
            "provider_name": ["Plataforma A"],
        })

        # Dos películas distintas comparten tmdb_id.
        dim_movie_ambigua = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": pd.Series(
                [550, 550],
                dtype="Int64",
            ),
        })

        with self.assertRaisesRegex(
            ValueError,
            "tmdb_id asociados a varios movie_id",
        ):
            build_fact_availability(
                availability=availability,
                dim_movie=dim_movie_ambigua,
                dim_platform=dim_platform,
            )

        # Caso correcto: cada identificador
        # TMDb corresponde a una sola película.
        dim_movie_valida = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": pd.Series(
                [550, 600],
                dtype="Int64",
            ),
        })

        resultado = build_fact_availability(
            availability=availability,
            dim_movie=dim_movie_valida,
            dim_platform=dim_platform,
        )

        self.assertEqual(len(resultado), 1)

        self.assertEqual(
            int(resultado.iloc[0]["movie_id"]),
            1,
        )

        self.assertEqual(
            int(resultado.iloc[0]["provider_id"]),
            10,
        )
    
    def test_dim_platform_rechaza_identificadores_contradictorios(self):

        # Caso válido: la misma plataforma aparece
        # en varias películas.
        availability_valida = pd.DataFrame({
            "tmdb_id": [550, 600, 700],
            "source_id": [10, 10, 20],
            "provider_name": [
                "Netflix",
                "Netflix",
                "Prime Video",
            ],
        })

        dim_platform = build_dim_platform(
            availability_valida
        )

        self.assertEqual(
            len(dim_platform),
            2,
        )

        self.assertEqual(
            set(dim_platform["provider_id"]),
            {10, 20},
        )

        self.assertFalse(
            dim_platform["provider_id"].duplicated().any()
        )

        # Caso inválido: un mismo identificador
        # corresponde a plataformas diferentes.
        availability_invalida = pd.DataFrame({
            "tmdb_id": [550, 600],
            "source_id": [10, 10],
            "provider_name": [
                "Netflix",
                "Disney+",
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "duplicados o contradictorios",
        ):
            build_dim_platform(
                availability_invalida
            )

        # Caso inválido: identificador nulo.
        availability_nula = pd.DataFrame({
            "source_id": pd.Series(
                [10, pd.NA],
                dtype="Int64",
            ),
            "provider_name": [
                "Netflix",
                "Prime Video",
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "source_id nulos",
        ):
            build_dim_platform(
                availability_nula
            )
    
    def test_pipeline_gold_bloquea_fechas_huerfanas(self):

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series(
                    [550],
                    dtype="Int64",
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
                "source_id": [1],
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

        # La calificación es de 2024-01-01,
        # pero la dimensión simulada contiene otra fecha.
        dim_date_invalida = pd.DataFrame({
            "date_key": [20240102],
        })

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "build_dim_date",
                return_value=dim_date_invalida,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
            ) as mock_particionado,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
            ) as mock_metadata,
        ):

            with self.assertRaisesRegex(
                ValueError,
                "referencias a fechas inexistentes",
            ):
                run_silver_to_gold()

        # El error se detecta antes de escribir Gold.
        mock_guardado.assert_not_called()
        mock_particionado.assert_not_called()
        mock_metadata.assert_not_called()

    def test_pipeline_gold_bloquea_peliculas_huerfanas(self):

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
                "canonical_title": ["Pelicula A"],
            }),
            "ratings": pd.DataFrame({
                "user_id": [10],
                "movie_id": [999],
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
                "source_id": [1],
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

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
            ) as mock_particionado,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
            ) as mock_metadata,
        ):

            with self.assertRaisesRegex(
                ValueError,
                "películas inexistentes",
            ):
                run_silver_to_gold()

        # Ningún archivo debe escribirse.
        mock_guardado.assert_not_called()
        mock_particionado.assert_not_called()
        mock_metadata.assert_not_called()
    
    def test_pipeline_gold_bloquea_usuarios_huerfanos(self):

        # Simulamos datasets pequeños.
        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series(
                    [550],
                    dtype="Int64",
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
                "source_id": [1],
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

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "build_dim_user",
                return_value=pd.DataFrame({
                    "user_id": [20],
                }),
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
            ) as mock_particionado,
        ):

            # La dimensión contiene al usuario 20,
            # pero el rating corresponde al 10.
            with self.assertRaisesRegex(
                ValueError,
                "usuarios inexistentes",
            ):
                run_silver_to_gold()

        # La validación debe bloquear la escritura.
        mock_guardado.assert_not_called()
        mock_particionado.assert_not_called()

    def test_pipeline_gold_guarda_dimension_usuarios(self):

        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
                "canonical_title": ["Pelicula A"],
            }),
            "ratings": pd.DataFrame({
                "user_id": [10, 20, 10],
                "movie_id": [1, 1, 1],
                "rating": [4.5, 3.0, 5.0],
                "rated_at": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-03",
                    ],
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
                "source_id": [1],
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

        guardados = {}

        def simular_guardado(df, directory, filename):
            guardados[filename] = df.copy()
            return Path(filename)

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
                side_effect=simular_guardado,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
                return_value=[
                    Path("fact_rating/part_000.parquet")
                ],
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
                return_value=Path("_metadata.json"),
            ),
        ):

            resultado = run_silver_to_gold()

        # El pipeline debe generar dim_user.
        self.assertIn(
            "dim_user.parquet",
            guardados,
        )

        dim_user = guardados["dim_user.parquet"]

        # Dos usuarios únicos: 10 y 20.
        self.assertEqual(
            dim_user["user_id"].tolist(),
            [10, 20],
        )

        self.assertFalse(
            dim_user["user_id"].duplicated().any()
        )

        # La dimensión figura en las rutas Gold.
        self.assertIn(
            "dim_user",
            resultado["rutas"],
        )

        self.assertEqual(
            resultado["rutas"]["dim_user"],
            Path("dim_user.parquet"),
        )
    
    def test_gold_construye_dimension_usuarios_unicos(self):

        ratings = pd.DataFrame({
            "user_id": [10, 20, 10, 30, 20],
            "movie_id": [1, 1, 2, 3, 2],
            "rating": [4.5, 3.0, 5.0, 4.0, 2.5],
        })

        dim_user = build_dim_user(ratings)

        # Debe existir una fila por usuario.
        self.assertEqual(
            len(dim_user),
            3,
        )

        self.assertEqual(
            dim_user["user_id"].tolist(),
            [10, 20, 30],
        )

        self.assertFalse(
            dim_user["user_id"].duplicated().any()
        )

        self.assertFalse(
            dim_user["user_id"].isna().any()
        )

        # No agregamos atributos demográficos
        # que MovieLens no proporciona.
        self.assertEqual(
            dim_user.columns.tolist(),
            ["user_id"],
        )

        # Comprobar rechazo de IDs inválidos.
        ratings_invalidos = pd.DataFrame({
            "user_id": [10, 0, -5],
        })

        with self.assertRaisesRegex(
            ValueError,
            "user_id inválidos",
        ):
            build_dim_user(ratings_invalidos)

    def test_pipeline_gold_usa_directorios_silver_seleccionados(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            run_dirs = build_silver_run_directories(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # Simular una interrupción controlada
            # durante la carga de Silver.
            # Así comprobamos la conexión sin
            # construir ni guardar archivos Gold.
            with patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                side_effect=FileNotFoundError(
                    "Prueba controlada de lectura Silver"
                ),
            ) as mock_carga:

                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "Prueba controlada de lectura Silver",
                ):
                    run_silver_to_gold(
                        run_dirs=run_dirs,
                    )

            # Gold debe transmitir exactamente
            # los directorios seleccionados.
            mock_carga.assert_called_once_with(
                run_dirs=run_dirs,
            )

            # La prueba no debe crear carpetas.
            self.assertFalse(
                run_dirs["root"].exists()
            )

    def test_gold_metadata_registra_particiones_como_lista(self):

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)
            metrics_dir = raiz / "metrics"

            rutas = {
                "dim_movie": (
                    raiz / "dimensions" / "dim_movie.parquet"
                ),
                "fact_rating": [
                    raiz / "facts" / "fact_rating"
                    / "part_000.parquet",
                    raiz / "facts" / "fact_rating"
                    / "part_001.parquet",
                ],
            }

            datasets = {
                "dim_movie": pd.DataFrame({
                    "movie_id": [1, 2],
                }),
                "fact_rating": pd.DataFrame({
                    "user_id": [10, 11],
                    "movie_id": [1, 2],
                    "rating": [4.5, 5.0],
                }),
            }

            with patch(
                "src.transformation.silver_to_gold."
                "GOLD_METRICS_DIR",
                metrics_dir,
            ):

                ruta_metadata = save_gold_metadata(
                    rutas=rutas,
                    datasets=datasets,
                )

            self.assertTrue(ruta_metadata.is_file())

            metadata = json.loads(
                ruta_metadata.read_text(
                    encoding="utf-8"
                )
            )

            # Una dimensión mantiene su ruta individual.
            dim = metadata["datasets"]["dim_movie"]

            self.assertEqual(
                dim["ruta"],
                str(rutas["dim_movie"]),
            )

            self.assertEqual(dim["filas"], 2)

            # Los ratings deben tener una lista JSON.
            ratings = metadata["datasets"]["fact_rating"]

            self.assertIsInstance(
                ratings["rutas"],
                list,
            )

            self.assertEqual(
                ratings["rutas"],
                [str(ruta) for ruta in rutas["fact_rating"]],
            )

            self.assertEqual(
                len(ratings["rutas"]),
                2,
            )

            self.assertEqual(ratings["filas"], 2)

            # Evitar que la lista se guarde como texto.
            self.assertNotIn(
                "ruta",
                ratings,
            )
    
    def test_gold_no_duplica_almacenamiento_fact_rating(self):

        from unittest.mock import patch

        from src.transformation.silver_to_gold import (
            run_silver_to_gold,
        )

        # Datos pequeños para ejecutar las funciones
        # de construcción sin archivos productivos.
        silver = {
            "movies": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": pd.Series([550], dtype="Int64"),
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
                "source_id": [1],
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

        with (
            patch(
                "src.transformation.silver_to_gold."
                "load_silver_for_gold",
                return_value=silver,
            ),
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_dataset",
                return_value=Path("archivo_simulado.parquet"),
            ) as mock_guardado,
            patch(
                "src.transformation.silver_to_gold."
                "save_partitioned_parquet",
                return_value=[
                    Path("fact_rating/part_000.parquet")
                ],
            ) as mock_particionado,
            patch(
                "src.transformation.silver_to_gold."
                "save_gold_metadata",
                return_value=Path("_metadata.json"),
            ),
        ):

            resultado = run_silver_to_gold()

        # La tabla debe guardarse por particiones.
        mock_particionado.assert_called_once()

        self.assertEqual(
            mock_particionado.call_args.args[2],
            "fact_rating",
        )

        # Ninguna llamada al guardado individual
        # debe escribir fact_rating.parquet.
        archivos_individuales = [
            llamada.args[2]
            for llamada in mock_guardado.call_args_list
        ]

        self.assertNotIn(
            "fact_rating.parquet",
            archivos_individuales,
        )

        # La ruta oficial corresponde a particiones.
        self.assertEqual(
            resultado["rutas"]["fact_rating"],
            [Path("fact_rating/part_000.parquet")],
        )

    def test_gold_lee_silver_de_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            run_dirs = build_silver_run_directories(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            movielens = run_dirs["movielens"]
            watchmode = run_dirs["watchmode"]
            integrated = run_dirs["integrated"]

            # Crear los directorios temporales.
            (movielens / "ratings").mkdir(
                parents=True,
                exist_ok=True,
            )

            watchmode.mkdir(
                parents=True,
                exist_ok=True,
            )

            integrated.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Catálogo integrado.
            pd.DataFrame({
                "movie_id": [1, 2],
                "canonical_title": [
                    "Pelicula A",
                    "Pelicula B",
                ],
            }).to_parquet(
                integrated / "movies_integrated.parquet",
                index=False,
            )

            # Calificaciones particionadas.
            pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "rating": [4.5],
            }).to_parquet(
                movielens / "ratings" / "part_000.parquet",
                index=False,
            )

            # Etiquetas.
            pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "tag": ["Drama"],
            }).to_parquet(
                movielens / "tags.parquet",
                index=False,
            )

            # Plataformas.
            pd.DataFrame({
                "tmdb_id": [550],
                "provider_name": ["Plataforma A"],
            }).to_parquet(
                watchmode / "availability.parquet",
                index=False,
            )

            # Ejecutar la lectura real.
            datasets = load_silver_for_gold(
                run_dirs=run_dirs,
            )

            self.assertEqual(
                set(datasets),
                {
                    "movies",
                    "ratings",
                    "tags",
                    "availability",
                },
            )

            self.assertEqual(
                len(datasets["movies"]),
                2,
            )

            self.assertEqual(
                datasets["ratings"].iloc[0]["rating"],
                4.5,
            )

            self.assertEqual(
                datasets["tags"].iloc[0]["tag"],
                "Drama",
            )

            self.assertEqual(
                datasets["availability"].iloc[0][
                    "provider_name"
                ],
                "Plataforma A",
            )
