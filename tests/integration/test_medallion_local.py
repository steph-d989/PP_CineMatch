
import hashlib
import inspect
import json
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from src.transformation import bronze_to_silver as b

from src.transformation.silver_pipeline import (
    prepare_silver_run,
    build_silver_transformers,
    execute_silver_run,
    validate_silver_run,
    reverify_silver_run,
)

from src.transformation.silver_to_gold import (
    run_silver_to_gold,
)

from src.utils.gold_runs import (
    build_gold_run_directories,
    verify_gold_manifest,
)


class MedallionLocalTests(unittest.TestCase):

    def test_bronze_silver_gold_integracion_local(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            base = Path(temporal)

            bronze_ml = base / "bronze" / "movielens"
            bronze_tmdb = base / "bronze" / "tmdb"
            bronze_watchmode = base / "bronze" / "watchmode"

            ml_dataset = bronze_ml / "ml-32m"

            for directorio in (
                ml_dataset,
                bronze_tmdb,
                bronze_watchmode,
            ):
                directorio.mkdir(
                    parents=True,
                    exist_ok=True,
                )

            # ---------------------------------------
            # 1. CREAR ARCHIVOS BRONZE MOVIELENS
            # ---------------------------------------

            pd.DataFrame({
                "movieId": [1, 2],
                "title": [
                    "Fight Club (1999)",
                    "Pelicula B (2001)",
                ],
                "genres": [
                    "Drama|Thriller",
                    "Drama",
                ],
            }).to_csv(
                ml_dataset / "movies.csv",
                index=False,
            )


            pd.DataFrame({
                "userId": [10, 20],
                "movieId": [1, 1],
                "rating": [4.5, 3.5],
                "timestamp": [
                    1704067200,
                    1704153600,
                ],
            }).to_csv(
                ml_dataset / "ratings.csv",
                index=False,
            )


            pd.DataFrame({
                "userId": [10],
                "movieId": [1],
                "tag": ["psychological thriller"],
                "timestamp": [1704067200],
            }).to_csv(
                ml_dataset / "tags.csv",
                index=False,
            )


            pd.DataFrame({
                "movieId": [1, 2],
                "imdbId": ["0137523", "0000002"],
                "tmdbId": [550, 551],
            }).to_csv(
                ml_dataset / "links.csv",
                index=False,
            )


            # ---------------------------------------
            # 2. CREAR ARCHIVO BRONZE TMDB
            # ---------------------------------------

            pelicula_tmdb = {
                "id": 550,
                "imdb_id": "tt0137523",
                "title": "Fight Club",
                "original_title": "Fight Club",
                "overview": "A psychological drama.",
                "release_date": "1999-10-15",
                "runtime": 139,
                "original_language": "en",
                "popularity": 45.0,
                "vote_average": 8.4,
                "vote_count": 20000,
                "poster_path": "/poster.jpg",
                "backdrop_path": "/backdrop.jpg",
                "genres": [
                    {"id": 18, "name": "Drama"},
                    {"id": 53, "name": "Thriller"},
                ],
                "production_companies": [
                    {"name": "Test Studio"},
                ],
                "credits": {
                    "cast": [
                        {"name": "Actor A"},
                        {"name": "Actor B"},
                    ],
                    "crew": [
                        {
                            "name": "Director A",
                            "job": "Director",
                        }
                    ],
                },
                "keywords": {
                    "keywords": [
                        {"name": "identity"},
                    ],
                },
            }

            (bronze_tmdb / "movie_550.json").write_text(
                json.dumps(pelicula_tmdb),
                encoding="utf-8",
            )

            
            # Segunda película TMDb de prueba.
            pelicula_tmdb_b = {
                **pelicula_tmdb,
                "id": 551,
                "imdb_id": "tt0000002",
                "title": "Pelicula B",
                "original_title": "Pelicula B",
                "overview": "A second test movie.",
                "release_date": "2001-01-01",
            }

            (
                bronze_tmdb / "movie_551.json"
            ).write_text(
                json.dumps(pelicula_tmdb_b),
                encoding="utf-8",
            )

            
            # ---------------------------------------
            # 3. CREAR WATCHMODE CON SHA-256 REAL
            # ---------------------------------------

            archivo_watchmode = (
                bronze_watchmode
                / "movie_550_sources.json"
            )

            ofertas = [
                {
                    "source_id": 20,
                    "name": "Plataforma de prueba",
                    "type": "sub",
                    "region": "PE",
                    "format": "HD",
                    "price": 0.0,
                    "web_url": None,
                    "ios_url": None,
                    "android_url": None,
                },
            ]

            archivo_watchmode.write_text(
                json.dumps(ofertas),
                encoding="utf-8",
            )

            sha256 = hashlib.sha256(
                archivo_watchmode.read_bytes()
            ).hexdigest()

            metadata_watchmode = {
                "tmdb_id": 550,
                "region": "PE",
                "region_verified": True,
                "retrieved_at": "2026-11-10T12:00:00Z",
                "sha256": sha256,
            }

            (
                bronze_watchmode
                / "movie_550_sources_metadata.json"
            ).write_text(
                json.dumps(metadata_watchmode),
                encoding="utf-8",
            )

            # Segunda película consultada en Watchmode,
            # con respuesta válida pero sin ofertas.
            archivo_watchmode_b = (
                bronze_watchmode
                / "movie_551_sources.json"
            )

            archivo_watchmode_b.write_text(
                json.dumps([]),
                encoding="utf-8",
            )

            hash_b = hashlib.sha256(
                archivo_watchmode_b.read_bytes()
            ).hexdigest()

            metadata_watchmode_b = {
                "tmdb_id": 551,
                "region": "PE",
                "region_verified": True,
                "retrieved_at": "2026-11-10T12:00:00Z",
                "sha256": hash_b,
            }

            (
                bronze_watchmode
                / "movie_551_sources_metadata.json"
            ).write_text(
                json.dumps(metadata_watchmode_b),
                encoding="utf-8",
            )

            # ---------------------------------------
            # 4. PREPARAR DIRECTORIOS SILVER/GOLD
            # ---------------------------------------

            plan = prepare_silver_run(
                silver_root=base / "silver",
                run_id=run_id,
            )

            silver_dirs = plan["directorios"]

            gold_dirs = build_gold_run_directories(
                gold_root=base / "gold",
                run_id=run_id,
            )

            # Localizar el módulo de procedencia
            # para redirigir únicamente la ruta.
            modulo_procedencia = inspect.getmodule(
                b.get_watchmode_provenance
            )

            # ---------------------------------------
            # 5. EJECUTAR SILVER REAL
            # ---------------------------------------

            with (
                patch.object(
                    b,
                    "BRONZE_MOVIELENS_DIR",
                    bronze_ml,
                ),
                patch.object(
                    b,
                    "BRONZE_TMDB_DIR",
                    bronze_tmdb,
                ),

                patch.object(
                    b,
                    "get_local_watchmode_files",
                    return_value=[
                        archivo_watchmode,
                        archivo_watchmode_b,
                    ],
                ),

                
                patch.object(
                    modulo_procedencia,
                    "build_local_watchmode_path",
                    side_effect=lambda tmdb_id, region="PE": {
                        550: archivo_watchmode,
                        551: archivo_watchmode_b,
                    }[tmdb_id],
                ),

            ):

                ejecucion = execute_silver_run(
                    plan=plan,
                    transformers=build_silver_transformers(),
                )

            self.assertEqual(
                ejecucion["estado"],
                "transformed",
                msg=str(ejecucion),
            )

            self.assertEqual(
                tuple(ejecucion["fuentes_completadas"]),
                (
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                ),
            )

            # ---------------------------------------
            # 6. VALIDAR Y REVERIFICAR SILVER
            # ---------------------------------------

            validacion = validate_silver_run(
                plan=plan,
                execution_result=ejecucion,
            )

            self.assertEqual(
                validacion["estado"],
                "validated",
                msg=str(validacion),
            )

            reverificacion = reverify_silver_run(
                plan=plan,
                validation_result=validacion,
            )

            self.assertEqual(
                reverificacion["estado"],
                "locally_verified",
                msg=str(reverificacion),
            )

            # ---------------------------------------
            # 7. EJECUTAR GOLD REAL
            # ---------------------------------------

            resultado_gold = run_silver_to_gold(
                run_dirs=silver_dirs,
                gold_dirs=gold_dirs,
                processing_date="2026-11-10",
                seal_run=True,
            )

            self.assertEqual(
                resultado_gold["integridad_dim_fact"][
                    "total_relaciones"
                ],
                6,
            )

            cierre = resultado_gold["sellado"]

            self.assertEqual(
                cierre["estado"],
                "sealed",
            )

            self.assertEqual(
                cierre["run_id"],
                run_id,
            )

            # ---------------------------------------
            # 8. VERIFICAR EL MANIFIESTO GOLD
            # ---------------------------------------

            with cierre["manifest_path"].open(
                "r",
                encoding="utf-8",
            ) as archivo:
                manifest = json.load(archivo)

            comprobacion = verify_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifest,
            )

            self.assertEqual(
                comprobacion["estado"],
                "manifest_verified",
            )

            # ---------------------------------------
            # 9. VERIFICAR RESULTADOS GOLD
            # ---------------------------------------

            rutas = resultado_gold["rutas"]

            dim_movie = pd.read_parquet(
                rutas["dim_movie"]
            )

            fact_availability = pd.read_parquet(
                rutas["fact_availability"]
            )

            self.assertIn(
                1,
                dim_movie["movie_id"].tolist(),
            )

            self.assertIn(
                1,
                fact_availability["movie_id"].tolist(),
            )

            self.assertIn(
                20,
                fact_availability["provider_id"].tolist(),
            )
            
            
            # ---------------------------------------
            # 10. AUDITORÍA DE RESULTADOS GOLD
            # ---------------------------------------

            dim_user = pd.read_parquet(
                rutas["dim_user"]
            )

            dim_platform = pd.read_parquet(
                rutas["dim_platform"]
            )

            fact_rating = pd.concat(
                [
                    pd.read_parquet(archivo)
                    for archivo in rutas["fact_rating"]
                ],
                ignore_index=True,
            )

            # El ejemplo Bronze contiene:
            # una película, un usuario,
            # una calificación y una plataforma.

            self.assertEqual(len(dim_movie), 2)
            self.assertEqual(len(dim_user), 2)
            self.assertEqual(len(dim_platform), 1)
            self.assertEqual(len(fact_rating), 2)
            self.assertEqual(len(fact_availability), 1)

            # Comprobar que los identificadores
            # originales llegaron correctamente.
            self.assertEqual(
                int(dim_movie.iloc[0]["movie_id"]),
                1,
            )

            self.assertEqual(
                int(dim_user.iloc[0]["user_id"]),
                10,
            )

            self.assertEqual(
                int(dim_platform.iloc[0]["provider_id"]),
                20,
            )

            # La calificación debe conservar
            # su valor original.
            self.assertAlmostEqual(
                float(fact_rating.iloc[0]["rating"]),
                4.5,
            )

            # Ambas tablas de hechos deben
            # referirse a la misma película.
            self.assertEqual(
                int(fact_rating.iloc[0]["movie_id"]),
                int(fact_availability.iloc[0]["movie_id"]),
            )
            
            
            # ---------------------------------------
            # 11. VALIDAR MÚLTIPLES USUARIOS
            # ---------------------------------------

            # Deben existir exactamente dos usuarios.
            self.assertEqual(
                set(dim_user["user_id"].tolist()),
                {10, 20},
            )

            # Ambas calificaciones deben conservarse.
            ratings_obtenidos = {
                int(fila.user_id): float(fila.rating)
                for fila in fact_rating.itertuples()
            }

            self.assertEqual(
                ratings_obtenidos,
                {
                    10: 4.5,
                    20: 3.5,
                },
            )

            # Ambos usuarios calificaron la película 1.
            self.assertEqual(
                set(fact_rating["movie_id"].tolist()),
                {1},
            )

            # No deben existir registros huérfanos.
            usuarios_huerfanos = (
                set(fact_rating["user_id"].tolist())
                - set(dim_user["user_id"].tolist())
            )

            self.assertEqual(
                usuarios_huerfanos,
                set(),
            )
            
            # ---------------------------------------
            # 12. PELÍCULAS SIN DISPONIBILIDAD
            # ---------------------------------------

            # Las dos películas deben permanecer
            # en el catálogo Gold.
            self.assertEqual(
                set(dim_movie["movie_id"].tolist()),
                {1, 2},
            )

            # Gold no debe inventar ofertas
            # para la película 2.
            self.assertEqual(
                set(fact_availability["movie_id"].tolist()),
                {1},
            )

            # Verificar los resultados reales
            # de la transformación Watchmode.
            watchmode_resultado = (
                ejecucion["resultados"]["watchmode"]
            )

            estados = (
                watchmode_resultado["availability_status"]
                .set_index("tmdb_id")
            )

            self.assertEqual(
                set(estados.index.tolist()),
                {550, 551},
            )

            self.assertTrue(
                bool(
                    estados.loc[
                        550, "has_availability"
                    ]
                )
            )

            self.assertFalse(
                bool(
                    estados.loc[
                        551, "has_availability"
                    ]
                )
            )

            self.assertEqual(
                int(
                    estados.loc[
                        551, "sources_count"
                    ]
                ),
                0,
            )



