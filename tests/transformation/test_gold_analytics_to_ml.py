
import unittest
import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.transformation.gold_analytics_to_ml import (
    load_gold_analytics,
    run_gold_analytics_to_ml,
    build_content_based_features,
)

from src.utils.gold_runs import (
    build_gold_run_directories,
)

from src.transformation.silver_to_gold import (
    build_dim_movie,
)

class GoldAnalyticsToMLTests(unittest.TestCase):

    
    def test_content_based_rechaza_catalogo_sin_semantica(self):

        # Un catálogo con títulos, pero sin
        # contenido semántico aprovechable.
        incompleto = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": [550, 551],
            "title": ["Pelicula A", "Pelicula B"],
            "genres": [[], []],
            "keywords": [[], []],
            "overview": [None, ""],
        })

        with self.assertRaisesRegex(
            ValueError,
            "información semántica aprovechable",
        ):
            build_content_based_features(
                incompleto
            )

        # Una película puede tener información
        # semántica aunque otra esté incompleta.
        parcial = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": [550, 551],
            "title": ["Pelicula A", "Pelicula B"],
            "genres": [["Drama"], []],
            "keywords": [[], []],
            "overview": [None, None],
        })

        resultado = build_content_based_features(
            parcial
        )

        self.assertEqual(
            len(resultado),
            2,
        )

        self.assertEqual(
            resultado.iloc[0]["genres"],
            ["Drama"],
        )

    
    def test_gold_ml_conserva_atributos_semanticos(self):

        # Representar una película Silver integrada
        # que ya contiene información de TMDb.
        silver_movies = pd.DataFrame({
            "movie_id": [1],
            "tmdb_id": [550],
            "canonical_title": ["Fight Club"],
            "genres": [["Drama", "Thriller"]],
            "tmdb_genres": [["Drama", "Thriller"]],
            "overview": [
                "A psychological drama about identity."
            ],
            "director": ["Director A"],
            "actors": [["Actor A", "Actor B"]],
            "keywords": [["identity", "psychology"]],
            "original_language": ["en"],
            "release_date": ["1999-10-15"],
        })

        # Silver -> Gold Analytics.
        dim_movie = build_dim_movie(
            silver_movies
        )

        # Gold Analytics -> Gold ML.
        content = build_content_based_features(
            dim_movie
        )

        atributos = {
            "genres",
            "tmdb_genres",
            "overview",
            "director",
            "actors",
            "keywords",
            "original_language",
            "release_date",
        }

        # Los atributos deben mantenerse.
        self.assertTrue(
            atributos.issubset(content.columns),
            msg=(
                "Gold ML perdió atributos "
                "necesarios para el recomendador."
            ),
        )

        self.assertEqual(
            len(content),
            1,
        )

        self.assertEqual(
            content.iloc[0]["director"],
            "Director A",
        )

        self.assertEqual(
            content.iloc[0]["keywords"],
            ["identity", "psychology"],
        )

        self.assertEqual(
            content.iloc[0]["overview"],
            "A psychological drama about identity.",
        )

    
    def test_gold_ml_guarda_en_mismo_run_id(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        analytics = {

            "dim_movie": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": [550],
                "title": ["Pelicula A"],
                "genres": [["Drama"]],
                "overview": [
                    "Una película dramática de prueba."
                ],
            }),

            "fact_rating": pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "rating": [4.5],
                "rated_at": pd.to_datetime(
                    ["2024-01-01"],
                    utc=True,
                ),
            }),
        }

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            with (
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "load_gold_analytics",
                    return_value=analytics,
                ),
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "save_ml_dataset",
                    side_effect=lambda df, directory, filename:
                        Path(directory) / filename,
                ) as mock_content,
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "save_partitioned_parquet",
                    side_effect=lambda df, directory, folder_name:
                        [
                            Path(directory)
                            / folder_name
                            / "part_000.parquet"
                        ],
                ) as mock_collaborative,
            ):

                resultado = run_gold_analytics_to_ml(
                    gold_dirs=gold_dirs,
                )

            # Verificar el directorio de contenido.
            self.assertEqual(
                Path(mock_content.call_args.args[1]),
                gold_dirs["ml_content"],
            )

            # Verificar el directorio colaborativo.
            self.assertEqual(
                Path(mock_collaborative.call_args.args[1]),
                gold_dirs["ml_collaborative"],
            )

            self.assertEqual(
                resultado["content_based_features"],
                gold_dirs["ml_content"]
                / "content_based_features.parquet",
            )

            self.assertEqual(
                resultado["collaborative_interactions"],
                [
                    gold_dirs["ml_collaborative"]
                    / "collaborative_interactions"
                    / "part_000.parquet"
                ],
            )

            # No deben crearse archivos reales.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_orquestador_gold_ml_recibe_run_id_versionado(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        analytics = {

            "dim_movie": pd.DataFrame({
                "movie_id": [1],
                "tmdb_id": [550],
                "title": ["Pelicula A"],
                "genres": [["Drama"]],
                "overview": [
                    "Una película dramática de prueba."
                ],
            }),

            "fact_rating": pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "rating": [4.5],
                "rated_at": pd.to_datetime(
                    ["2024-01-01"],
                    utc=True,
                ),
            }),
        }

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            with (
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "load_gold_analytics",
                    return_value=analytics,
                ) as mock_carga,
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "save_ml_dataset",
                    return_value=Path(
                        "content_based_features.parquet"
                    ),
                ),
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "save_partitioned_parquet",
                    return_value=[
                        Path("part_000.parquet")
                    ],
                ),
            ):

                resultado = run_gold_analytics_to_ml(
                    gold_dirs=gold_dirs,
                )

            # El orquestador debe solicitar
            # exactamente la ejecución seleccionada.
            mock_carga.assert_called_once_with(
                gold_dirs=gold_dirs,
            )

            self.assertIn(
                "content_based_features",
                resultado,
            )

            self.assertIn(
                "collaborative_interactions",
                resultado,
            )

            # No se realizaron escrituras reales.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_gold_ml_rechaza_ejecucion_sin_manifiesto(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.utils.gold_runs import (
            build_gold_run_directories,
        )

        from src.transformation.gold_analytics_to_ml import (
            load_gold_analytics,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            # Gold ML debe rechazar una ejecución
            # que no contiene _manifest.json.
            with self.assertRaisesRegex(
                FileNotFoundError,
                "requiere _manifest.json",
            ):
                load_gold_analytics(
                    gold_dirs=gold_dirs,
                )

            # La verificación no debe crear archivos.
            self.assertFalse(
                gold_dirs["root"].exists()
            )

    def test_gold_ml_lee_ejecucion_versionada(self):


        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            dim_movie_path = (
                gold_dirs["dimensions"]
                / "dim_movie.parquet"
            )

            ratings_dir = (
                gold_dirs["facts"]
                / "fact_rating"
            )

            dim_movie_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            ratings_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "movie_id": [1, 2],
                "title": ["Pelicula A", "Pelicula B"],
            }).to_parquet(
                dim_movie_path,
                index=False,
            )

            pd.DataFrame({
                "user_id": [10, 20],
                "movie_id": [1, 2],
                "rating": [4.5, 3.5],
            }).to_parquet(
                ratings_dir / "part_000.parquet",
                index=False,
            )

            
            # ---------------------------------------
            # PREPARAR MANIFIESTO GOLD COMPLETO
            # ---------------------------------------

            from src.utils.gold_runs import (
                build_gold_manifest,
                save_gold_manifest,
            )

            # El manifiesto exige los nueve datasets
            # obligatorios de Gold Analytics.
            archivos_adicionales = {
                "dim_user": (
                    gold_dirs["dimensions"]
                    / "dim_user.parquet"
                ),
                "dim_platform": (
                    gold_dirs["dimensions"]
                    / "dim_platform.parquet"
                ),
                "dim_date": (
                    gold_dirs["dimensions"]
                    / "dim_date.parquet"
                ),
                "fact_availability": (
                    gold_dirs["facts"]
                    / "fact_availability.parquet"
                ),
                "movie_metrics": (
                    gold_dirs["metrics"]
                    / "movie_metrics.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"]
                    / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"]
                    / "availability_unlinked.parquet"
                ),
            }

            # Crear archivos físicos de laboratorio.
            # Esta prueba evalúa la lectura versionada,
            # no los esquemas de todas las tablas.
            for ruta in archivos_adicionales.values():

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

            # Crear metadata Gold.
            metadata = (
                gold_dirs["metrics"]
                / "_metadata.json"
            )

            metadata.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            metadata.write_text(
                '{"capa": "gold"}',
                encoding="utf-8",
            )

            # Registrar también los dos datasets
            # que esta prueba ya creó anteriormente.
            rutas_manifest = {
                **archivos_adicionales,
                "dim_movie": dim_movie_path,
                "fact_rating": [
                    ratings_dir / "part_000.parquet"
                ],
            }

            # Construir y guardar el manifiesto real.
            manifest = build_gold_manifest(
                gold_dirs=gold_dirs,
                rutas=rutas_manifest,
                metadata_path=metadata,
            )

            save_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifest,
            )

            
            resultado = load_gold_analytics(
                gold_dirs=gold_dirs,
            )

            self.assertEqual(
                len(resultado["dim_movie"]),
                2,
            )

            self.assertEqual(
                len(resultado["fact_rating"]),
                2,
            )

            self.assertEqual(
                set(
                    resultado["dim_movie"][
                        "movie_id"
                    ].tolist()
                ),
                {1, 2},
            )

            self.assertEqual(
                set(
                    resultado["fact_rating"][
                        "user_id"
                    ].tolist()
                ),
                {10, 20},
            )

    def test_gold_ml_prioriza_ratings_particionados(self):

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)
            dimensiones = raiz / "dimensions"
            hechos = raiz / "facts"
            ratings_dir = hechos / "fact_rating"

            dimensiones.mkdir(parents=True)
            ratings_dir.mkdir(parents=True)

            # Dimensión de películas de prueba.
            pd.DataFrame({
                "movie_id": [1, 2],
                "title": ["Pelicula A", "Pelicula B"],
            }).to_parquet(
                dimensiones / "dim_movie.parquet",
                index=False,
            )

            # Primera partición de calificaciones.
            pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "rating": [4.5],
                "rated_at": [pd.Timestamp("2024-01-01")],
            }).to_parquet(
                ratings_dir / "part_000.parquet",
                index=False,
            )

            # Segunda partición.
            pd.DataFrame({
                "user_id": [11],
                "movie_id": [2],
                "rating": [5.0],
                "rated_at": [pd.Timestamp("2024-01-02")],
            }).to_parquet(
                ratings_dir / "part_001.parquet",
                index=False,
            )

            # Archivo antiguo: contiene datos diferentes.
            pd.DataFrame({
                "user_id": [999],
                "movie_id": [1],
                "rating": [1.0],
            }).to_parquet(
                hechos / "fact_rating.parquet",
                index=False,
            )

            # Redirigir la lectura a carpetas temporales.
            with (
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "GOLD_DIMENSIONS_DIR",
                    dimensiones,
                ),
                patch(
                    "src.transformation.gold_analytics_to_ml."
                    "GOLD_FACTS_DIR",
                    hechos,
                ),
            ):

                datasets = load_gold_analytics()

            # Deben recuperarse las dos particiones.
            self.assertEqual(
                len(datasets["fact_rating"]),
                2,
            )

            self.assertEqual(
                set(datasets["fact_rating"]["user_id"]),
                {10, 11},
            )

            # No debe utilizar el archivo antiguo.
            self.assertNotIn(
                999,
                datasets["fact_rating"]["user_id"].tolist(),
            )

            self.assertEqual(
                len(datasets["dim_movie"]),
                2,
            )

    def test_gold_ml_no_duplica_interacciones_colaborativas(self):

        from unittest.mock import patch
        from pathlib import Path

        from src.transformation.gold_analytics_to_ml import (
            run_gold_analytics_to_ml,
        )

        analytics = {
            "dim_movie": pd.DataFrame({
                "movie_id": [1, 2],
                "title": [
                    "Pelicula A",
                    "Pelicula B",
                ],
                "genres": ["Drama", "Comedy"],
            }),
            "fact_rating": pd.DataFrame({
                "user_id": [10, 11],
                "movie_id": [1, 2],
                "rating": [4.5, 5.0],
                "rated_at": pd.to_datetime(
                    ["2024-01-01", "2024-01-02"],
                    utc=True,
                ),
            }),
        }

        particiones = [
            Path("collaborative_interactions/part_000.parquet"),
            Path("collaborative_interactions/part_001.parquet"),
        ]

        with (
            patch(
                "src.transformation.gold_analytics_to_ml."
                "load_gold_analytics",
                return_value=analytics,
            ),
            patch(
                "src.transformation.gold_analytics_to_ml."
                "save_ml_dataset",
                return_value=Path(
                    "content_based_features.parquet"
                ),
            ) as mock_guardado,
            patch(
                "src.transformation.gold_analytics_to_ml."
                "save_partitioned_parquet",
                return_value=particiones,
            ) as mock_particionado,
        ):

            resultado = run_gold_analytics_to_ml()

        # El guardado individual solamente debe
        # utilizarse para Content-Based.
        mock_guardado.assert_called_once()

        self.assertEqual(
            mock_guardado.call_args.args[2],
            "content_based_features.parquet",
        )

        # Las interacciones colaborativas deben
        # guardarse exclusivamente por particiones.
        mock_particionado.assert_called_once()

        self.assertEqual(
            mock_particionado.call_args.args[2],
            "collaborative_interactions",
        )

        # El resultado debe conservar la lista
        # completa de particiones.
        self.assertEqual(
            resultado["collaborative_interactions"],
            particiones,
        )

        # No debe producirse el guardado individual.
        archivos_individuales = [
            llamada.args[2]
            for llamada in mock_guardado.call_args_list
        ]

        self.assertNotIn(
            "collaborative_interactions.parquet",
            archivos_individuales,
        )


if __name__ == "__main__":
    unittest.main()
