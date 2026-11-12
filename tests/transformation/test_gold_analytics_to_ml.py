
import unittest
import pandas as pd

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.transformation.gold_analytics_to_ml import (
    load_gold_analytics,
)


class GoldAnalyticsToMLTests(unittest.TestCase):

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
