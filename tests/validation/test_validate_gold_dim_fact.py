
import io
import runpy
import unittest

from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd


class ValidateGoldDimFactTests(unittest.TestCase):

    def test_validador_gold_lee_ratings_particionados(self):

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)
            dimensions = raiz / "dimensions"
            facts = raiz / "facts"
            ratings = facts / "fact_rating"

            dimensions.mkdir(parents=True)
            ratings.mkdir(parents=True)

            # Dimensión de películas.
            pd.DataFrame({
                "movie_id": [1, 2],
            }).to_parquet(
                dimensions / "dim_movie.parquet",
                index=False,
            )

            # Dimensión de plataformas.
            pd.DataFrame({
                "provider_id": [10],
                "provider_name": ["Plataforma A"],
            }).to_parquet(
                dimensions / "dim_platform.parquet",
                index=False,
            )

            # Dimensión de fechas.
            pd.DataFrame({
                "date_key": [20240101, 20240102],
            }).to_parquet(
                dimensions / "dim_date.parquet",
                index=False,
            )

            # Primera partición de ratings.
            pd.DataFrame({
                "movie_id": [1],
                "date_key": [20240101],
            }).to_parquet(
                ratings / "part_000.parquet",
                index=False,
            )

            # Segunda partición de ratings.
            pd.DataFrame({
                "movie_id": [2],
                "date_key": [20240102],
            }).to_parquet(
                ratings / "part_001.parquet",
                index=False,
            )

            # Archivo antiguo con un ID distinto.
            pd.DataFrame({
                "movie_id": [999],
                "date_key": [20240101],
            }).to_parquet(
                facts / "fact_rating.parquet",
                index=False,
            )

            # Hechos de disponibilidad.
            pd.DataFrame({
                "movie_id": [1],
                "provider_id": [10],
                "provider_name": ["Plataforma A"],
                "date_key": [20240101],
            }).to_parquet(
                facts / "fact_availability.parquet",
                index=False,
            )

            salida = io.StringIO()

            # Ejecutar el validador con rutas temporales.
            with (
                patch(
                    "src.utils.paths.GOLD_DIMENSIONS_DIR",
                    dimensions,
                ),
                patch(
                    "src.utils.paths.GOLD_FACTS_DIR",
                    facts,
                ),
                redirect_stdout(salida),
            ):

                resultado = runpy.run_module(
                    "src.validation.validate_gold_dim_fact",
                    run_name="__main__",
                )

            # Debe haber leído las dos particiones.
            fact_rating = resultado["fact_rating"]

            self.assertEqual(
                len(fact_rating),
                2,
            )

            self.assertEqual(
                set(fact_rating["movie_id"]),
                {1, 2},
            )

            # No utilizar el archivo antiguo.
            self.assertNotIn(
                999,
                fact_rating["movie_id"].tolist(),
            )

            # Las relaciones válidas deben conservarse.
            self.assertEqual(
                len(resultado["movies_sin_dim"]),
                0,
            )


if __name__ == "__main__":
    unittest.main()
