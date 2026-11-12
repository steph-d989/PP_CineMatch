
import unittest

import pandas as pd

from src.models.real_hybrid_tuning import (
    tune_real_hybrid,
)


class RealHybridTuningTests(unittest.TestCase):

    def test_optimizar_pesos_con_validation(self):

        train = pd.DataFrame({
            "user_id": [
                10, 10, 10,
                20, 20, 20,
                30, 30, 30,
                40, 40, 40,
            ],
            "movie_id": [
                1, 2, 3,
                1, 3, 4,
                2, 3, 4,
                1, 2, 4,
            ],
            "rating": [
                5.0, 4.0, 4.5,
                4.5, 5.0, 4.0,
                4.0, 5.0, 4.5,
                5.0, 4.5, 4.0,
            ],
        })

        validation = pd.DataFrame({
            "user_id": [10, 20, 30, 40],
            "movie_id": [4, 2, 1, 3],
            "rating": [5.0, 5.0, 5.0, 5.0],
        })

        catalog = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Drama A",
                "Drama B",
                "Comedy A",
                "Comedy B",
            ],
            "genres": [
                "Drama|Mystery",
                "Drama|Mystery",
                "Comedy|Family",
                "Comedy|Family",
            ],
        })

        resultado = tune_real_hybrid(
            train=train,
            validation=validation,
            catalog=catalog,
            n_factors=2,
            k=2,
        )

        self.assertEqual(
            resultado["estado"],
            "selected",
        )

        opciones = resultado[
            "resultados_validation"
        ]

        self.assertEqual(len(opciones), 5)

        mejor = resultado[
            "mejor_configuracion"
        ]

        self.assertAlmostEqual(
            mejor["content_weight"]
            + mejor["collaborative_weight"],
            1.0,
        )

        self.assertEqual(
            mejor["hit_rate_global"],
            max(
                opcion["hit_rate_global"]
                for opcion in opciones
            ),
        )

        self.assertGreaterEqual(
            mejor["hit_rate_global"],
            0.0,
        )

        self.assertLessEqual(
            mejor["hit_rate_global"],
            1.0,
        )

        # Esta función solo utiliza Train
        # y Validation, sin requerir Test.
        self.assertEqual(len(train), 12)
        self.assertEqual(len(validation), 4)


if __name__ == "__main__":
    unittest.main()
