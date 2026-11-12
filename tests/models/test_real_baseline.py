
import unittest

import pandas as pd

from src.models.real_baseline import (
    evaluate_real_baseline,
)


class RealBaselineTests(unittest.TestCase):

    def test_baseline_real_con_validation(self):

        train = pd.DataFrame({
            "user_id": [10, 10, 20, 20, 30, 30],
            "movie_id": [1, 2, 2, 3, 1, 3],
            "rating": [5.0, 4.0, 4.5, 5.0, 4.0, 4.5],
        })

        validation = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [3, 1, 2],
            "rating": [5.0, 5.0, 4.5],
        })

        catalog = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": ["A", "B", "C"],
        })

        resultado = evaluate_real_baseline(
            train=train,
            validation=validation,
            catalog=catalog,
            k=2,
        )

        self.assertEqual(
            resultado["usuarios_relevantes"],
            3,
        )

        self.assertEqual(
            resultado["metrica"],
            "Hit Rate@2",
        )

        self.assertGreaterEqual(
            resultado["hit_rate_global"],
            0.0,
        )

        self.assertLessEqual(
            resultado["hit_rate_global"],
            1.0,
        )

        self.assertAlmostEqual(
            resultado["hit_rate_global"],
            resultado["aciertos"] / 3,
        )

        # Los datos originales no se alteran.
        self.assertEqual(len(train), 6)
        self.assertEqual(len(validation), 3)


if __name__ == "__main__":
    unittest.main()
