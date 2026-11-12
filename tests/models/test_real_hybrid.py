
import unittest

import pandas as pd

from src.models.real_hybrid import (
    evaluate_real_hybrid,
)


class RealHybridTests(unittest.TestCase):

    def test_hybrid_real_con_validation(self):

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

        resultado = evaluate_real_hybrid(
            train=train,
            validation=validation,
            catalog=catalog,
            n_factors=2,
            k_values=(2,),
        )

        self.assertEqual(
            resultado["total_users"],
            4,
        )

        self.assertEqual(
            resultado["total_movies_content"],
            4,
        )

        self.assertEqual(
            resultado["total_movies_collaborative"],
            4,
        )

        self.assertEqual(
            resultado["n_factors"],
            2,
        )

        evaluacion = resultado[
            "evaluaciones"
        ][2]

        self.assertEqual(
            evaluacion["metrica"],
            "Hit Rate@2",
        )

        self.assertEqual(
            evaluacion["usuarios_relevantes"],
            4,
        )

        self.assertGreaterEqual(
            evaluacion["hit_rate_global"],
            0.0,
        )

        self.assertLessEqual(
            evaluacion["hit_rate_global"],
            1.0,
        )

        self.assertAlmostEqual(
            evaluacion["hit_rate_global"],
            evaluacion["aciertos"] / 4,
        )

        self.assertEqual(len(train), 12)
        self.assertEqual(len(validation), 4)


if __name__ == "__main__":
    unittest.main()
