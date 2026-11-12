
import unittest

import pandas as pd

from src.models.real_collaborative import (
    evaluate_real_collaborative,
)


class RealCollaborativeTests(unittest.TestCase):

    def test_collaborative_real_con_validation(self):

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

        resultado = evaluate_real_collaborative(
            train=train,
            validation=validation,
            n_factors=2,
            k_values=(2,),
        )

        self.assertEqual(
            resultado["total_users"],
            4,
        )

        self.assertEqual(
            resultado["total_movies"],
            4,
        )

        self.assertEqual(
            resultado["total_interactions"],
            12,
        )

        self.assertEqual(
            resultado["matrix_shape"],
            (4, 4),
        )

        self.assertEqual(
            resultado["matrix_nnz"],
            12,
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

        # No se modifican Train ni Validation.
        self.assertEqual(len(train), 12)
        self.assertEqual(len(validation), 4)


if __name__ == "__main__":
    unittest.main()
