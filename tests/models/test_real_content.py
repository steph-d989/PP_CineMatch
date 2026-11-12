
import unittest

import pandas as pd

from src.models.real_content import (
    evaluate_real_content,
)


class RealContentTests(unittest.TestCase):

    def test_content_real_con_validation(self):

        train = pd.DataFrame({
            "user_id": [
                10, 10,
                20, 20,
                30, 30,
            ],
            "movie_id": [
                1, 2,
                2, 3,
                3, 4,
            ],
            "rating": [
                5.0, 4.0,
                4.5, 5.0,
                5.0, 4.0,
            ],
        })

        validation = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [3, 4, 1],
            "rating": [5.0, 5.0, 4.5],
        })

        catalog = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
                "Comedia B",
            ],
            "genres": [
                "Drama|Mystery",
                "Drama|Mystery",
                "Comedy|Family",
                "Comedy|Family",
            ],
        })

        resultado = evaluate_real_content(
            train=train,
            validation=validation,
            catalog=catalog,
            k_values=(2,),
        )

        self.assertEqual(
            resultado["total_movies"],
            4,
        )

        self.assertGreater(
            resultado["total_features"],
            0,
        )

        evaluacion = resultado[
            "evaluaciones"
        ][2]

        self.assertEqual(
            evaluacion["metrica"],
            "Hit Rate@2",
        )

        self.assertGreaterEqual(
            evaluacion["hit_rate_global"],
            0.0,
        )

        self.assertLessEqual(
            evaluacion["hit_rate_global"],
            1.0,
        )

        self.assertLessEqual(
            evaluacion["aciertos"],
            evaluacion["usuarios_relevantes"],
        )

        # Confirmar que no modificamos
        # las columnas originales.
        self.assertEqual(
            catalog.loc[0, "genres"],
            "Drama|Mystery",
        )

        self.assertEqual(
            len(train),
            6,
        )

        self.assertEqual(
            len(validation),
            3,
        )


if __name__ == "__main__":
    unittest.main()
