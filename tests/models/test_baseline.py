
import unittest
import pandas as pd

from src.models.baseline import (
    build_baseline_scores,
    recommend_baseline_top_n,
    build_training_movie_metrics,
)


class BaselineTests(unittest.TestCase):

    def test_baseline_calcula_puntuacion_ponderada(self):

        # Catálogo Gold Analytics.
        dim_movie = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        # Métricas de calificaciones.
        movie_metrics = pd.DataFrame({
            "movie_id": [1, 2],
            "rating_count": [100, 10],
            "rating_mean": [4.0, 5.0],
        })

        resultado = build_baseline_scores(
            dim_movie=dim_movie,
            movie_metrics=movie_metrics,
            quantile=0.90,
        )

        self.assertEqual(
            len(resultado),
            3,
        )

        # El promedio global debe considerar
        # la cantidad de calificaciones.
        C_esperado = (
            100 * 4.0 + 10 * 5.0
        ) / 110

        self.assertAlmostEqual(
            resultado.attrs["global_mean"],
            C_esperado,
        )

        # Comprobar la fórmula ponderada
        # para la película A.
        m = resultado.attrs["min_votes_parameter"]

        score_esperado = (
            (100 / (100 + m)) * 4.0
            + (m / (100 + m)) * C_esperado
        )

        pelicula_a = resultado.loc[
            resultado["movie_id"] == 1
        ].iloc[0]

        self.assertAlmostEqual(
            pelicula_a["weighted_score"],
            score_esperado,
        )

        # Una película sin calificaciones
        # no debe recibir una puntuación inventada.
        pelicula_c = resultado.loc[
            resultado["movie_id"] == 3
        ].iloc[0]

        self.assertTrue(
            pd.isna(
                pelicula_c["weighted_score"]
            )
        )

        # Cada película aparece una sola vez.
        self.assertFalse(
            resultado["movie_id"].duplicated().any()
        )

    def test_baseline_recomienda_top_n(self):

        scores = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
                "Pelicula D",
            ],
            "rating_count": [100, 20, 10, 0],
            "weighted_score": [4.2, 4.5, 4.0, float("nan")],
        })

        resultado = recommend_baseline_top_n(
            baseline_scores=scores,
            n=2,
            min_ratings=10,
        )

        self.assertEqual(
            len(resultado),
            2,
        )

        self.assertEqual(
            resultado["movie_id"].tolist(),
            [2, 1],
        )

        # La película sin puntuación no aparece.
        self.assertNotIn(
            4,
            resultado["movie_id"].tolist(),
        )

        # Filtrar películas con pocos ratings.
        filtrado = recommend_baseline_top_n(
            baseline_scores=scores,
            n=10,
            min_ratings=50,
        )

        self.assertEqual(
            filtrado["movie_id"].tolist(),
            [1],
        )

        # Rechazar parámetros inválidos.
        with self.assertRaises(ValueError):
            recommend_baseline_top_n(
                scores,
                n=0,
            )

        with self.assertRaises(ValueError):
            recommend_baseline_top_n(
                scores,
                min_ratings=0,
            )

    def test_baseline_metricas_solo_entrenamiento(self):

        # Calificaciones disponibles para entrenar.
        train_ratings = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [1, 1, 2],
            "rating": [4.0, 5.0, 3.0],
        })

        # Esta calificación está reservada
        # exclusivamente para evaluación.
        test_ratings = pd.DataFrame({
            "user_id": [40],
            "movie_id": [1],
            "rating": [1.0],
        })

        metricas = build_training_movie_metrics(
            train_ratings
        )

        pelicula_a = metricas.set_index(
            "movie_id"
        ).loc[1]

        self.assertEqual(
            int(pelicula_a["rating_count"]),
            2,
        )

        self.assertAlmostEqual(
            float(pelicula_a["rating_mean"]),
            4.5,
        )

        # No debe obtener el promedio que
        # resultaría al incluir el conjunto test.
        promedio_con_fuga = (
            pd.concat(
                [train_ratings, test_ratings]
            )
            .query("movie_id == 1")["rating"]
            .mean()
        )

        self.assertNotAlmostEqual(
            float(pelicula_a["rating_mean"]),
            float(promedio_con_fuga),
        )

        # La función conserva los datos originales.
        self.assertEqual(
            len(train_ratings),
            3,
        )

        # Una entrada vacía debe rechazarse.
        with self.assertRaises(ValueError):
            build_training_movie_metrics(
                train_ratings.iloc[0:0]
            )

if __name__ == "__main__":
    unittest.main()
