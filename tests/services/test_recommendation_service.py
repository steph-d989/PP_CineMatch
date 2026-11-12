
import unittest
from unittest.mock import patch

import pandas as pd

from src.services.recommendation_service import (
    recommend_movies_for_user,
)


class RecommendationServiceTests(unittest.TestCase):

    def test_servicio_recomendaciones_mvp(self):

        train = pd.DataFrame({
            "user_id": [10, 10, 20],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.0, 4.5],
        })

        catalog = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "genres": [
                "Drama",
                "Comedy",
                "Action",
                "Sci-Fi",
            ],
            "tmdb_id": [101, 102, 103, 104],
        })

        # El modelo simulado devuelve una película
        # ya vista y dos películas candidatas.
        ranking = pd.DataFrame({
            "movie_id": [1, 4, 3],
            "collaborative_score": [
                0.9, 0.8, 0.7,
            ],
        })

        with patch(
            "src.services.recommendation_service."
            "recommend_collaborative_top_n",
            return_value=ranking,
        ) as mock_model:

            resultado = recommend_movies_for_user(
                user_id=10,
                algorithm="collaborative",
                n=3,
                train=train,
                catalog=catalog,
                content_model={},
                collaborative_model={},
            )

        self.assertEqual(
            resultado["movie_id"].tolist(),
            [4, 3],
        )

        self.assertEqual(
            resultado["title"].tolist(),
            ["D", "C"],
        )

        self.assertEqual(
            resultado["tmdb_id"].tolist(),
            [104, 103],
        )

        self.assertEqual(
            resultado["algorithm"].tolist(),
            ["collaborative", "collaborative"],
        )

        # La película 1 ya fue calificada.
        self.assertNotIn(
            1,
            resultado["movie_id"].tolist(),
        )

        self.assertTrue(mock_model.called)

        # También comprobar el camino Content-Based.
        content_ranking = pd.DataFrame({
            "movie_id": [4, 3],
            "title": ["D", "C"],
            "content_score": [0.8, 0.6],
        })

        with patch(
            "src.services.recommendation_service."
            "recommend_content_for_user",
            return_value=content_ranking,
        ):

            content_result = recommend_movies_for_user(
                user_id=10,
                algorithm="content",
                n=2,
                train=train,
                catalog=catalog,
                content_model={
                    "content_df": pd.DataFrame(),
                    "tfidf_matrix": None,
                },
                collaborative_model={},
            )

        self.assertEqual(
            content_result["movie_id"].tolist(),
            [4, 3],
        )

        self.assertEqual(
            content_result["algorithm"].tolist(),
            ["content", "content"],
        )

        # No se modifican Train ni Catálogo.
        self.assertEqual(len(train), 3)
        self.assertEqual(len(catalog), 4)


if __name__ == "__main__":
    unittest.main()
