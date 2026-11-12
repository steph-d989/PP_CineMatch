
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.real_data import *


class RealDataTests(unittest.TestCase):

    
    def test_split_real_diagnostico_cold_start(self):

        muestra = pd.DataFrame({
            "user_id": [10, 10, 10, 10, 20, 20, 20],
            "movie_id": [1, 2, 3, 4, 3, 1, 2],
            "rating": [5.0, 4.0, 4.5, 5.0, 4.0, 5.0, 4.5],
            "rated_at": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ], utc=True),
        })

        catalog = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
        })

        resultado = prepare_real_evaluation_split(
            sample_ratings=muestra,
            catalog=catalog,
        )

        train = resultado["train"]
        validation = resultado["validation"]
        test = resultado["test"]

        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)

        diagnosticos = resultado["diagnosticos"]

        # Películas 1, 2 y 3 existen en Train.
        # Película 4 es nueva para el modelo.
        self.assertEqual(
            set(train["movie_id"]),
            {1, 2, 3},
        )

        self.assertEqual(
            diagnosticos["validation"][
                "peliculas_nuevas_interacciones"
            ],
            0,
        )

        self.assertEqual(
            diagnosticos["test"][
                "peliculas_nuevas_interacciones"
            ],
            1,
        )

        self.assertAlmostEqual(
            diagnosticos["test"][
                "cobertura_peliculas_train"
            ],
            0.5,
        )

        self.assertEqual(
            diagnosticos["test"][
                "interacciones_relevantes"
            ],
            2,
        )

        self.assertEqual(
            diagnosticos["test"][
                "usuarios_nuevos_interacciones"
            ],
            0,
        )

        # No se deben eliminar las películas nuevas.
        self.assertEqual(
            set(test["movie_id"]),
            {2, 4},
        )

    
    def test_muestra_movielens_reproducible(self):

        origen = pd.DataFrame({
            "userId": [
                1, 1, 1,
                2, 2, 2,
                3,
            ],
            "movieId": [
                10, 11, 12,
                10, 13, 14,
                15,
            ],
            "rating": [
                5.0, 4.0, 3.5,
                4.5, 5.0, 3.0,
                4.0,
            ],
            "timestamp": [
                1000000000,
                1000000100,
                1000000200,
                1000000000,
                1000000100,
                1000000200,
                1000000000,
            ],
        })

        with tempfile.TemporaryDirectory() as tmp:

            archivo = Path(tmp) / "ratings.csv"
            origen.to_csv(archivo, index=False)

            argumentos = {
                "ratings_path": archivo,
                "sample_users": 3,
                "min_ratings": 3,
                "seed": 42,
                "chunksize": 2,
            }

            muestra_1 = sample_movielens_ratings(
                **argumentos
            )

            muestra_2 = sample_movielens_ratings(
                **argumentos
            )

            pd.testing.assert_frame_equal(
                muestra_1,
                muestra_2,
            )

            # El usuario 3 no alcanza
            # el mínimo de interacciones.
            self.assertEqual(
                set(muestra_1["user_id"]),
                {1, 2},
            )

            self.assertEqual(
                len(muestra_1),
                6,
            )

            self.assertEqual(
                list(muestra_1.columns),
                [
                    "user_id",
                    "movie_id",
                    "rating",
                    "rated_at",
                ],
            )

            self.assertTrue(
                pd.api.types.is_datetime64_any_dtype(
                    muestra_1["rated_at"]
                )
            )

            self.assertFalse(
                muestra_1.duplicated(
                    ["user_id", "movie_id"]
                ).any()
            )

    def test_catalogo_muestra_movielens_tmdb(self):

        muestra = pd.DataFrame({
            "user_id": [10, 10, 20],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.0, 4.5],
        })

        movies = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
            "genres": [
                "Drama",
                "Comedy",
                "Action",
                "Drama",
            ],
        })

        links = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "imdbId": [
                "00123",
                "00456",
                "00789",
                "00111",
            ],
            "tmdbId": [
                100.0,
                200.0,
                None,
                400.0,
            ],
        })

        with tempfile.TemporaryDirectory() as tmp:

            movies_path = Path(tmp) / "movies.csv"
            links_path = Path(tmp) / "links.csv"

            movies.to_csv(movies_path, index=False)
            links.to_csv(links_path, index=False)

            catalog = build_movielens_sample_catalog(
                sample_ratings=muestra,
                movies_path=movies_path,
                links_path=links_path,
            )

            # Solo incluir las películas de la
            # muestra, no todo MovieLens.
            self.assertEqual(
                catalog["movie_id"].tolist(),
                [1, 2, 3],
            )

            self.assertEqual(len(catalog), 3)

            # Conservar identificadores IMDb
            # incluyendo ceros iniciales.
            self.assertEqual(
                catalog.loc[
                    catalog["movie_id"] == 1,
                    "imdb_id",
                ].iloc[0],
                "00123",
            )

            # TMDb no es obligatorio.
            self.assertEqual(
                catalog["tmdb_id"].notna().sum(),
                2,
            )

            self.assertTrue(
                pd.isna(
                    catalog.loc[
                        catalog["movie_id"] == 3,
                        "tmdb_id",
                    ].iloc[0]
                )
            )

            # No debe perder películas.
            self.assertEqual(
                catalog["movie_id"].nunique(),
                muestra["movie_id"].nunique(),
            )

            # Una película desconocida debe
            # provocar un error explícito.
            muestra_invalida = pd.DataFrame({
                "movie_id": [1, 999],
            })

            with self.assertRaisesRegex(
                ValueError,
                "no aparecen en movies.csv",
            ):
                build_movielens_sample_catalog(
                    sample_ratings=muestra_invalida,
                    movies_path=movies_path,
                    links_path=links_path,
                )


if __name__ == "__main__":
    unittest.main()
