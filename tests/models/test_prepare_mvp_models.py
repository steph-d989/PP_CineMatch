
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.models.prepare_mvp_models import (
    prepare_and_save_mvp_models,
)


class PrepareMvpModelsTests(unittest.TestCase):

    def test_guardar_y_verificar_modelos_mvp(self):

        train = pd.DataFrame({
            "user_id": [
                10, 10, 10,
                20, 20, 20,
                30, 30, 30,
            ],
            "movie_id": [
                1, 2, 3,
                1, 3, 4,
                2, 3, 4,
            ],
            "rating": [
                5.0, 4.0, 4.5,
                4.5, 5.0, 4.0,
                4.0, 5.0, 4.5,
            ],
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

        with tempfile.TemporaryDirectory() as tmp:

            output_dir = Path(tmp) / "models"

            resultado = prepare_and_save_mvp_models(
                train=train,
                catalog=catalog,
                output_dir=output_dir,
                run_id="a4c8d701-52b6-4f90-8e31-6d2c9f057ab4",
                n_factors=2,
            )

            self.assertEqual(
                resultado["verification"],
                "OK",
            )

            self.assertEqual(
                resultado["content_movies"],
                4,
            )

            self.assertEqual(
                resultado["collaborative_users"],
                3,
            )

            self.assertEqual(
                resultado["collaborative_movies"],
                4,
            )

            self.assertEqual(
                resultado["n_factors"],
                2,
            )

            self.assertTrue(
                (
                    output_dir
                    / "content_based"
                    / "_manifest.json"
                ).is_file()
            )

            self.assertTrue(
                (
                    output_dir
                    / "collaborative"
                    / "_manifest.json"
                ).is_file()
            )

            # No permitir sobrescritura accidental.
            with self.assertRaises(
                FileExistsError
            ):
                prepare_and_save_mvp_models(
                    train=train,
                    catalog=catalog,
                    output_dir=output_dir,
                    run_id="a4c8d701-52b6-4f90-8e31-6d2c9f057ab4",
                    n_factors=2,
                )
