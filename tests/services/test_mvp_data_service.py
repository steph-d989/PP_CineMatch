
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.services.mvp_data_service import (
    load_mvp_context,
    get_user_history,
)


class MvpDataServiceTests(unittest.TestCase):

    def test_cargar_contexto_y_historial_usuario(self):

        train = pd.DataFrame({
            "user_id": [10, 10, 20],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.5, 4.0],
            "rated_at": pd.to_datetime([
                "2024-01-01",
                "2024-02-01",
                "2024-01-15",
            ], utc=True),
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

        with tempfile.TemporaryDirectory() as tmp:

            base = Path(tmp)

            train.to_parquet(
                base / "train.parquet",
                index=False,
            )

            catalog.to_parquet(
                base / "catalog_sample.parquet",
                index=False,
            )

            # En esta prueba simulamos únicamente
            # la lectura de los artefactos.
            with (
                patch(
                    "src.services.mvp_data_service."
                    "load_content_based_model",
                    return_value={"content_df": catalog},
                ) as load_content,
                patch(
                    "src.services.mvp_data_service."
                    "load_collaborative_model",
                    return_value={
                        "user_ids": [10, 20],
                        "movie_ids": [1, 2, 3],
                    },
                ) as load_collaborative,
            ):

                context = load_mvp_context(
                    experiment_dir=base,
                    run_id="demo-uuid-test",
                )

            self.assertEqual(
                context["run_id"],
                "demo-uuid-test",
            )

            self.assertEqual(
                len(context["train"]),
                3,
            )

            load_content.assert_called_once_with(
                base
                / "models"
                / "demo-uuid-test"
                / "content_based",
                trusted_source=True,
                expected_run_id="demo-uuid-test",
            )

            load_collaborative.assert_called_once()

            historial = get_user_history(
                user_id=10,
                train=context["train"],
                catalog=context["catalog"],
                n=2,
            )

            # La calificación más reciente
            # debe aparecer primero.
            self.assertEqual(
                historial["movie_id"].tolist(),
                [2, 1],
            )

            self.assertEqual(
                historial["rating"].tolist(),
                [4.5, 5.0],
            )

            self.assertEqual(
                historial["title"].tolist(),
                ["B", "A"],
            )

            self.assertEqual(
                historial["tmdb_id"].tolist(),
                [102, 101],
            )

            self.assertTrue(
                get_user_history(
                    user_id=999,
                    train=context["train"],
                    catalog=context["catalog"],
                ).empty
            )


if __name__ == "__main__":
    unittest.main()
