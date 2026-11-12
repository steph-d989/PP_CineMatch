
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import pandas as pd

from src.models.baseline import (
    build_baseline_scores,
    recommend_baseline_top_n,
    build_training_movie_metrics,
    recommend_baseline_from_gold,
)

from src.utils.gold_runs import (
    build_gold_run_directories,
    build_gold_manifest,
    save_gold_manifest,
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
        
    def test_baseline_lee_gold_versionado(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            dim_movie_path = (
                gold_dirs["dimensions"]
                / "dim_movie.parquet"
            )

            metrics_path = (
                gold_dirs["metrics"]
                / "movie_metrics.parquet"
            )

            dim_movie_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            metrics_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame({
                "movie_id": [1, 2],
                "title": ["Pelicula A", "Pelicula B"],
            }).to_parquet(
                dim_movie_path,
                index=False,
            )

            pd.DataFrame({
                "movie_id": [1, 2],
                "rating_count": [100, 10],
                "rating_mean": [4.0, 5.0],
            }).to_parquet(
                metrics_path,
                index=False,
            )

            # Crear el resto de datasets obligatorios
            # para un manifiesto Gold completo.
            rutas = {
                "dim_movie": dim_movie_path,
                "movie_metrics": metrics_path,
            }

            adicionales = {
                "dim_user": (
                    gold_dirs["dimensions"] / "dim_user.parquet"
                ),
                "dim_platform": (
                    gold_dirs["dimensions"] / "dim_platform.parquet"
                ),
                "dim_date": (
                    gold_dirs["dimensions"] / "dim_date.parquet"
                ),
                "fact_rating": [
                    gold_dirs["facts"]
                    / "fact_rating"
                    / "part_000.parquet"
                ],
                "fact_availability": (
                    gold_dirs["facts"]
                    / "fact_availability.parquet"
                ),
                "user_metrics": (
                    gold_dirs["metrics"] / "user_metrics.parquet"
                ),
                "availability_unlinked": (
                    gold_dirs["quality"]
                    / "availability_unlinked.parquet"
                ),
            }

            for referencia in adicionales.values():

                archivos = (
                    referencia
                    if isinstance(referencia, list)
                    else [referencia]
                )

                for ruta in archivos:
                    ruta.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    pd.DataFrame({
                        "id": [1],
                    }).to_parquet(
                        ruta,
                        index=False,
                    )

            rutas.update(adicionales)

            metadata_path = (
                gold_dirs["metrics"] / "_metadata.json"
            )

            metadata_path.write_text(
                json.dumps({"capa": "gold"}),
                encoding="utf-8",
            )

            manifest = build_gold_manifest(
                gold_dirs=gold_dirs,
                rutas=rutas,
                metadata_path=metadata_path,
            )

            save_gold_manifest(
                gold_dirs=gold_dirs,
                manifest=manifest,
            )

            # Ranking desde los archivos físicos.
            resultado = recommend_baseline_from_gold(
                gold_dirs=gold_dirs,
                n=2,
            )

            self.assertEqual(len(resultado), 2)

            self.assertEqual(
                set(resultado["movie_id"]),
                {1, 2},
            )

            self.assertIn(
                "weighted_score",
                resultado.columns,
            )

            # Corromper una métrica después
            # de guardar el manifiesto.
            metrics_path.write_bytes(b"alterado")

            with self.assertRaises(ValueError):
                recommend_baseline_from_gold(
                    gold_dirs=gold_dirs,
                    n=2,
                )

    
if __name__ == "__main__":
    unittest.main()
