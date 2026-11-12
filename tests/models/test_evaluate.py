import unittest
import pandas as pd

from src.models.evaluate import *

from src.models.content_based import (
    build_tfidf_features,
)


class EvaluateTests(unittest.TestCase):

    def test_division_temporal_por_usuario(self):

        interactions = pd.DataFrame({
            "user_id": [10, 10, 10, 20, 20, 30],
            "movie_id": [1, 2, 3, 1, 4, 2],
            "rating": [4.0, 5.0, 4.5, 3.0, 4.0, 5.0],
            "rated_at": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-01-15",
                    "2024-04-01",
                    "2024-05-01",
                ],
                utc=True,
            ),
        })

        train, test = temporal_train_test_split(
            interactions
        )

        # Se conservan todas las interacciones.
        self.assertEqual(
            len(train) + len(test),
            6,
        )

        # Solo usuarios con dos o más
        # valoraciones tienen registros de prueba.
        self.assertEqual(
            set(test["user_id"]),
            {10, 20},
        )

        # Última película de cada usuario.
        peliculas_test = dict(
            zip(test["user_id"], test["movie_id"])
        )

        self.assertEqual(
            peliculas_test,
            {10: 3, 20: 4},
        )

        # Usuario con una sola interacción:
        # permanece en entrenamiento.
        self.assertIn(
            30,
            train["user_id"].tolist(),
        )

        # Ninguna interacción se repite.
        pares_train = set(
            zip(train["user_id"], train["movie_id"])
        )

        pares_test = set(
            zip(test["user_id"], test["movie_id"])
        )

        self.assertTrue(
            pares_train.isdisjoint(pares_test)
        )

        # Rechazar usuarios sin historial suficiente
        # para obtener un conjunto de prueba.
        solo_una = interactions.loc[
            interactions["user_id"] == 30
        ]

        with self.assertRaises(ValueError):
            temporal_train_test_split(solo_una)

    def test_baseline_hit_rate_sin_fuga(self):

        dim_movie = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        train = pd.DataFrame({
            "user_id": [10, 20, 30, 40],
            "movie_id": [1, 2, 1, 3],
            "rating": [4.0, 5.0, 4.0, 3.0],
        })

        test = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [2, 1],
            "rating": [5.0, 4.5],
        })

        resultado = evaluate_baseline_hit_rate(
            train=train,
            test=test,
            dim_movie=dim_movie,
            k=1,
        )

        self.assertEqual(
            resultado["metrica"],
            "Hit Rate@1",
        )

        self.assertEqual(
            resultado["usuarios_evaluables"],
            2,
        )

        self.assertEqual(
            resultado["aciertos"],
            2,
        )

        self.assertAlmostEqual(
            resultado["hit_rate"],
            1.0,
        )

        # Una película ausente de entrenamiento
        # no tiene puntuación baseline.
        test_cold_start = pd.DataFrame({
            "user_id": [10],
            "movie_id": [99],
            "rating": [5.0],
        })

        cold_start = evaluate_baseline_hit_rate(
            train=train,
            test=test_cold_start,
            dim_movie=dim_movie,
            k=1,
        )

        self.assertEqual(
            cold_start["usuarios_evaluables"],
            0,
        )

        self.assertEqual(
            cold_start["usuarios_excluidos"],
            1,
        )

        self.assertIsNone(
            cold_start["hit_rate"]
        )

    def test_baseline_reporta_cobertura_evaluacion(self):

        dim_movie = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        train = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [1, 2, 1],
            "rating": [4.0, 5.0, 4.5],
        })

        test = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [2, 3],
            "rating": [5.0, 5.0],
        })

        resultado = evaluate_baseline_hit_rate(
            train=train,
            test=test,
            dim_movie=dim_movie,
            k=2,
        )

        # Dos usuarios tienen películas relevantes
        # reservadas en el conjunto test.
        self.assertEqual(
            resultado["usuarios_relevantes"],
            2,
        )

        # Solo la película 2 recibió ratings
        # durante entrenamiento.
        self.assertEqual(
            resultado["usuarios_evaluables"],
            1,
        )

        self.assertEqual(
            resultado["usuarios_excluidos"],
            1,
        )

        # El usuario 10 recibe la película 2.
        self.assertEqual(
            resultado["aciertos"],
            1,
        )

        self.assertAlmostEqual(
            resultado["hit_rate"],
            1.0,
        )

        self.assertAlmostEqual(
            resultado["hit_rate_global"],
            0.5,
        )

        self.assertAlmostEqual(
            resultado["cobertura_evaluacion"],
            0.5,
        )
    
    def test_baseline_rechaza_solapamiento_train_test(self):

        dim_movie = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        train = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [1, 2, 3],
            "rating": [4.0, 5.0, 3.5],
        })

        # El usuario 10 ya calificó la película 1
        # en entrenamiento.
        test_solapado = pd.DataFrame({
            "user_id": [10],
            "movie_id": [1],
            "rating": [4.5],
        })

        with self.assertRaisesRegex(
            ValueError,
            "solapamiento usuario-película",
        ):
            evaluate_baseline_hit_rate(
                train=train,
                test=test_solapado,
                dim_movie=dim_movie,
                k=2,
            )

        # El mismo usuario puede aparecer en ambos
        # conjuntos, siempre que evalúe otra película.
        test_valido = pd.DataFrame({
            "user_id": [10],
            "movie_id": [2],
            "rating": [5.0],
        })

        resultado = evaluate_baseline_hit_rate(
            train=train,
            test=test_valido,
            dim_movie=dim_movie,
            k=2,
        )

        self.assertEqual(
            resultado["usuarios_evaluables"],
            1,
        )

        self.assertEqual(
            resultado["aciertos"],
            1,
        )

        # También deben rechazarse duplicados
        # dentro del propio entrenamiento.
        train_duplicado = pd.concat(
            [train, train.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Train contiene interacciones",
        ):
            evaluate_baseline_hit_rate(
                train=train_duplicado,
                test=test_valido,
                dim_movie=dim_movie,
            )

    def test_content_hit_rate_sin_fuga_interacciones(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3, 4, 5],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
                "Comedia B",
                "Aventura",
            ],
            "content_soup": [
                "drama mystery detective",
                "drama mystery detective",
                "comedy family humor",
                "comedy family humor",
                "adventure space fantasy",
            ],
        })

        _, matriz = build_tfidf_features(
            peliculas,
            min_df=1,
        )

        train = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [1, 3],
            "rating": [5.0, 5.0],
        })

        test = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [2, 4, 5],
            "rating": [5.0, 4.5, 5.0],
        })

        resultado = evaluate_content_hit_rate(
            train=train,
            test=test,
            content_df=peliculas,
            tfidf_matrix=matriz,
            k=1,
        )

        self.assertEqual(
            resultado["usuarios_relevantes"],
            3,
        )

        # El usuario 30 no tiene historial Train.
        self.assertEqual(
            resultado["usuarios_excluidos"],
            1,
        )

        self.assertEqual(
            resultado["usuarios_evaluables"],
            2,
        )

        # Usuario 10 -> Drama B.
        # Usuario 20 -> Comedia B.
        self.assertEqual(
            resultado["aciertos"],
            2,
        )

        self.assertAlmostEqual(
            resultado["hit_rate"],
            1.0,
        )

        self.assertAlmostEqual(
            resultado["hit_rate_global"],
            2 / 3,
        )

        self.assertAlmostEqual(
            resultado["cobertura_evaluacion"],
            2 / 3,
        )

        # No permitir que la misma interacción
        # aparezca en Train y Test.
        test_invalido = pd.DataFrame({
            "user_id": [10],
            "movie_id": [1],
            "rating": [5.0],
        })

        with self.assertRaisesRegex(
            ValueError,
            "solapamiento usuario-película",
        ):
            evaluate_content_hit_rate(
                train=train,
                test=test_invalido,
                content_df=peliculas,
                tfidf_matrix=matriz,
            )

    def test_content_based_carga_gold_ml_versionado(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory

        import pandas as pd

        from src.utils.gold_runs import (
            build_gold_run_directories,
        )

        from src.models.content_based import (
            load_content_features_from_gold_ml,
            fit_content_based_model,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            gold_dirs = build_gold_run_directories(
                gold_root=Path(temporal) / "gold",
                run_id=run_id,
            )

            ruta = (
                gold_dirs["ml_content"]
                / "content_based_features.parquet"
            )

            ruta.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            peliculas = pd.DataFrame({
                "movie_id": [1, 2, 3],
                "title": [
                    "Drama A",
                    "Drama B",
                    "Comedia A",
                ],
                "genres": [
                    ["Drama", "Mystery"],
                    ["Drama", "Mystery"],
                    ["Comedy", "Family"],
                ],
                "keywords": [
                    ["Detective"],
                    ["Investigation"],
                    ["Humor"],
                ],
                "overview": [
                    "A detective mystery",
                    "A mystery investigation",
                    "A family comedy",
                ],
            })

            peliculas.to_parquet(
                ruta,
                index=False,
            )

            # Cargar desde Gold ML versionado.
            cargadas = (
                load_content_features_from_gold_ml(
                    gold_dirs=gold_dirs,
                )
            )

            self.assertEqual(
                len(cargadas),
                3,
            )

            self.assertEqual(
                cargadas["movie_id"].tolist(),
                [1, 2, 3],
            )

            # Comprobar que los datos permiten
            # entrenar el modelo real.
            modelo = fit_content_based_model(
                content_features=cargadas,
                min_df=1,
            )

            self.assertEqual(
                modelo["total_movies"],
                3,
            )

            self.assertGreater(
                modelo["total_features"],
                0,
            )

            # No aceptar identificadores duplicados.
            duplicadas = peliculas.copy()
            duplicadas.loc[2, "movie_id"] = 1

            duplicadas.to_parquet(
                ruta,
                index=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicados",
            ):
                load_content_features_from_gold_ml(
                    gold_dirs=gold_dirs,
                )


if __name__ == "__main__":
    unittest.main()
