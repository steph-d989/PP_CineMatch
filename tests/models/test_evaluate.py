import unittest
import pandas as pd
import numpy as np


from unittest.mock import patch
from math import log2

from src.models.evaluate import *

from src.models.content_based import (
    build_tfidf_features,
)


class EvaluateTests(unittest.TestCase):

    
    def test_seleccion_pesos_hibridos_solo_validation(self):

        train = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [1, 2],
            "rating": [5.0, 4.5],
        })

        validation = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [3, 4],
            "rating": [5.0, 5.0],
        })

        # Resultados controlados para verificar
        # la selección, no el entrenamiento.
        resultados = [
            {
                "hit_rate_global": 0.50,
                "cobertura_recomendaciones": 1.0,
                "aciertos": 1,
            },
            {
                "hit_rate_global": 1.00,
                "cobertura_recomendaciones": 1.0,
                "aciertos": 2,
            },
            {
                "hit_rate_global": 0.75,
                "cobertura_recomendaciones": 1.0,
                "aciertos": 1,
            },
        ]

        with patch(
            "src.models.evaluate.evaluate_hybrid_hit_rate",
            side_effect=resultados,
        ) as mock_evaluador:

            seleccion = select_hybrid_weights(
                train=train,
                validation=validation,
                content_model={},
                collaborative_model={},
                k=2,
            )

        self.assertEqual(
            seleccion["estado"],
            "selected",
        )

        self.assertEqual(
            len(seleccion["resultados_validation"]),
            3,
        )

        # En este ejemplo la combinación 50/50
        # consigue la mejor puntuación.
        mejor = seleccion["mejor_configuracion"]

        self.assertAlmostEqual(
            mejor["content_weight"],
            0.5,
        )

        self.assertAlmostEqual(
            mejor["collaborative_weight"],
            0.5,
        )

        self.assertAlmostEqual(
            mejor["hit_rate_global"],
            1.0,
        )

        # Todas las evaluaciones deben utilizar
        # exclusivamente Validation como objetivo.
        self.assertEqual(
            mock_evaluador.call_count,
            3,
        )

        for llamada in mock_evaluador.call_args_list:

            pd.testing.assert_frame_equal(
                llamada.kwargs["train"],
                train,
            )

            pd.testing.assert_frame_equal(
                llamada.kwargs["test"],
                validation,
            )

        # No se modifica ningún conjunto original.
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 2)

    def test_division_temporal_train_validation_test(self):

        interactions = pd.DataFrame({
            "user_id": [
                10, 10, 10, 10,
                20, 20, 20,
                30, 30,
            ],
            "movie_id": [
                1, 2, 3, 4,
                1, 3, 5,
                2, 4,
            ],
            "rating": [
                5.0, 4.0, 4.5, 5.0,
                4.0, 5.0, 4.5,
                3.0, 4.0,
            ],
            "rated_at": pd.to_datetime([
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-01-15",
                "2024-02-15",
                "2024-03-15",
                "2024-01-10",
                "2024-02-10",
            ], utc=True),
        })

        train, validation, test = (
            temporal_train_validation_test_split(
                interactions
            )
        )

        # Los nueve registros se conservan.
        self.assertEqual(
            len(train) + len(validation) + len(test),
            9,
        )

        self.assertEqual(len(train), 5)
        self.assertEqual(len(validation), 2)
        self.assertEqual(len(test), 2)

        # Usuario 10:
        # películas 1 y 2 -> Train
        # película 3 -> Validation
        # película 4 -> Test
        self.assertEqual(
            validation.loc[
                validation["user_id"] == 10,
                "movie_id",
            ].tolist(),
            [3],
        )

        self.assertEqual(
            test.loc[
                test["user_id"] == 10,
                "movie_id",
            ].tolist(),
            [4],
        )

        # Usuario 20:
        # película 1 -> Train
        # película 3 -> Validation
        # película 5 -> Test
        self.assertEqual(
            validation.loc[
                validation["user_id"] == 20,
                "movie_id",
            ].tolist(),
            [3],
        )

        self.assertEqual(
            test.loc[
                test["user_id"] == 20,
                "movie_id",
            ].tolist(),
            [5],
        )

        # Usuario 30 tiene dos interacciones
        # y permanece completamente en Train.
        self.assertEqual(
            train.loc[
                train["user_id"] == 30,
                "movie_id",
            ].tolist(),
            [2, 4],
        )

        # Ningún par usuario-película
        # puede aparecer en dos conjuntos.
        pares = [
            set(zip(df["user_id"], df["movie_id"]))
            for df in (train, validation, test)
        ]

        self.assertTrue(pares[0].isdisjoint(pares[1]))
        self.assertTrue(pares[0].isdisjoint(pares[2]))
        self.assertTrue(pares[1].isdisjoint(pares[2]))

        # Para cada usuario elegible:
        # Train precede a Validation y Test.
        for usuario in (10, 20):

            train_u = train.loc[
                train["user_id"] == usuario
            ]

            val_u = validation.loc[
                validation["user_id"] == usuario
            ]

            test_u = test.loc[
                test["user_id"] == usuario
            ]

            self.assertLessEqual(
                train_u["rated_at"].max(),
                val_u["rated_at"].min(),
            )

            self.assertLessEqual(
                val_u["rated_at"].max(),
                test_u["rated_at"].min(),
            )

        # No modificar las interacciones originales.
        self.assertEqual(len(interactions), 9)

        # Un conjunto sin usuarios elegibles
        # debe rechazarse.
        solo_dos = interactions.loc[
            interactions["user_id"] == 30
        ]

        with self.assertRaisesRegex(
            ValueError,
            "al menos tres interacciones",
        ):
            temporal_train_validation_test_split(
                solo_dos
            )
    
    def test_metricas_ranking_recall_ndcg(self):

        # Películas relevantes reservadas
        # para la evaluación de tres usuarios.
        relevantes = {
            10: {2, 3},
            20: {5},
            30: {8},
        }

        # Ranking ordenado de cada usuario.
        recomendaciones = {
            10: [3, 4, 2],
            20: [7, 5],
            30: [],
        }

        resultado = evaluate_ranking_metrics(
            relevant_by_user=relevantes,
            recommendations_by_user=recomendaciones,
            k=2,
        )

        self.assertEqual(
            resultado["usuarios_relevantes"],
            3,
        )

        self.assertEqual(
            resultado["usuarios_con_recomendaciones"],
            2,
        )

        # Usuario 10: acierta película 3.
        # Usuario 20: acierta película 5.
        # Usuario 30: no recibe recomendaciones.
        self.assertAlmostEqual(
            resultado["hit_rate_at_k"],
            2 / 3,
        )

        # Recall:
        # usuario 10 = 1/2
        # usuario 20 = 1/1
        # usuario 30 = 0/1
        recall_esperado = (
            0.5 + 1.0 + 0.0
        ) / 3

        self.assertAlmostEqual(
            resultado["recall_at_k"],
            recall_esperado,
        )

        # NDCG:
        # usuario 10: acierto en posición 1,
        # con dos películas relevantes.
        ndcg_10 = (
            1.0
            / (1.0 + 1.0 / log2(3))
        )

        # usuario 20: acierto en posición 2.
        ndcg_20 = 1.0 / log2(3)

        ndcg_esperado = (
            ndcg_10 + ndcg_20 + 0.0
        ) / 3

        self.assertAlmostEqual(
            resultado["ndcg_at_k"],
            ndcg_esperado,
        )

        self.assertAlmostEqual(
            resultado["cobertura_recomendaciones"],
            2 / 3,
        )

        # Rechazar películas repetidas en
        # el mismo ranking.
        with self.assertRaisesRegex(
            ValueError,
            "duplicadas",
        ):
            evaluate_ranking_metrics(
                relevant_by_user={10: {2}},
                recommendations_by_user={
                    10: [2, 2]
                },
                k=2,
            )

        # No aceptar K inválido.
        with self.assertRaisesRegex(
            ValueError,
            "k debe ser",
        ):
            evaluate_ranking_metrics(
                relevant_by_user=relevantes,
                recommendations_by_user=recomendaciones,
                k=0,
            )

    def test_hybrid_hit_rate_y_cobertura(self):

        train = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.5, 4.0],
        })

        test = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [4, 5, 6],
            "rating": [5.0, 4.5, 5.0],
        })

        # Recomendaciones controladas:
        # usuario 10: acierto.
        # usuario 20: recomendación incorrecta.
        # usuario 30: ningún candidato.
        resultados = {
            10: pd.DataFrame({
                "movie_id": [4],
                "hybrid_score": [0.02],
            }),
            20: pd.DataFrame({
                "movie_id": [4],
                "hybrid_score": [0.01],
            }),
            30: pd.DataFrame({
                "movie_id": pd.Series(dtype="int64"),
                "hybrid_score": pd.Series(
                    dtype="float64"
                ),
            }),
        }

        def simular_recomendacion(
            user_id,
            **kwargs,
        ):
            # El evaluador debe transmitir
            # únicamente el historial Train.
            pd.testing.assert_frame_equal(
                kwargs["train_interactions"],
                train,
            )

            return resultados[user_id]

        with patch(
            "src.models.hybrid.recommend_hybrid_for_user",
            side_effect=simular_recomendacion,
        ) as mock_recomendador:

            resultado = evaluate_hybrid_hit_rate(
                train=train,
                test=test,
                content_model={},
                collaborative_model={},
                k=1,
            )

        self.assertEqual(
            mock_recomendador.call_count,
            3,
        )

        self.assertEqual(
            resultado["usuarios_relevantes"],
            3,
        )

        self.assertEqual(
            resultado["usuarios_con_recomendaciones"],
            2,
        )

        self.assertEqual(
            resultado["usuarios_sin_recomendaciones"],
            1,
        )

        self.assertEqual(
            resultado["aciertos"],
            1,
        )

        self.assertAlmostEqual(
            resultado["hit_rate_global"],
            1 / 3,
        )

        self.assertAlmostEqual(
            resultado["hit_rate_con_recomendaciones"],
            1 / 2,
        )

        self.assertAlmostEqual(
            resultado["cobertura_recomendaciones"],
            2 / 3,
        )

        # No permitir interacciones repetidas
        # entre Train y Test.
        test_solapado = pd.DataFrame({
            "user_id": [10],
            "movie_id": [1],
            "rating": [5.0],
        })

        with self.assertRaisesRegex(
            ValueError,
            "solapamiento usuario-película",
        ):
            evaluate_hybrid_hit_rate(
                train=train,
                test=test_solapado,
                content_model={},
                collaborative_model={},
            )

    def test_collaborative_hit_rate_y_cobertura(self):

        # Datos de entrenamiento.
        train = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.5, 4.0],
        })

        # Interacciones reservadas para prueba.
        test = pd.DataFrame({
            "user_id": [10, 20, 40],
            "movie_id": [2, 1, 3],
            "rating": [5.0, 5.0, 4.5],
        })

        # Factores conocidos para verificar el
        # ranking sin depender del ajuste de SVD.
        modelo = {
            "user_factors": np.array([
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ]),
            "movie_components": np.array([
                [0.2, 0.9, 0.1],
                [0.8, 0.2, 0.1],
            ]),
            "user_to_idx": {
                10: 0,
                20: 1,
                30: 2,
            },
            "movie_to_idx": {
                1: 0,
                2: 1,
                3: 2,
            },
            "movie_ids": [1, 2, 3],
        }

        resultado = evaluate_collaborative_hit_rate(
            train=train,
            test=test,
            fitted_model=modelo,
            k=1,
        )

        self.assertEqual(
            resultado["metrica"],
            "Hit Rate@1",
        )

        # Tres usuarios tienen ratings relevantes.
        self.assertEqual(
            resultado["usuarios_relevantes"],
            3,
        )

        # El usuario 40 no existe en el modelo.
        self.assertEqual(
            resultado["usuarios_excluidos"],
            1,
        )

        self.assertEqual(
            resultado["usuarios_evaluables"],
            2,
        )

        # Usuario 10 -> película 2.
        # Usuario 20 -> película 1.
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

        # Rechazar interacciones presentes
        # simultáneamente en Train y Test.
        test_solapado = pd.DataFrame({
            "user_id": [10],
            "movie_id": [1],
            "rating": [5.0],
        })

        with self.assertRaisesRegex(
            ValueError,
            "solapamiento usuario-película",
        ):
            evaluate_collaborative_hit_rate(
                train=train,
                test=test_solapado,
                fitted_model=modelo,
            )

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
