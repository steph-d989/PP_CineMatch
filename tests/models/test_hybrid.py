import unittest

import pandas as pd
import numpy as np

from src.models.hybrid import *

from src.models.content_based import *
from src.models.collaborative import *
from src.models.evaluate import *

class HybridTests(unittest.TestCase):

    
    def test_hybrid_evaluacion_integral_modelos_reales(self):

        # ---------------------------------------
        # 1. Catálogo de películas
        # ---------------------------------------

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
                "Comedia B",
            ],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Mystery"],
                ["Comedy", "Family"],
                ["Comedy", "Family"],
            ],
            "overview": [
                "A detective investigates a mystery",
                "A detective solves a mystery",
                "A funny family adventure",
                "A family comedy adventure",
            ],
        })

        # ---------------------------------------
        # 2. Historial con fechas conocidas
        # ---------------------------------------

        interactions = pd.DataFrame({
            "user_id": [
                10, 10, 10,
                20, 20, 20,
                30, 30, 30,
                40, 40, 40,
            ],
            "movie_id": [
                1, 2, 3,
                2, 3, 4,
                3, 4, 1,
                4, 1, 2,
            ],
            "rating": [
                5.0, 4.5, 5.0,
                5.0, 4.0, 5.0,
                5.0, 4.5, 5.0,
                5.0, 4.0, 5.0,
            ],
            "rated_at": pd.to_datetime([
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ], utc=True),
        })

        # ---------------------------------------
        # 3. Separar Train y Test
        # ---------------------------------------

        train, test = temporal_train_test_split(
            interactions
        )

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 4)

        # Confirmar que las interacciones
        # reservadas no aparecen en Train.
        pares_train = set(
            zip(train["user_id"], train["movie_id"])
        )

        pares_test = set(
            zip(test["user_id"], test["movie_id"])
        )

        self.assertTrue(
            pares_train.isdisjoint(pares_test)
        )

        # ---------------------------------------
        # 4. Entrenar Content-Based
        # ---------------------------------------

        # TF-IDF se ajusta sobre los atributos
        # del catálogo, no sobre las calificaciones
        # reservadas en Test.
        content_model = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        # ---------------------------------------
        # 5. Entrenar Collaborative
        # ---------------------------------------

        # Únicamente Train alimenta la matriz.
        matrix_data = build_user_movie_matrix(
            train
        )

        self.assertEqual(
            matrix_data["total_interactions"],
            8,
        )

        collaborative_model = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        # ---------------------------------------
        # 6. Evaluar Hybrid con Test
        # ---------------------------------------

        evaluacion = evaluate_hybrid_hit_rate(
            train=train,
            test=test,
            content_model=content_model,
            collaborative_model=collaborative_model,
            k=2,
            content_weight=0.5,
            collaborative_weight=0.5,
        )

        self.assertEqual(
            evaluacion["metrica"],
            "Hit Rate@2",
        )

        self.assertEqual(
            evaluacion["usuarios_relevantes"],
            4,
        )

        # En esta muestra todos los usuarios
        # tienen historial y existen candidatos.
        self.assertEqual(
            evaluacion["usuarios_con_recomendaciones"],
            4,
        )

        self.assertEqual(
            evaluacion["usuarios_sin_recomendaciones"],
            0,
        )

        self.assertAlmostEqual(
            evaluacion["cobertura_recomendaciones"],
            1.0,
        )

        # No imponemos un porcentaje de aciertos:
        # queremos observar el modelo real.
        self.assertGreaterEqual(
            evaluacion["hit_rate_global"],
            0.0,
        )

        self.assertLessEqual(
            evaluacion["hit_rate_global"],
            1.0,
        )

        self.assertEqual(
            evaluacion["aciertos"],
            round(
                evaluacion["hit_rate_global"]
                * evaluacion["usuarios_relevantes"]
            ),
        )

        # ---------------------------------------
        # 7. Comprobar reproducibilidad
        # ---------------------------------------

        repeticion = evaluate_hybrid_hit_rate(
            train=train,
            test=test,
            content_model=content_model,
            collaborative_model=collaborative_model,
            k=2,
            content_weight=0.5,
            collaborative_weight=0.5,
        )

        self.assertEqual(
            evaluacion,
            repeticion,
        )
        
        
        # ---------------------------------------
        # 8. COMPARACIÓN DE LOS CUATRO MODELOS
        # ---------------------------------------

        # Todos los evaluadores reciben exactamente
        # el mismo conjunto Train y Test.

        resultado_baseline = evaluate_baseline_hit_rate(
            train=train,
            test=test,
            dim_movie=peliculas,
            k=2,
        )

        resultado_content = evaluate_content_hit_rate(
            train=train,
            test=test,
            content_df=content_model["content_df"],
            tfidf_matrix=content_model["tfidf_matrix"],
            k=2,
        )

        resultado_collaborative = (
            evaluate_collaborative_hit_rate(
                train=train,
                test=test,
                fitted_model=collaborative_model,
                k=2,
            )
        )

        resultado_hybrid = evaluacion

        resultados = {
            "Baseline": resultado_baseline,
            "Content-Based": resultado_content,
            "Collaborative": resultado_collaborative,
            "Hybrid": resultado_hybrid,
        }

        # 9. Todos deben evaluar el mismo universo
        # de usuarios con interacciones relevantes.
        usuarios_esperados = test.loc[
            test["rating"] >= 4.0,
            "user_id",
        ].nunique()

        self.assertEqual(
            usuarios_esperados,
            4,
        )

        for nombre, resultado in resultados.items():

            self.assertEqual(
                resultado["usuarios_relevantes"],
                usuarios_esperados,
                msg=f"Denominador diferente: {nombre}",
            )

            self.assertEqual(
                resultado["metrica"],
                "Hit Rate@2",
            )

            self.assertAlmostEqual(
                resultado["hit_rate_global"],
                resultado["aciertos"] / usuarios_esperados,
            )

            self.assertGreaterEqual(
                resultado["hit_rate_global"],
                0.0,
            )

            self.assertLessEqual(
                resultado["hit_rate_global"],
                1.0,
            )

        # 10. Crear un informe comparable.
        comparacion = pd.DataFrame([
            {
                "modelo": nombre,
                "usuarios_relevantes": resultado[
                    "usuarios_relevantes"
                ],
                "aciertos": resultado["aciertos"],
                "hit_rate_global": resultado[
                    "hit_rate_global"
                ],
            }
            for nombre, resultado in resultados.items()
        ])

        self.assertEqual(
            len(comparacion),
            4,
        )

        self.assertEqual(
            set(comparacion["modelo"]),
            {
                "Baseline",
                "Content-Based",
                "Collaborative",
                "Hybrid",
            },
        )

        print("\nCOMPARACIÓN DE MODELOS CINEMATCH")
        print(
            comparacion.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    def test_hybrid_cold_start_utiliza_baseline(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
            ],
            "genres": [
                ["Drama"],
                ["Drama"],
                ["Comedy"],
            ],
        })

        content_model = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        # Modelo colaborativo válido, pero
        # el usuario 99 no está representado.
        collaborative_model = {
            "user_factors": __import__("numpy").array([
                [1.0, 0.0],
                [0.0, 1.0],
            ]),
            "movie_components": __import__("numpy").array([
                [0.8, 0.5, 0.1],
                [0.2, 0.6, 0.9],
            ]),
            "user_to_idx": {10: 0, 20: 1},
            "movie_to_idx": {1: 0, 2: 1, 3: 2},
            "movie_ids": [1, 2, 3],
        }

        train = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [1, 2],
            "rating": [5.0, 4.5],
        })

        # Ranking ya calculado por nuestro Baseline.
        baseline = pd.DataFrame({
            "movie_id": [2, 1, 3],
            "weighted_score": [4.8, 4.5, 3.9],
        })

        resultado = recommend_hybrid_for_user(
            user_id=99,
            train_interactions=train,
            content_model=content_model,
            collaborative_model=collaborative_model,
            baseline_ranking=baseline,
            n=2,
        )

        self.assertEqual(
            resultado["movie_id"].tolist(),
            [2, 1],
        )

        self.assertTrue(
            (
                resultado["recommendation_source"]
                == "baseline"
            ).all()
        )

        self.assertTrue(
            resultado["hybrid_score"].isna().all()
        )

        self.assertEqual(
            resultado["baseline_score"].tolist(),
            [4.8, 4.5],
        )

        # El respaldo no inventa preferencias.
        self.assertTrue(
            resultado["content_rank"].isna().all()
        )

        self.assertTrue(
            resultado["collaborative_rank"].isna().all()
        )

        # Comprobar que no se modifica el ranking.
        self.assertEqual(
            baseline["movie_id"].tolist(),
            [2, 1, 3],
        )

    def test_hybrid_recomienda_directamente_por_usuario(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
                "Comedia B",
            ],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Mystery"],
                ["Comedy", "Family"],
                ["Comedy", "Family"],
            ],
            "overview": [
                "A detective mystery",
                "Another detective mystery",
                "A funny family story",
                "A family comedy",
            ],
        })

        content_model = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        train = pd.DataFrame({
            "user_id": [10, 10, 20, 20],
            "movie_id": [1, 3, 2, 4],
            "rating": [5.0, 2.0, 4.5, 4.0],
        })

        # Factores controlados para comprobar
        # la integración sin depender de SVD.
        collaborative_model = {
            "user_factors": np.array([
                [0.8, 0.3],
                [0.2, 0.9],
            ]),
            "movie_components": np.array([
                [0.7, 0.9, 0.1, 0.4],
                [0.4, 0.2, 0.8, 0.6],
            ]),
            "user_to_idx": {10: 0, 20: 1},
            "movie_to_idx": {
                1: 0, 2: 1, 3: 2, 4: 3
            },
            "movie_ids": [1, 2, 3, 4],
        }

        # ---------------------------------------
        # ESCENARIO 1: AMBOS MODELOS DISPONIBLES
        # ---------------------------------------

        resultado = recommend_hybrid_for_user(
            user_id=10,
            train_interactions=train,
            content_model=content_model,
            collaborative_model=collaborative_model,
            n=3,
        )

        self.assertGreater(
            len(resultado),
            0,
        )

        self.assertTrue(
            set(resultado["movie_id"]).isdisjoint(
                {1, 3}
            )
        )

        self.assertFalse(
            resultado["movie_id"].duplicated().any()
        )

        self.assertTrue(
            (resultado["hybrid_score"] > 0).all()
        )

        # ---------------------------------------
        # ESCENARIO 2: SIN PERFIL CONTENT-BASED
        # ---------------------------------------

        # El usuario 20 solo tiene valoraciones
        # inferiores al umbral para construir perfil.
        train_sin_positivas = train.copy()

        train_sin_positivas.loc[
            train_sin_positivas["user_id"] == 20,
            "rating",
        ] = 2.0

        respaldo = recommend_hybrid_for_user(
            user_id=20,
            train_interactions=train_sin_positivas,
            content_model=content_model,
            collaborative_model=collaborative_model,
            n=2,
        )

        self.assertGreater(
            len(respaldo),
            0,
        )

        self.assertTrue(
            respaldo["content_rank"].isna().all()
        )

        self.assertTrue(
            respaldo["collaborative_rank"].notna().all()
        )

        # ---------------------------------------
        # ESCENARIO 3: USUARIO SIN HISTORIAL
        # ---------------------------------------

        nuevo = recommend_hybrid_for_user(
            user_id=99,
            train_interactions=train,
            content_model=content_model,
            collaborative_model=collaborative_model,
            n=3,
        )

        self.assertTrue(
            nuevo.empty
        )

    def test_hybrid_integra_modelos_reales(self):

        # ---------------------------------------
        # 1. Catálogo de películas de prueba
        # ---------------------------------------

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
                "Comedia B",
            ],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Mystery"],
                ["Comedy", "Family"],
                ["Comedy", "Family"],
            ],
            "overview": [
                "A detective investigates a mystery",
                "A detective solves a mystery",
                "A funny family adventure",
                "A family comedy adventure",
            ],
        })

        # ---------------------------------------
        # 2. Interacciones de entrenamiento
        # ---------------------------------------

        train = pd.DataFrame({
            "user_id": [
                10, 10,
                20, 20,
                30, 30,
                40, 40,
            ],
            "movie_id": [
                1, 3,
                1, 2,
                2, 3,
                3, 4,
            ],
            "rating": [
                5.0, 2.0,
                4.0, 5.0,
                5.0, 4.0,
                3.0, 5.0,
            ],
        })

        # ---------------------------------------
        # 3. Entrenar Content-Based real
        # ---------------------------------------

        content_model = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        content_ranking = recommend_content_for_user(
            user_id=10,
            interactions=train,
            content_df=content_model["content_df"],
            tfidf_matrix=content_model["tfidf_matrix"],
            n=4,
        )

        # ---------------------------------------
        # 4. Entrenar Collaborative real
        # ---------------------------------------

        matrix_data = build_user_movie_matrix(
            train
        )

        collaborative_model = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        collaborative_ranking = (
            recommend_collaborative_top_n(
                fitted_model=collaborative_model,
                user_id=10,
                train_interactions=train,
                n=4,
            )
        )

        # ---------------------------------------
        # 5. Generar ranking híbrido
        # ---------------------------------------

        vistas = set(
            train.loc[
                train["user_id"] == 10,
                "movie_id",
            ]
        )

        resultado = fuse_hybrid_rankings(
            content_ranking=content_ranking,
            collaborative_ranking=collaborative_ranking,
            n=3,
            content_weight=0.5,
            collaborative_weight=0.5,
            seen_movie_ids=vistas,
        )

        # El resultado debe contener candidatos.
        self.assertGreater(
            len(resultado),
            0,
        )

        # El usuario 10 ya calificó películas 1 y 3.
        self.assertTrue(
            set(resultado["movie_id"]).isdisjoint(
                {1, 3}
            )
        )

        # No deben duplicarse películas.
        self.assertFalse(
            resultado["movie_id"].duplicated().any()
        )

        # Los candidatos deben proceder de al menos
        # uno de los dos modelos.
        candidatos = (
            set(content_ranking["movie_id"])
            | set(collaborative_ranking["movie_id"])
        )

        self.assertTrue(
            set(resultado["movie_id"]).issubset(
                candidatos
            )
        )

        # Las puntuaciones híbridas deben ser positivas.
        self.assertTrue(
            (resultado["hybrid_score"] > 0).all()
        )

        # ---------------------------------------
        # 6. Comprobar resultados reproducibles
        # ---------------------------------------

        repetido = fuse_hybrid_rankings(
            content_ranking=content_ranking,
            collaborative_ranking=collaborative_ranking,
            n=3,
            seen_movie_ids=vistas,
        )

        pd.testing.assert_frame_equal(
            resultado,
            repetido,
        )

        # ---------------------------------------
        # 7. Comprobar respaldo con un solo modelo
        # ---------------------------------------

        collaborative_vacio = pd.DataFrame({
            "movie_id": pd.Series(dtype="int64"),
            "collaborative_score": pd.Series(
                dtype="float64"
            ),
        })

        respaldo = fuse_hybrid_rankings(
            content_ranking=content_ranking,
            collaborative_ranking=collaborative_vacio,
            n=3,
            seen_movie_ids=vistas,
        )

        self.assertEqual(
            respaldo["movie_id"].tolist(),
            content_ranking["movie_id"].head(3).tolist(),
        )

    def test_hybrid_combina_rankings_rrf(self):

        content = pd.DataFrame({
            "movie_id": [1, 2],
            "content_score": [0.95, 0.70],
        })

        collaborative = pd.DataFrame({
            "movie_id": [2, 3],
            "collaborative_score": [4.8, -0.5],
        })

        resultado = fuse_hybrid_rankings(
            content_ranking=content,
            collaborative_ranking=collaborative,
            n=3,
        )

        # La película 2 aparece en ambos rankings.
        self.assertEqual(
            resultado["movie_id"].tolist(),
            [2, 1, 3],
        )

        # RRF no suma 0.70 + 4.8:
        # utiliza las posiciones 2 y 1.
        esperado = (
            0.5 / (60 + 2)
            + 0.5 / (60 + 1)
        )

        self.assertAlmostEqual(
            resultado.iloc[0]["hybrid_score"],
            esperado,
        )

        # Excluir una película ya calificada.
        sin_vistas = fuse_hybrid_rankings(
            content_ranking=content,
            collaborative_ranking=collaborative,
            n=3,
            seen_movie_ids={1},
        )

        self.assertEqual(
            sin_vistas["movie_id"].tolist(),
            [2, 3],
        )

        # Cambiar la escala de las puntuaciones
        # sin cambiar el orden de cada modelo
        # no debe alterar el ranking híbrido.
        collaborative_alterado = collaborative.copy()

        collaborative_alterado[
            "collaborative_score"
        ] = [1000.0, -1000.0]

        segundo = fuse_hybrid_rankings(
            content_ranking=content,
            collaborative_ranking=collaborative_alterado,
            n=3,
        )

        self.assertEqual(
            segundo["movie_id"].tolist(),
            resultado["movie_id"].tolist(),
        )

        # Rechazar pesos inconsistentes.
        with self.assertRaisesRegex(
            ValueError,
            "sumar 1",
        ):
            fuse_hybrid_rankings(
                content,
                collaborative,
                content_weight=0.8,
                collaborative_weight=0.8,
            )


if __name__ == "__main__":
    unittest.main()
