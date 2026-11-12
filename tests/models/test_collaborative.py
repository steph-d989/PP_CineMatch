import json
import unittest
import pandas as pd
import numpy as np
import hashlib

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.models.collaborative import *

from src.models.evaluate import (
    temporal_train_test_split,
    evaluate_collaborative_hit_rate,
)

class CollaborativeTests(unittest.TestCase):
    
    
    def test_collaborative_bloquea_carga_modelo_alterado(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        interactions = pd.DataFrame({
            "user_id": [10, 10, 20, 20, 30, 30],
            "movie_id": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 4.5, 3.0, 5.0, 4.0],
        })

        matrix_data = build_user_movie_matrix(
            interactions
        )

        modelo = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "collaborative_model"

            rutas = save_collaborative_model(
                fitted_model=modelo,
                output_dir=destino,
                run_id=run_id,
            )

            # 1. Un modelo íntegro puede cargarse.
            recuperado = load_collaborative_model(
                model_dir=destino,
                expected_run_id=run_id,
            )

            self.assertEqual(
                recuperado["n_factors"],
                2,
            )

            # 2. Modificar un byte sin variar
            # el tamaño del archivo.
            original = rutas["factors"].read_bytes()

            alterado = bytearray(original)
            alterado[0] ^= 1

            rutas["factors"].write_bytes(alterado)

            # 3. Comprobar que se bloquea antes
            # de abrir factors.npz con NumPy.
            with patch(
                "src.models.collaborative.np.load"
            ) as mock_np_load:

                with self.assertRaisesRegex(
                    ValueError,
                    "no coincide con SHA-256",
                ):
                    load_collaborative_model(
                        model_dir=destino,
                        expected_run_id=run_id,
                    )

                mock_np_load.assert_not_called()

            # 4. Restaurar el archivo original.
            rutas["factors"].write_bytes(original)

            restaurado = load_collaborative_model(
                model_dir=destino,
                expected_run_id=run_id,
            )

            self.assertEqual(
                restaurado["user_ids"],
                modelo["user_ids"],
            )

    def test_collaborative_verifica_manifiesto_sha256(self):

        interactions = pd.DataFrame({
            "user_id": [10, 10, 20, 20, 30, 30],
            "movie_id": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 4.5, 3.0, 5.0, 4.0],
        })

        matrix_data = build_user_movie_matrix(
            interactions
        )

        modelo = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        with TemporaryDirectory() as temporal:

            destino = (
                Path(temporal) / "collaborative_model"
            )

            rutas = save_collaborative_model(
                fitted_model=modelo,
                output_dir=destino,
            )

            # 1. Manifiesto original válido.
            resultado = verify_collaborative_manifest(
                destino
            )

            self.assertEqual(
                resultado["estado"],
                "manifest_verified",
            )

            self.assertEqual(
                resultado["total_files"],
                2,
            )

            # 2. Modificar factors.npz manteniendo
            # exactamente el mismo tamaño.
            original = rutas["factors"].read_bytes()

            alterado = bytearray(original)
            alterado[0] ^= 1

            rutas["factors"].write_bytes(
                alterado
            )

            with self.assertRaisesRegex(
                ValueError,
                "no coincide con SHA-256",
            ):
                verify_collaborative_manifest(
                    destino
                )

            # Restaurar los bytes originales.
            rutas["factors"].write_bytes(
                original
            )

            # 3. Comprobar archivo faltante.
            rutas["factors"].unlink()

            with self.assertRaises(
                FileNotFoundError
            ):
                verify_collaborative_manifest(
                    destino
                )

            # Restaurar para comprobar el
            # contenido del manifiesto.
            rutas["factors"].write_bytes(
                original
            )

            # 4. Manipular una ruta declarada.
            manifest = json.loads(
                rutas["manifest"].read_text(
                    encoding="utf-8"
                )
            )

            manifest["files"][0]["path"] = (
                "../archivo_externo.npz"
            )

            rutas["manifest"].write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "componentes inesperados",
            ):
                verify_collaborative_manifest(
                    destino
                )

    def test_collaborative_genera_manifiesto_sha256(self):

        interactions = pd.DataFrame({
            "user_id": [10, 10, 20, 20, 30, 30],
            "movie_id": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 4.5, 3.0, 5.0, 4.0],
        })

        matrix_data = build_user_movie_matrix(
            interactions
        )

        modelo = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        with TemporaryDirectory() as temporal:

            destino = (
                Path(temporal) / "collaborative_model"
            )

            rutas = save_collaborative_model(
                fitted_model=modelo,
                output_dir=destino,
            )

            # El manifiesto debe existir.
            self.assertTrue(
                rutas["manifest"].is_file()
            )

            manifest = json.loads(
                rutas["manifest"].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest["model_type"],
                "collaborative_truncated_svd",
            )

            self.assertEqual(
                manifest["algorithm"],
                "sha256",
            )

            self.assertEqual(
                manifest["total_files"],
                2,
            )

            self.assertEqual(
                {
                    entrada["path"]
                    for entrada in manifest["files"]
                },
                {
                    "factors.npz",
                    "metadata.json",
                },
            )

            # El SHA-256 de los factores debe
            # corresponder a los bytes originales.
            entrada_factores = next(
                entrada
                for entrada in manifest["files"]
                if entrada["component"] == "factors"
            )

            esperado = hashlib.sha256(
                rutas["factors"].read_bytes()
            ).hexdigest()

            self.assertEqual(
                entrada_factores["sha256"],
                esperado,
            )

            # Reconstruir el manifiesto sin
            # alterar los componentes.
            reconstruido = build_collaborative_manifest(
                destino
            )

            self.assertEqual(
                reconstruido,
                manifest,
            )

            # Alterar los factores manteniendo
            # exactamente el mismo tamaño.
            original = rutas["factors"].read_bytes()

            alterado = bytearray(original)
            alterado[0] ^= 1

            rutas["factors"].write_bytes(
                alterado
            )

            modificado = build_collaborative_manifest(
                destino
            )

            self.assertNotEqual(
                modificado,
                manifest,
            )

            self.assertEqual(
                modificado["files"][0]["bytes"],
                manifest["files"][0]["bytes"],
            )

    def test_collaborative_recupera_modelo_sin_reentrenar(self):

        run_id_a = (
            "12345678-1234-4234-8234-123456789abc"
        )

        run_id_b = (
            "87654321-4321-4321-8321-cba987654321"
        )

        train = pd.DataFrame({
            "user_id": [10, 10, 20, 20, 30, 30],
            "movie_id": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 4.5, 3.0, 5.0, 4.0],
        })

        matrix_data = build_user_movie_matrix(
            train
        )

        modelo_original = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        with TemporaryDirectory() as temporal:

            destino = (
                Path(temporal) / "collaborative_model"
            )

            save_collaborative_model(
                fitted_model=modelo_original,
                output_dir=destino,
                run_id=run_id_a,
            )

            # Recuperar los factores sin volver
            # a entrenar el modelo.
            with patch(
                "src.models.collaborative."
                "fit_collaborative_svd",
                side_effect=AssertionError(
                    "No debe volver a entrenarse SVD."
                ),
            ):

                recuperado = load_collaborative_model(
                    model_dir=destino,
                    expected_run_id=run_id_a,
                )

            # Los factores deben ser idénticos.
            np.testing.assert_allclose(
                recuperado["user_factors"],
                modelo_original["user_factors"],
            )

            np.testing.assert_allclose(
                recuperado["movie_components"],
                modelo_original["movie_components"],
            )

            self.assertEqual(
                recuperado["user_ids"],
                modelo_original["user_ids"],
            )

            self.assertEqual(
                recuperado["movie_ids"],
                modelo_original["movie_ids"],
            )

            # Comparar recomendaciones antes
            # y después de recuperar el modelo.
            originales = recommend_collaborative_top_n(
                fitted_model=modelo_original,
                user_id=10,
                train_interactions=train,
                n=2,
            )

            restauradas = recommend_collaborative_top_n(
                fitted_model=recuperado,
                user_id=10,
                train_interactions=train,
                n=2,
            )

            pd.testing.assert_frame_equal(
                originales,
                restauradas,
            )

            # Impedir utilizar otra ejecución.
            with self.assertRaisesRegex(
                ValueError,
                "run_id diferente",
            ):
                load_collaborative_model(
                    model_dir=destino,
                    expected_run_id=run_id_b,
                )

    
    def test_collaborative_guarda_factores_entrenados(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        interactions = pd.DataFrame({
            "user_id": [10, 10, 20, 20, 30, 30],
            "movie_id": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 4.5, 3.0, 5.0, 4.0],
        })

        matrix_data = build_user_movie_matrix(
            interactions
        )

        modelo = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        with TemporaryDirectory() as temporal:

            destino = (
                Path(temporal) / "collaborative_model"
            )

            rutas = save_collaborative_model(
                fitted_model=modelo,
                output_dir=destino,
                run_id=run_id,
            )

            self.assertTrue(
                rutas["factors"].is_file()
            )

            self.assertTrue(
                rutas["metadata"].is_file()
            )

            # Leer NumPy sin deserializar
            # objetos Python.
            with np.load(
                rutas["factors"],
                allow_pickle=False,
            ) as datos:

                np.testing.assert_allclose(
                    datos["user_factors"],
                    modelo["user_factors"],
                )

                np.testing.assert_allclose(
                    datos["movie_components"],
                    modelo["movie_components"],
                )

                self.assertEqual(
                    datos["user_ids"].tolist(),
                    [10, 20, 30],
                )

                self.assertEqual(
                    datos["movie_ids"].tolist(),
                    [1, 2, 3],
                )

            metadata = json.loads(
                rutas["metadata"].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                metadata["model_type"],
                "collaborative_truncated_svd",
            )

            self.assertEqual(
                metadata["run_id"],
                run_id,
            )

            self.assertEqual(
                metadata["n_factors"],
                2,
            )

            # No permitir sobrescribir un modelo.
            with self.assertRaises(FileExistsError):
                save_collaborative_model(
                    fitted_model=modelo,
                    output_dir=destino,
                    run_id=run_id,
                )

    def test_collaborative_ciclo_completo_train_test(self):

        # 1. Historial de cuatro usuarios.
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
                5.0, 4.0, 5.0,
                5.0, 4.0, 5.0,
                5.0, 4.0, 5.0,
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

        # 2. Separación cronológica por usuario.
        train, test = temporal_train_test_split(
            interactions
        )

        self.assertEqual(len(train), 8)
        self.assertEqual(len(test), 4)

        # 3. Preparar exclusivamente Train.
        preparados = prepare_collaborative_ratings(
            train
        )

        # 4. Construir matriz usuario-película.
        matrix_data = build_user_movie_matrix(
            preparados
        )

        self.assertEqual(
            matrix_data["matrix"].shape,
            (4, 4),
        )

        # 5. Entrenar SVD exclusivamente con Train.
        modelo = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=2,
            random_state=42,
        )

        self.assertEqual(
            modelo["n_factors"],
            2,
        )

        # 6. Comprobar recomendaciones.
        recomendaciones = recommend_collaborative_top_n(
            fitted_model=modelo,
            user_id=10,
            train_interactions=train,
            n=2,
        )

        vistas_usuario = set(
            train.loc[
                train["user_id"] == 10,
                "movie_id",
            ]
        )

        self.assertTrue(
            set(recomendaciones["movie_id"]).isdisjoint(
                vistas_usuario
            )
        )

        self.assertTrue(
            np.isfinite(
                recomendaciones["collaborative_score"]
            ).all()
        )

        # 7. Evaluar con las interacciones Test.
        evaluacion = evaluate_collaborative_hit_rate(
            train=train,
            test=test,
            fitted_model=modelo,
            k=1,
        )

        self.assertEqual(
            evaluacion["usuarios_relevantes"],
            4,
        )

        self.assertEqual(
            evaluacion["usuarios_evaluables"],
            4,
        )

        self.assertEqual(
            evaluacion["usuarios_excluidos"],
            0,
        )

        # Verificar coherencia de las métricas.
        self.assertEqual(
            evaluacion["aciertos"],
            round(
                evaluacion["hit_rate"]
                * evaluacion["usuarios_evaluables"]
            ),
        )

        self.assertGreaterEqual(
            evaluacion["hit_rate"],
            0.0,
        )

        self.assertLessEqual(
            evaluacion["hit_rate"],
            1.0,
        )

        self.assertAlmostEqual(
            evaluacion["cobertura_evaluacion"],
            1.0,
        )

        # 8. Ninguna interacción de Test
        # aparece en la matriz de entrenamiento.
        pares_train = set(
            zip(train["user_id"], train["movie_id"])
        )

        pares_test = set(
            zip(test["user_id"], test["movie_id"])
        )

        self.assertTrue(
            pares_train.isdisjoint(pares_test)
        )

    def test_collaborative_prepara_interacciones(self):

        interactions = pd.DataFrame({
            "user_id": [20, 10, 10],
            "movie_id": [3, 2, 1],
            "rating": [3.0, 4.0, 5.0],
            "rated_at": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-02-01",
                    "2024-01-01",
                ],
                utc=True,
            ),
        })

        resultado = prepare_collaborative_ratings(
            interactions
        )

        self.assertEqual(
            len(resultado),
            3,
        )

        self.assertEqual(
            list(resultado.columns),
            ["user_id", "movie_id", "rating"],
        )

        self.assertEqual(
            list(zip(
                resultado["user_id"],
                resultado["movie_id"],
            )),
            [(10, 1), (10, 2), (20, 3)],
        )

        # No modificar las entradas originales.
        self.assertIn(
            "rated_at",
            interactions.columns,
        )

        # Rechazar dos ratings del mismo
        # usuario para la misma película.
        duplicadas = pd.DataFrame({
            "user_id": [10, 10],
            "movie_id": [1, 1],
            "rating": [4.0, 5.0],
        })

        with self.assertRaisesRegex(
            ValueError,
            "duplicadas",
        ):
            prepare_collaborative_ratings(
                duplicadas
            )

        # Rechazar ratings fuera del rango.
        invalidas = pd.DataFrame({
            "user_id": [10],
            "movie_id": [1],
            "rating": [6.0],
        })

        with self.assertRaisesRegex(
            ValueError,
            "ratings inválidos",
        ):
            prepare_collaborative_ratings(
                invalidas
            )
    
    def test_collaborative_construye_matriz_dispersa(self):

        import numpy as np

        from scipy.sparse import csr_matrix

        from src.models.collaborative import (
            build_user_movie_matrix,
        )

        interactions = pd.DataFrame({
            "user_id": [20, 10, 20, 10],
            "movie_id": [3, 2, 1, 1],
            "rating": [3.0, 4.0, 4.5, 5.0],
        })

        resultado = build_user_movie_matrix(
            interactions
        )

        matriz = resultado["matrix"]

        # Debe utilizar almacenamiento CSR.
        self.assertIsInstance(
            matriz,
            csr_matrix,
        )

        # Dos usuarios y tres películas.
        self.assertEqual(
            matriz.shape,
            (2, 3),
        )

        # Deben existir cuatro calificaciones.
        self.assertEqual(
            matriz.nnz,
            4,
        )

        # Identificadores ordenados.
        self.assertEqual(
            resultado["user_ids"],
            [10, 20],
        )

        self.assertEqual(
            resultado["movie_ids"],
            [1, 2, 3],
        )

        # Comprobar las posiciones.
        esperado = np.array([
            [5.0, 4.0, 0.0],
            [4.5, 0.0, 3.0],
        ])

        np.testing.assert_allclose(
            matriz.toarray(),
            esperado,
        )

        # Comprobar el mapeo usuario/película.
        self.assertEqual(
            resultado["user_to_idx"][10],
            0,
        )

        self.assertEqual(
            resultado["movie_to_idx"][3],
            2,
        )

        # La función no altera los datos originales.
        self.assertEqual(
            interactions["user_id"].tolist(),
            [20, 10, 20, 10],
        )

    def test_collaborative_entrena_svd(self):

        interactions = pd.DataFrame({
            "user_id": [
                10, 10, 20, 20, 30, 30
            ],
            "movie_id": [
                1, 2, 1, 3, 2, 3
            ],
            "rating": [
                5.0, 4.0, 4.5, 3.0, 5.0, 4.0
            ],
        })

        matrix_data = build_user_movie_matrix(
            interactions
        )

        # Solicitar 50 factores sobre una matriz
        # 3x3: el algoritmo debe ajustarlos.
        resultado = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=50,
            random_state=42,
        )

        self.assertEqual(
            resultado["n_factors"],
            2,
        )

        self.assertEqual(
            resultado["user_factors"].shape,
            (3, 2),
        )

        self.assertEqual(
            resultado["movie_components"].shape,
            (2, 3),
        )

        self.assertTrue(
            np.isfinite(
                resultado["user_factors"]
            ).all()
        )

        self.assertTrue(
            np.isfinite(
                resultado["movie_components"]
            ).all()
        )

        # El entrenamiento debe ser reproducible
        # con los mismos datos y random_state.
        segundo = fit_collaborative_svd(
            matrix_data=matrix_data,
            n_factors=50,
            random_state=42,
        )

        np.testing.assert_allclose(
            resultado["user_factors"],
            segundo["user_factors"],
        )

        # No se aceptan factores inválidos.
        with self.assertRaisesRegex(
            ValueError,
            "n_factors",
        ):
            fit_collaborative_svd(
                matrix_data=matrix_data,
                n_factors=0,
            )
    
    def test_collaborative_predice_puntuacion_latente(self):

        # Modelo pequeño con factores conocidos.
        # No necesitamos volver a entrenar SVD
        # para comprobar el producto escalar.
        modelo = {
            "user_factors": np.array([
                [0.8, 0.3],
                [0.2, 0.9],
            ]),
            "movie_components": np.array([
                [0.7, 0.1, 0.5],
                [0.4, 0.8, 0.2],
            ]),
            "user_to_idx": {
                10: 0,
                20: 1,
            },
            "movie_to_idx": {
                1: 0,
                2: 1,
                3: 2,
            },
        }

        score = predict_collaborative_score(
            fitted_model=modelo,
            user_id=10,
            movie_id=1,
        )

        # 0.8 * 0.7 + 0.3 * 0.4 = 0.68
        self.assertAlmostEqual(
            score,
            0.68,
        )

        # Comprobar otro usuario y otra película.
        segundo = predict_collaborative_score(
            fitted_model=modelo,
            user_id=20,
            movie_id=2,
        )

        self.assertAlmostEqual(
            segundo,
            0.74,
        )

        # Usuario inexistente.
        with self.assertRaisesRegex(
            ValueError,
            "Usuario desconocido",
        ):
            predict_collaborative_score(
                fitted_model=modelo,
                user_id=99,
                movie_id=1,
            )

        # Película inexistente.
        with self.assertRaisesRegex(
            ValueError,
            "Película desconocida",
        ):
            predict_collaborative_score(
                fitted_model=modelo,
                user_id=10,
                movie_id=99,
            )

    def test_collaborative_recomienda_top_n(self):

        # Modelo con puntuaciones conocidas.
        modelo = {
            "user_factors": np.array([
                [0.8, 0.3],
                [0.2, 0.9],
            ]),
            "movie_components": np.array([
                [0.7, 0.1, 0.5, 0.4],
                [0.4, 0.8, 0.2, 0.6],
            ]),
            "user_to_idx": {
                10: 0,
                20: 1,
            },
            "movie_to_idx": {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
            },
            "movie_ids": [1, 2, 3, 4],
        }

        train = pd.DataFrame({
            "user_id": [10, 10, 20],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 2.0, 4.0],
        })

        resultado = recommend_collaborative_top_n(
            fitted_model=modelo,
            user_id=10,
            train_interactions=train,
            n=2,
        )

        # Usuario 10 ya calificó películas 1 y 2,
        # por lo que solo quedan películas 3 y 4.
        self.assertEqual(
            resultado["movie_id"].tolist(),
            [4, 3],
        )

        # Comprobar las puntuaciones esperadas:
        # película 4 = 0.8*0.4 + 0.3*0.6 = 0.50
        # película 3 = 0.8*0.5 + 0.3*0.2 = 0.46
        self.assertAlmostEqual(
            resultado.iloc[0]["collaborative_score"],
            0.50,
        )

        self.assertAlmostEqual(
            resultado.iloc[1]["collaborative_score"],
            0.46,
        )

        # Las películas vistas no aparecen.
        self.assertTrue(
            set(resultado["movie_id"]).isdisjoint(
                {1, 2}
            )
        )

        # Usuario desconocido.
        with self.assertRaisesRegex(
            ValueError,
            "Usuario desconocido",
        ):
            recommend_collaborative_top_n(
                fitted_model=modelo,
                user_id=99,
                train_interactions=train,
            )

        # Parámetros inválidos.
        with self.assertRaisesRegex(
            ValueError,
            "n debe ser",
        ):
            recommend_collaborative_top_n(
                fitted_model=modelo,
                user_id=10,
                train_interactions=train,
                n=0,
            )


if __name__ == "__main__":
    unittest.main()
