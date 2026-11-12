import pandas as pd
import unittest
import numpy as np
import json
import hashlib

from scipy.sparse import load_npz
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory

from src.models.content_based import *

class ContentBasedTests(unittest.TestCase):


    def test_content_based_modelo_vinculado_run_id(self):

        run_id_a = (
            "12345678-1234-4234-8234-123456789abc"
        )

        run_id_b = (
            "87654321-4321-4321-8321-cba987654321"
        )

        peliculas = pd.DataFrame({
            "movie_id": [1, 2],
            "title": ["Drama A", "Drama B"],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Thriller"],
            ],
            "overview": [
                "A detective mystery",
                "A psychological drama",
            ],
        })

        modelo = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "content_model"

            # Guardar un modelo asociado a la
            # ejecución A.
            rutas = save_content_based_model(
                model=modelo,
                output_dir=destino,
                run_id=run_id_a,
            )

            metadata = json.loads(
                rutas["metadata"].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                metadata["run_id"],
                run_id_a,
            )

            # Se permite recuperar el modelo
            # para la ejecución correcta.
            recuperado = load_content_based_model(
                model_dir=destino,
                trusted_source=True,
                expected_run_id=run_id_a,
            )

            self.assertEqual(
                recuperado["total_movies"],
                2,
            )

            # Se rechaza el modelo cuando se
            # solicita una ejecución diferente.
            with self.assertRaisesRegex(
                ValueError,
                "run_id diferente",
            ):
                load_content_based_model(
                    model_dir=destino,
                    trusted_source=True,
                    expected_run_id=run_id_b,
                )

            # Los archivos originales permanecen
            # intactos después del rechazo.
            self.assertTrue(
                rutas["manifest"].is_file()
            )

            self.assertTrue(
                rutas["vectorizer"].is_file()
            )

    def test_content_based_recupera_modelo_sin_reentrenar(self):

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
            "overview": [
                "A detective mystery",
                "A mystery investigation",
                "A family comedy",
            ],
        })

        # Entrenar una sola vez.
        modelo_original = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "content_model"

            save_content_based_model(
                model=modelo_original,
                output_dir=destino,
            )

            # No permitir carga sin declarar
            # que el origen es confiable.
            with self.assertRaisesRegex(
                ValueError,
                "fuente confiable",
            ):
                load_content_based_model(
                    model_dir=destino,
                )

            # Verificar que la recuperación no
            # ejecuta nuevamente el entrenamiento.
            with patch(
                "src.models.content_based."
                "fit_content_based_model",
                side_effect=AssertionError(
                    "No debe reentrenarse TF-IDF."
                ),
            ):

                modelo_recuperado = (
                    load_content_based_model(
                        model_dir=destino,
                        trusted_source=True,
                    )
                )

            # Comprobar dimensiones.
            self.assertEqual(
                modelo_recuperado["total_movies"],
                modelo_original["total_movies"],
            )

            self.assertEqual(
                modelo_recuperado["total_features"],
                modelo_original["total_features"],
            )

            # Verificar el orden del catálogo.
            self.assertEqual(
                modelo_recuperado["content_df"][
                    "movie_id"
                ].tolist(),
                [1, 2, 3],
            )

            # Verificar igualdad numérica
            # entre matrices original y recuperada.
            np.testing.assert_allclose(
                modelo_original[
                    "tfidf_matrix"
                ].toarray(),
                modelo_recuperado[
                    "tfidf_matrix"
                ].toarray(),
            )

            # Obtener recomendaciones con
            # el modelo original.
            recomendaciones_originales = (
                recommend_similar_movies(
                    movie_id=1,
                    content_df=modelo_original[
                        "content_df"
                    ],
                    tfidf_matrix=modelo_original[
                        "tfidf_matrix"
                    ],
                    n=2,
                )
            )

            # Obtener recomendaciones con
            # el modelo recuperado.
            recomendaciones_recuperadas = (
                recommend_similar_movies(
                    movie_id=1,
                    content_df=modelo_recuperado[
                        "content_df"
                    ],
                    tfidf_matrix=modelo_recuperado[
                        "tfidf_matrix"
                    ],
                    n=2,
                )
            )

            pd.testing.assert_frame_equal(
                recomendaciones_originales,
                recomendaciones_recuperadas,
            )

    def test_content_based_verifica_manifiesto_guardado(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2],
            "title": ["Drama A", "Drama B"],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Thriller"],
            ],
            "overview": [
                "A detective mystery",
                "A psychological drama",
            ],
        })

        modelo = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "content_model"

            rutas = save_content_based_model(
                model=modelo,
                output_dir=destino,
            )

            # Los cuatro archivos coinciden.
            resultado = verify_content_model_manifest(
                destino
            )

            self.assertEqual(
                resultado["estado"],
                "manifest_verified",
            )

            self.assertEqual(
                resultado["total_files"],
                4,
            )

            # Modificar joblib conservando el tamaño.
            contenido_original = rutas[
                "vectorizer"
            ].read_bytes()

            contenido_alterado = bytearray(
                contenido_original
            )

            contenido_alterado[0] ^= 1

            rutas["vectorizer"].write_bytes(
                contenido_alterado
            )

            with self.assertRaisesRegex(
                ValueError,
                "no coincide con SHA-256",
            ):
                verify_content_model_manifest(
                    destino
                )

            # Restaurar el archivo original.
            rutas["vectorizer"].write_bytes(
                contenido_original
            )

            # Comprobar un componente inexistente.
            rutas["matrix"].unlink()

            with self.assertRaises(
                FileNotFoundError
            ):
                verify_content_model_manifest(
                    destino
                )

            # Restaurar la matriz para comprobar
            # una manipulación del manifiesto.
            from scipy.sparse import save_npz

            save_npz(
                rutas["matrix"],
                modelo["tfidf_matrix"],
            )

            manifest = json.loads(
                rutas["manifest"].read_text(
                    encoding="utf-8"
                )
            )

            manifest["files"][0]["path"] = (
                "../archivo_externo.parquet"
            )

            rutas["manifest"].write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "componentes inesperados",
            ):
                verify_content_model_manifest(
                    destino
                )

    def test_content_based_manifiesto_sha256(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2],
            "title": ["Drama A", "Drama B"],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Thriller"],
            ],
            "overview": [
                "A detective mystery",
                "A psychological drama",
            ],
        })

        modelo = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "content_model"

            rutas = save_content_based_model(
                model=modelo,
                output_dir=destino,
            )

            manifest_path = rutas["manifest"]

            self.assertTrue(
                manifest_path.is_file()
            )

            manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                manifest["model_type"],
                "content_based_tfidf",
            )

            self.assertEqual(
                manifest["total_files"],
                4,
            )

            self.assertEqual(
                {
                    item["path"]
                    for item in manifest["files"]
                },
                {
                    "catalog.parquet",
                    "vectorizer.joblib",
                    "tfidf_matrix.npz",
                    "metadata.json",
                },
            )

            # Verificar la huella real del vectorizador.
            entrada = next(
                item
                for item in manifest["files"]
                if item["component"] == "vectorizer"
            )

            hash_esperado = hashlib.sha256(
                rutas["vectorizer"].read_bytes()
            ).hexdigest()

            self.assertEqual(
                entrada["sha256"],
                hash_esperado,
            )

            # El manifiesto puede reconstruirse
            # sin modificar los archivos.
            reconstruido = build_content_model_manifest(
                destino
            )

            self.assertEqual(
                reconstruido,
                manifest,
            )

            # Una alteración debe generar un hash diferente.
            rutas["vectorizer"].write_bytes(
                b"archivo alterado"
            )

            modificado = build_content_model_manifest(
                destino
            )

            self.assertNotEqual(
                modificado,
                manifest,
            )

    def test_content_based_guarda_modelo_entrenado(self):

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
            "overview": [
                "A detective mystery",
                "A mystery investigation",
                "A family comedy",
            ],
        })

        modelo = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        with TemporaryDirectory() as temporal:

            destino = (
                Path(temporal) / "content_model"
            )

            rutas = save_content_based_model(
                model=modelo,
                output_dir=destino,
            )

            for ruta in rutas.values():
                self.assertTrue(
                    ruta.is_file()
                )

            matriz_guardada = load_npz(
                rutas["matrix"]
            )

            self.assertEqual(
                matriz_guardada.shape,
                modelo["tfidf_matrix"].shape,
            )

            catalogo_guardado = pd.read_parquet(
                rutas["catalog"]
            )

            self.assertEqual(
                catalogo_guardado["movie_id"].tolist(),
                [1, 2, 3],
            )

            metadata = json.loads(
                rutas["metadata"].read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                metadata["total_movies"],
                3,
            )

            self.assertEqual(
                metadata["total_features"],
                modelo["total_features"],
            )

            # No se permite sobrescribir
            # el modelo ya guardado.
            with self.assertRaises(FileExistsError):
                save_content_based_model(
                    model=modelo,
                    output_dir=destino,
                )

    def test_normalizacion_caracteristicas_semanticas(self):

        # Listas de géneros.
        self.assertEqual(
            normalize_list_field(
                ["Science Fiction", "Drama"]
            ),
            "science_fiction drama",
        )

        # Arreglos de NumPy recuperados
        # desde archivos Parquet.
        self.assertEqual(
            normalize_list_field(
                np.array(["Thriller", "Mystery"])
            ),
            "thriller mystery",
        )

        # Valores faltantes.
        self.assertEqual(
            normalize_list_field(None),
            "",
        )

        self.assertEqual(
            normalize_list_field([]),
            "",
        )

        # Texto simple.
        self.assertEqual(
            normalize_text(
                "  A Psychological   Drama  "
            ),
            "a psychological drama",
        )

        # Director.
        self.assertEqual(
            normalize_text("Director A").replace(
                " ", "_"
            ),
            "director_a",
        )

        # Valores nulos.
        self.assertEqual(
            normalize_text(None),
            "",
        )

    def test_content_soup_combina_atributos(self):


        peliculas = pd.DataFrame({
            "movie_id": [1, 2],
            "title": ["Pelicula A", "Pelicula B"],
            "genres": [
                ["Drama", "Thriller"],
                [],
            ],
            "tmdb_genres": [
                ["Drama"],
                [],
            ],
            "keywords": [
                ["Secret Identity"],
                [],
            ],
            "actors": [
                ["Actor A", "Actor B"],
                [],
            ],
            "director": [
                "Director A",
                None,
            ],
            "overview": [
                "A psychological drama",
                None,
            ],
            "original_language": [
                "en",
                None,
            ],
        })

        resultado = build_content_soup(
            peliculas
        )

        self.assertEqual(
            len(resultado),
            2,
        )

        sopa_a = resultado.loc[
            resultado["movie_id"] == 1,
            "content_soup",
        ].iloc[0]

        self.assertIn(
            "drama thriller",
            sopa_a,
        )

        self.assertIn(
            "secret_identity",
            sopa_a,
        )

        self.assertIn(
            "actor_a actor_b",
            sopa_a,
        )

        self.assertIn(
            "director_a",
            sopa_a,
        )

        self.assertIn(
            "a psychological drama",
            sopa_a,
        )

        # Una película sin atributos se conserva.
        sopa_b = resultado.loc[
            resultado["movie_id"] == 2,
            "content_soup",
        ].iloc[0]

        self.assertEqual(
            sopa_b,
            "",
        )

        # No alterar el DataFrame original.
        self.assertNotIn(
            "content_soup",
            peliculas.columns,
        )
    
    def test_tfidf_vectoriza_peliculas(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "content_soup": [
                "drama mystery detective",
                "drama mystery investigation",
                "comedy family adventure",
            ],
        })

        vectorizador, matriz = build_tfidf_features(
            peliculas,
            min_df=1,
        )

        # Tres películas, una fila por película.
        self.assertEqual(
            matriz.shape[0],
            3,
        )

        # Debe existir al menos una característica.
        self.assertGreater(
            matriz.shape[1],
            0,
        )

        # El número de columnas coincide
        # con el vocabulario aprendido.
        self.assertEqual(
            matriz.shape[1],
            len(vectorizador.vocabulary_),
        )

        # Los términos compartidos deben
        # estar presentes en el vocabulario.
        self.assertIn(
            "drama",
            vectorizador.vocabulary_,
        )

        self.assertIn(
            "mystery",
            vectorizador.vocabulary_,
        )

        # Comprobar que se utiliza una matriz
        # dispersa para ahorrar memoria.
        self.assertEqual(
            matriz.getformat(),
            "csr",
        )

        # No se modifica la entrada original.
        self.assertEqual(
            peliculas.iloc[0]["content_soup"],
            "drama mystery detective",
        )

        # Un catálogo completamente vacío
        # debe rechazarse.
        vacio = pd.DataFrame({
            "content_soup": ["", ""],
        })

        with self.assertRaisesRegex(
            ValueError,
            "No existen textos aprovechables",
        ):
            build_tfidf_features(vacio)

    def test_content_based_recomienda_peliculas_similares(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
            "content_soup": [
                "drama mystery detective",
                "drama mystery investigation",
                "comedy family adventure",
            ],
        })

        _, matriz = build_tfidf_features(
            peliculas,
            min_df=1,
        )

        resultado = recommend_similar_movies(
            movie_id=1,
            content_df=peliculas,
            tfidf_matrix=matriz,
            n=2,
        )

        # La película consultada no se recomienda.
        self.assertNotIn(
            1,
            resultado["movie_id"].tolist(),
        )

        # La película B comparte características
        # con la película A.
        self.assertEqual(
            resultado.iloc[0]["movie_id"],
            2,
        )

        self.assertGreater(
            resultado.iloc[0]["content_score"],
            0,
        )

        # La película C no comparte términos
        # y no debe aparecer como similar.
        self.assertNotIn(
            3,
            resultado["movie_id"].tolist(),
        )

        # Una película desconocida se rechaza.
        with self.assertRaisesRegex(
            ValueError,
            "no existe",
        ):
            recommend_similar_movies(
                movie_id=99,
                content_df=peliculas,
                tfidf_matrix=matriz,
            )

    def test_content_based_construye_perfil_usuario(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "content_soup": [
                "drama mystery detective",
                "drama mystery investigation",
                "comedy family adventure",
            ],
        })

        _, matriz = build_tfidf_features(
            peliculas,
            min_df=1,
        )

        interactions = pd.DataFrame({
            "user_id": [10, 10, 10],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.5, 2.0],
        })

        perfil = build_user_content_profile(
            user_id=10,
            interactions=interactions,
            content_df=peliculas,
            tfidf_matrix=matriz,
        )

        # El perfil tiene la misma dimensión
        # que los vectores TF-IDF.
        self.assertEqual(
            perfil.shape,
            (matriz.shape[1],),
        )

        # Debe ser un vector numérico.
        self.assertTrue(
            np.isfinite(perfil).all()
        )

        # El perfil corresponde al promedio
        # de las dos películas positivas.
        esperado = np.asarray(
            matriz[[0, 1]].mean(axis=0)
        ).ravel()

        np.testing.assert_allclose(
            perfil,
            esperado,
        )

        # La película con rating 2.0 no
        # participa en el perfil.
        self.assertFalse(
            np.allclose(
                perfil,
                np.asarray(
                    matriz[[0, 1, 2]].mean(axis=0)
                ).ravel(),
            )
        )

        # Un usuario desconocido no tiene
        # perfil construido desde su historial.
        with self.assertRaisesRegex(
            ValueError,
            "no tiene calificaciones positivas",
        ):
            build_user_content_profile(
                user_id=99,
                interactions=interactions,
                content_df=peliculas,
                tfidf_matrix=matriz,
            )

    def test_content_based_recomienda_por_usuario(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3, 4, 5],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
                "Pelicula D",
                "Pelicula E",
            ],
            "content_soup": [
                "drama mystery detective",
                "drama mystery investigation",
                "comedy family adventure",
                "drama mystery investigation",
                "",
            ],
        })

        _, matriz = build_tfidf_features(
            peliculas,
            min_df=1,
        )

        interactions = pd.DataFrame({
            "user_id": [10, 10, 10],
            "movie_id": [1, 2, 3],
            "rating": [5.0, 4.5, 2.0],
        })

        recomendaciones = recommend_content_for_user(
            user_id=10,
            interactions=interactions,
            content_df=peliculas,
            tfidf_matrix=matriz,
            n=3,
        )

        # La película D comparte características
        # con las favoritas del usuario.
        self.assertEqual(
            recomendaciones["movie_id"].tolist(),
            [4],
        )

        # Las películas ya calificadas no
        # deben aparecer, incluso con rating bajo.
        self.assertTrue(
            set(recomendaciones["movie_id"]).isdisjoint(
                {1, 2, 3}
            )
        )

        # La película E no tiene contenido semántico.
        self.assertNotIn(
            5,
            recomendaciones["movie_id"].tolist(),
        )

        # Las recomendaciones deben tener
        # puntuaciones de similitud positivas.
        self.assertTrue(
            (
                recomendaciones["content_score"] > 0
            ).all()
        )

        # Un usuario sin historial positivo
        # no puede construir este tipo de perfil.
        with self.assertRaisesRegex(
            ValueError,
            "no tiene calificaciones positivas",
        ):
            recommend_content_for_user(
                user_id=99,
                interactions=interactions,
                content_df=peliculas,
                tfidf_matrix=matriz,
            )

    def test_content_based_entrenamiento_integrado(self):

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3, 4],
            "title": [
                "Drama A",
                "Drama B",
                "Comedia A",
                "Drama C",
            ],
            "genres": [
                ["Drama", "Mystery"],
                ["Drama", "Mystery"],
                ["Comedy", "Family"],
                ["Drama", "Mystery"],
            ],
            "keywords": [
                ["Detective"],
                ["Investigation"],
                ["Humor"],
                ["Detective"],
            ],
            "overview": [
                "A detective investigates a mystery",
                "A mystery investigation",
                "A family comedy",
                "A detective investigates a mystery",
            ],
        })

        # Entrenar mediante una sola operación.
        modelo = fit_content_based_model(
            content_features=peliculas,
            min_df=1,
        )

        self.assertEqual(
            modelo["total_movies"],
            4,
        )

        self.assertGreater(
            modelo["total_features"],
            0,
        )

        self.assertEqual(
            modelo["tfidf_matrix"].shape[0],
            4,
        )

        self.assertIn(
            "content_soup",
            modelo["content_df"].columns,
        )

        # Comprobar recomendaciones similares.
        similares = recommend_similar_movies(
            movie_id=1,
            content_df=modelo["content_df"],
            tfidf_matrix=modelo["tfidf_matrix"],
            n=2,
        )

        self.assertEqual(
            int(similares.iloc[0]["movie_id"]),
            4,
        )

        # Comprobar recomendaciones por usuario.
        train = pd.DataFrame({
            "user_id": [10, 10],
            "movie_id": [1, 3],
            "rating": [5.0, 2.0],
        })

        recomendaciones = recommend_content_for_user(
            user_id=10,
            interactions=train,
            content_df=modelo["content_df"],
            tfidf_matrix=modelo["tfidf_matrix"],
            n=2,
        )

        self.assertEqual(
            int(recomendaciones.iloc[0]["movie_id"]),
            4,
        )

        # Excluir todas las películas calificadas.
        self.assertTrue(
            set(recomendaciones["movie_id"]).isdisjoint(
                {1, 3}
            )
        )

        # El DataFrame original debe permanecer intacto.
        self.assertNotIn(
            "content_soup",
            peliculas.columns,
        )

    
if __name__ == "__main__":
    unittest.main()
