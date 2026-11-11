import unittest
import pandas as pd
import json

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.transformation.bronze_to_silver import * 

from src.ingestion.watchmode import *

from src.utils.silver_runs import *


class BronzeToSilverTests(unittest.TestCase):

    ### FUNCION QUE TIENE RATINGS VALIDOS PERO APUNTA A UN TAG QUE TIENE PELICULA INEXISTENTE.
    def test_pipeline_movielens_bloquea_tags_huerfanos(self):

        # Ratings válidos, pero un tag apunta
        # a una película que no existe.
        bronze = {
            "movies": pd.DataFrame({
                "movieId": [1, 2],
                "title": [
                    "Pelicula A",
                    "Pelicula B",
                ],
                "genres": [
                    "Drama",
                    "Comedy",
                ],
            }),
            "ratings": pd.DataFrame({
                "userId": [10],
                "movieId": [1],
                "rating": [4.5],
                "timestamp": [1700000000],
            }),
            "tags": pd.DataFrame({
                "userId": [11],
                "movieId": [999],
                "tag": ["Suspenso"],
                "timestamp": [1700000000],
            }),
            "links": pd.DataFrame({
                "movieId": [1, 2],
                "imdbId": ["0000550", "0000600"],
                "tmdbId": [550, 600],
            }),
        }

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "movielens"

            with (
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_movielens_bronze",
                    return_value=bronze,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_movielens_silver",
                ) as mock_parquet,
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_movielens_silver_metadata",
                ) as mock_metadata,
            ):

                with self.assertRaisesRegex(
                    ValueError,
                    "referencias a películas inexistentes",
                ):
                    run_movielens_bronze_to_silver(
                        output_dir=destino,
                    )

            # Ningún archivo debe guardarse.
            mock_parquet.assert_not_called()
            mock_metadata.assert_not_called()

            self.assertFalse(destino.exists())

    
    def test_movielens_rechaza_tags_de_peliculas_inexistentes(self):

        movies = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        tags_validos = pd.DataFrame({
            "user_id": [10, 11],
            "movie_id": [1, 2],
            "tag": ["Drama", "Suspenso"],
        })

        # Todas las etiquetas tienen películas válidas.
        validate_movielens_tag_references(
            movies=movies,
            tags=tags_validos,
        )

        tags_invalidos = pd.DataFrame({
            "user_id": [10, 11],
            "movie_id": [1, 999],
            "tag": ["Drama", "Ciencia ficcion"],
        })

        # La película 999 no está en MovieLens.
        with self.assertRaisesRegex(
            ValueError,
            "referencias a películas inexistentes",
        ):
            validate_movielens_tag_references(
                movies=movies,
                tags=tags_invalidos,
            )

    
    def test_watchmode_sin_plataformas_conserva_columnas(self):

        registros = [
            {"tmdb_id": 550, "sources": []},
            {"tmdb_id": 600, "sources": []},
        ]

        availability = transform_watchmode_sources(
            registros
        )

        # No existen plataformas disponibles.
        self.assertTrue(availability.empty)

        # Aun así, debemos conservar la estructura.
        self.assertEqual(
            list(availability.columns),
            [
                "tmdb_id",
                "source_id",
                "provider_name",
                "type",
                "region",
                "format",
                "price",
                "web_url",
                "ios_url",
                "android_url",
            ],
        )

        # La limpieza no debe eliminar las columnas.
        availability_clean = clean_watchmode_sources(
            availability
        )

        self.assertEqual(
            list(availability_clean.columns),
            list(availability.columns),
        )

        # Las películas consultadas deben conservar
        # su estado, aunque no tengan plataformas.
        status = transform_watchmode_status(
            registros
        )

        self.assertEqual(len(status), 2)

        self.assertEqual(
            status["has_availability"].tolist(),
            [False, False],
        )

        self.assertEqual(
            status["sources_count"].tolist(),
            [0, 0],
        )

    
    def test_watchmode_parquet_vacio_conserva_esquema(self):

        # Dos películas consultadas sin plataformas.
        registros = [
            {"tmdb_id": 550, "sources": []},
            {"tmdb_id": 600, "sources": []},
        ]

        availability = transform_watchmode_sources(
            registros
        )

        availability = clean_watchmode_sources(
            availability
        )

        # Comprobar tipos antes de guardar.
        self.assertEqual(
            str(availability["tmdb_id"].dtype),
            "Int64",
        )

        self.assertEqual(
            str(availability["provider_name"].dtype),
            "string",
        )

        self.assertEqual(
            str(availability["price"].dtype),
            "Float64",
        )

        # Guardar y recuperar un Parquet temporal.
        with TemporaryDirectory() as temporal:

            ruta = (
                Path(temporal)
                / "availability.parquet"
            )

            availability.to_parquet(
                ruta,
                index=False,
            )

            recuperado = pd.read_parquet(ruta)

        # El archivo sigue sin plataformas.
        self.assertTrue(recuperado.empty)

        # Conserva las diez columnas.
        self.assertEqual(
            list(recuperado.columns),
            list(availability.columns),
        )

        # Y conserva los tipos relevantes.
        self.assertEqual(
            str(recuperado["tmdb_id"].dtype),
            "Int64",
        )

        self.assertEqual(
            str(recuperado["provider_name"].dtype),
            "string",
        )

        self.assertEqual(
            str(recuperado["price"].dtype),
            "Float64",
        )

    
    def test_tmdb_rechaza_dataset_vacio(self):


        # Simular que TMDb no entrega películas.
        registros = []

        df = transform_tmdb_movies(
            registros
        )

        # Silver debe detenerse con un mensaje claro.
        with self.assertRaisesRegex(
            ValueError,
            "no contiene películas",
        ):
            clean_tmdb_movies(df)

    
    def test_tmdb_rechaza_todos_ids_invalidos(self):

        # Existen registros, pero ninguno
        # tiene un identificador TMDb válido.
        registros = [
            {
                "id": None,
                "title": "Pelicula A",
            },
            {
                "id": "abc",
                "title": "Pelicula B",
            },
            {
                "title": "Pelicula C",
            },
        ]

        df = transform_tmdb_movies(registros)

        # Inicialmente existen tres películas.
        self.assertEqual(len(df), 3)

        # La limpieza debe detectar que ninguna
        # tiene un identificador utilizable.
        with self.assertRaisesRegex(
            ValueError,
            "no contiene películas con identificadores válidos",
        ):
            clean_tmdb_movies(df)

    
    def test_ratings_bloquea_particiones_obsoletas(self):

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            ratings_dir = carpeta / "ratings"
            ratings_dir.mkdir()

            # Simular archivos de una ejecución anterior.
            anterior_0 = ratings_dir / "part_000.parquet"
            anterior_1 = ratings_dir / "part_001.parquet"

            anterior_0.write_bytes(b"version_anterior_0")
            anterior_1.write_bytes(b"version_anterior_1")

            # La nueva ejecución solo necesita
            # una partición de dos registros.
            ratings = pd.DataFrame({
                "user_id": [1, 2],
                "movie_id": [550, 600],
                "rating": [4.0, 5.0],
            })

            with patch(
                "src.transformation.bronze_to_silver."
                "SILVER_MOVIELENS_DIR",
                carpeta,
            ):

                with self.assertRaisesRegex(
                    RuntimeError,
                    "particiones antiguas",
                ):
                    save_ratings_partitioned(
                        df=ratings,
                        rows_per_file=2,
                    )

            # Ningún archivo anterior debe cambiar.
            self.assertEqual(
                anterior_0.read_bytes(),
                b"version_anterior_0",
            )

            self.assertEqual(
                anterior_1.read_bytes(),
                b"version_anterior_1",
            )

    
    def test_ratings_utiliza_directorio_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        ratings = pd.DataFrame({
            "user_id": [1, 2, 3],
            "movie_id": [550, 600, 700],
            "rating": [4.0, 5.0, 3.5],
        })

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            carpetas = build_silver_run_directories(
                silver_root=raiz,
                run_id=run_id,
            )

            destino = carpetas["movielens"] / "ratings"

            rutas = save_ratings_partitioned(
                df=ratings,
                rows_per_file=2,
                output_dir=destino,
            )

            # Tres registros generan dos particiones.
            self.assertEqual(len(rutas), 2)

            self.assertEqual(
                rutas[0],
                destino / "part_000.parquet",
            )

            self.assertEqual(
                rutas[1],
                destino / "part_001.parquet",
            )

            # Los archivos fueron creados.
            for ruta in rutas:
                self.assertTrue(ruta.is_file())

            # Verificar cantidad de registros.
            primera = pd.read_parquet(rutas[0])
            segunda = pd.read_parquet(rutas[1])

            self.assertEqual(len(primera), 2)
            self.assertEqual(len(segunda), 1)

            # No se creó otra ejecución.
            self.assertEqual(
                len(list((raiz / "runs").iterdir())),
                1,
            )

    
    def test_movielens_guarda_todos_archivos_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        datasets = {
            "movies": pd.DataFrame({
                "movie_id": [1, 2],
                "title": ["Pelicula A", "Pelicula B"],
                "genres": ["Drama", "Action"],
            }),

            "links": pd.DataFrame({
                "movie_id": [1, 2],
                "tmdb_id": pd.Series(
                    [550, 600],
                    dtype="Int64",
                ),
                "imdb_id": ["0111161", "0123456"],
            }),

            "tags": pd.DataFrame({
                "user_id": [10],
                "movie_id": [1],
                "tag": ["Emocionante"],
            }),

            "ratings": pd.DataFrame({
                "user_id": [10, 20, 30],
                "movie_id": [1, 2, 1],
                "rating": [4.5, 5.0, 3.5],
            }),
        }

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["movielens"]

            rutas = save_movielens_silver(
                datasets=datasets,
                output_dir=destino,
            )

            # Verificar los archivos principales.
            for nombre in ("movies", "links", "tags"):

                self.assertEqual(
                    rutas[nombre],
                    destino / f"{nombre}.parquet",
                )

                self.assertTrue(
                    rutas[nombre].is_file()
                )

            # Verificar que ratings utiliza
            # la misma carpeta de ejecución.
            self.assertEqual(
                len(rutas["ratings"]),
                1,
            )

            self.assertEqual(
                rutas["ratings"][0],
                destino / "ratings" / "part_000.parquet",
            )

            # Recuperar y verificar los registros.
            movies = pd.read_parquet(
                rutas["movies"]
            )

            ratings = pd.read_parquet(
                rutas["ratings"][0]
            )

            self.assertEqual(len(movies), 2)
            self.assertEqual(len(ratings), 3)

            # No se generó una segunda carpeta
            # de ejecución.
            self.assertEqual(
                len(list((Path(temporal) / "runs").iterdir())),
                1,
            )

    
    def test_metadata_movielens_guarda_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        datasets = {
            "movies": pd.DataFrame({
                "movie_id": [1, 2],
                "title": ["Pelicula A", "Pelicula B"],
            }),
            "links": pd.DataFrame({
                "movie_id": [1, 2],
                "tmdb_id": [550, 600],
            }),
            "tags": pd.DataFrame({
                "movie_id": [1],
                "tag": ["Drama"],
            }),
            "ratings": pd.DataFrame({
                "user_id": [10, 20, 30],
                "movie_id": [1, 2, 1],
                "rating": [4.0, 5.0, 3.5],
            }),
        }

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["movielens"]

            ruta = save_movielens_silver_metadata(
                datasets=datasets,
                output_dir=destino,
            )

            self.assertEqual(
                ruta,
                destino / "_metadata.json",
            )

            self.assertTrue(ruta.is_file())

            metadata = json.loads(
                ruta.read_text(encoding="utf-8")
            )

            self.assertEqual(
                metadata["fuente"],
                "MovieLens",
            )

            self.assertEqual(
                metadata["capa"],
                "silver",
            )

            self.assertEqual(
                metadata["formato"],
                "parquet",
            )

            self.assertEqual(
                metadata["datasets"]["movies"]["filas"],
                2,
            )

            self.assertEqual(
                metadata["datasets"]["ratings"]["filas"],
                3,
            )

            self.assertEqual(
                metadata["estado"],
                "success",
            )

    
    def test_pipeline_movielens_usa_misma_carpeta_ejecucion(self):

        bronze_simulado = {
            "movies": pd.DataFrame({
                "movieId": [1],
                "title": ["Pelicula A"],
                "genres": ["Drama"],
            }),
            "ratings": pd.DataFrame({
                "userId": [10],
                "movieId": [1],
                "rating": [4.5],
                "timestamp": [1000000000],
            }),
            "tags": pd.DataFrame({
                "userId": [10],
                "movieId": [1],
                "tag": ["Interesante"],
                "timestamp": [1000000000],
            }),
            "links": pd.DataFrame({
                "movieId": [1],
                "imdbId": ["0111161"],
                "tmdbId": [550],
            }),
        }

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["movielens"]

            with (
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_movielens_bronze",
                    return_value=bronze_simulado,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_movielens_silver",
                    return_value={
                        "movies": destino / "movies.parquet",
                    },
                ) as mock_parquet,
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_movielens_silver_metadata",
                    return_value=destino / "_metadata.json",
                ) as mock_metadata,
            ):

                resultado = run_movielens_bronze_to_silver(
                    output_dir=destino,
                )

        mock_parquet.assert_called_once()
        mock_metadata.assert_called_once()

        # Ambos guardados reciben el mismo destino.
        self.assertEqual(
            mock_parquet.call_args.kwargs["output_dir"],
            destino,
        )

        self.assertEqual(
            mock_metadata.call_args.kwargs["output_dir"],
            destino,
        )

        # La transformación continúa funcionando.
        self.assertEqual(
            resultado["movies"].iloc[0]["movie_id"],
            1,
        )

        self.assertEqual(
            resultado["links"].iloc[0]["tmdb_id"],
            550,
        )

        self.assertEqual(
            resultado["ratings"].iloc[0]["rating"],
            4.5,
        )

    
    def test_tmdb_guarda_parquet_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        peliculas = pd.DataFrame({
            "tmdb_id": [550, 600],
            "title": [
                "Pelicula A",
                "Pelicula B",
            ],
            "overview": [
                "Historia de ejemplo A",
                "Historia de ejemplo B",
            ],
        })

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["tmdb"]

            ruta = save_tmdb_silver(
                df=peliculas,
                output_dir=destino,
            )

            # Verificar la ruta.
            self.assertEqual(
                ruta,
                destino / "movies_tmdb.parquet",
            )

            # Verificar que se guardó el archivo.
            self.assertTrue(ruta.is_file())

            # Recuperar el contenido.
            recuperado = pd.read_parquet(ruta)

            self.assertEqual(
                len(recuperado),
                2,
            )

            self.assertEqual(
                recuperado["tmdb_id"].tolist(),
                [550, 600],
            )

            self.assertEqual(
                recuperado["title"].tolist(),
                ["Pelicula A", "Pelicula B"],
            )

            # No debe crear carpetas de otras fuentes.
            self.assertFalse(
                carpetas["watchmode"].exists()
            )

            self.assertFalse(
                carpetas["integrated"].exists()
            )

    
    def test_tmdb_metadata_guarda_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        peliculas = pd.DataFrame({
            "tmdb_id": [550, 600],
            "title": ["Pelicula A", "Pelicula B"],
            "overview": ["Historia A", "Historia B"],
        })

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["tmdb"]

            ruta = save_tmdb_silver_metadata(
                df=peliculas,
                output_dir=destino,
            )

            self.assertEqual(
                ruta,
                destino / "_metadata.json",
            )

            self.assertTrue(ruta.is_file())

            metadata = json.loads(
                ruta.read_text(encoding="utf-8")
            )

            self.assertEqual(
                metadata["fuente"],
                "TMDb",
            )

            self.assertEqual(
                metadata["capa"],
                "silver",
            )

            self.assertEqual(
                metadata["formato"],
                "parquet",
            )

            self.assertEqual(
                metadata["filas"],
                2,
            )

            self.assertEqual(
                metadata["columnas"],
                3,
            )

            self.assertEqual(
                metadata["estado"],
                "success",
            )

    
    def test_pipeline_tmdb_utiliza_misma_carpeta_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        # Simular la respuesta original de TMDb.
        registros = [
            {
                "id": 550,
                "title": "Pelicula A",
                "overview": "Historia de ejemplo",
                "genres": [
                    {"name": "Drama"}
                ],
            }
        ]

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["tmdb"]

            with (
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_tmdb_bronze",
                    return_value=registros,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_tmdb_silver",
                    return_value=destino / "movies_tmdb.parquet",
                ) as mock_parquet,
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_tmdb_silver_metadata",
                    return_value=destino / "_metadata.json",
                ) as mock_metadata,
            ):

                resultado = run_tmdb_bronze_to_silver(
                    output_dir=destino,
                )

        # Ambas funciones de guardado se ejecutaron.
        mock_parquet.assert_called_once()
        mock_metadata.assert_called_once()

        # Las dos recibieron la misma carpeta.
        self.assertEqual(
            mock_parquet.call_args.kwargs["output_dir"],
            destino,
        )

        self.assertEqual(
            mock_metadata.call_args.kwargs["output_dir"],
            destino,
        )

        # Comprobar la transformación.
        self.assertEqual(
            len(resultado),
            1,
        )

        self.assertEqual(
            resultado.iloc[0]["tmdb_id"],
            550,
        )

        self.assertEqual(
            resultado.iloc[0]["title"],
            "Pelicula A",
        )

        self.assertEqual(
            resultado.iloc[0]["genres"],
            ["Drama"],
        )

    
    def test_watchmode_guarda_parquet_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        availability = pd.DataFrame({
            "tmdb_id": [550, 550],
            "source_id": [1, 2],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
            ],
            "region": ["PE", "PE"],
        })

        status = pd.DataFrame({
            "tmdb_id": [550, 600],
            "has_availability": [True, False],
            "sources_count": [2, 0],
        })

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["watchmode"]

            rutas = save_watchmode_silver(
                availability_df=availability,
                status_df=status,
                output_dir=destino,
            )

            # Comprobar que ambos archivos
            # pertenecen a la misma ejecución.
            self.assertEqual(
                rutas["availability"],
                destino / "availability.parquet",
            )

            self.assertEqual(
                rutas["availability_status"],
                destino / "availability_status.parquet",
            )

            self.assertTrue(
                rutas["availability"].is_file()
            )

            self.assertTrue(
                rutas["availability_status"].is_file()
            )

            # Recuperar los archivos guardados.
            availability_leida = pd.read_parquet(
                rutas["availability"]
            )

            status_leido = pd.read_parquet(
                rutas["availability_status"]
            )

            self.assertEqual(
                len(availability_leida),
                2,
            )

            self.assertEqual(
                len(status_leido),
                2,
            )

            # Película 600: consultada sin plataformas.
            pelicula_600 = status_leido.loc[
                status_leido["tmdb_id"] == 600
            ].iloc[0]

            self.assertFalse(
                pelicula_600["has_availability"]
            )

            self.assertEqual(
                pelicula_600["sources_count"],
                0,
            )

            # No se crean carpetas de otras fuentes.
            self.assertFalse(
                carpetas["integrated"].exists()
            )
    
    
    def test_pipeline_watchmode_usa_misma_carpeta_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        registros = [
            {
                "tmdb_id": 550,
                "sources": [
                    {
                        "source_id": 1,
                        "name": "Plataforma A",
                        "type": "sub",
                        "region": "PE",
                    }
                ],
            },
            {
                "tmdb_id": 600,
                "sources": [],
            },
        ]

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["watchmode"]

            with (
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_watchmode_bronze",
                    return_value=registros,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_watchmode_silver",
                    return_value={
                        "availability": (
                            destino / "availability.parquet"
                        ),
                        "availability_status": (
                            destino / "availability_status.parquet"
                        ),
                    },
                ) as mock_parquet,
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_watchmode_silver_metadata",
                    return_value=destino / "_metadata.json",
                ) as mock_metadata,
            ):

                resultado = run_watchmode_bronze_to_silver(
                    output_dir=destino,
                )

        mock_parquet.assert_called_once()
        mock_metadata.assert_called_once()

        # Ambos guardados reciben la misma carpeta.
        self.assertEqual(
            mock_parquet.call_args.kwargs["output_dir"],
            destino,
        )

        self.assertEqual(
            mock_metadata.call_args.kwargs["output_dir"],
            destino,
        )

        # Comprobar la transformación de disponibilidad.
        availability = resultado["availability"]

        self.assertEqual(len(availability), 1)

        self.assertEqual(
            availability.iloc[0]["tmdb_id"],
            550,
        )

        # La película 600 fue consultada,
        # pero no tiene plataformas.
        status = resultado["availability_status"]

        pelicula_600 = status.loc[
            status["tmdb_id"] == 600
        ].iloc[0]

        self.assertFalse(
            pelicula_600["has_availability"]
        )

        self.assertEqual(
            pelicula_600["sources_count"],
            0,
        )

    
    def test_movielens_rechaza_ratings_de_peliculas_inexistentes(self):

        movies = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        ratings_validos = pd.DataFrame({
            "user_id": [10, 11, 12],
            "movie_id": [1, 2, 3],
            "rating": [4.5, 3.0, 5.0],
        })

        # Todas las películas existen.
        validate_movielens_rating_references(
            movies=movies,
            ratings=ratings_validos,
        )

        ratings_invalidos = pd.DataFrame({
            "user_id": [10, 11, 12],
            "movie_id": [1, 2, 999],
            "rating": [4.5, 3.0, 5.0],
        })

        # La película 999 no existe.
        with self.assertRaisesRegex(
            ValueError,
            "referencias a películas inexistentes",
        ):
            validate_movielens_rating_references(
                movies=movies,
                ratings=ratings_invalidos,
            )

    
    def test_pipeline_movielens_bloquea_ratings_huerfanos(self):

        # El catálogo contiene las películas 1 y 2.
        # El rating apunta a la película 999.
        bronze = {
            "movies": pd.DataFrame({
                "movieId": [1, 2],
                "title": [
                    "Pelicula A",
                    "Pelicula B",
                ],
                "genres": [
                    "Drama",
                    "Comedy",
                ],
            }),
            "ratings": pd.DataFrame({
                "userId": [10],
                "movieId": [999],
                "rating": [4.5],
                "timestamp": [1700000000],
            }),
            "tags": pd.DataFrame({
                "userId": [10],
                "movieId": [1],
                "tag": ["Drama"],
                "timestamp": [1700000000],
            }),
            "links": pd.DataFrame({
                "movieId": [1, 2],
                "imdbId": ["0000550", "0000600"],
                "tmdbId": [550, 600],
            }),
        }

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "movielens"

            with (
                patch(
                    "src.transformation.bronze_to_silver."
                    "load_movielens_bronze",
                    return_value=bronze,
                ),
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_movielens_silver",
                ) as mock_parquet,
                patch(
                    "src.transformation.bronze_to_silver."
                    "save_movielens_silver_metadata",
                ) as mock_metadata,
            ):

                with self.assertRaisesRegex(
                    ValueError,
                    "referencias a películas inexistentes",
                ):
                    run_movielens_bronze_to_silver(
                        output_dir=destino,
                    )

            # No debe intentar escribir ningún dataset.
            mock_parquet.assert_not_called()
            mock_metadata.assert_not_called()

            # Tampoco debe crear la carpeta de salida.
            self.assertFalse(destino.exists())

    
    def test_watchmode_metadata_guarda_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        availability = pd.DataFrame({
            "tmdb_id": [550, 550],
            "source_id": [1, 2],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
            ],
        })

        status = pd.DataFrame({
            "tmdb_id": [550, 600],
            "has_availability": [True, False],
            "sources_count": [2, 0],
        })

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["watchmode"]

            ruta = save_watchmode_silver_metadata(
                availability_df=availability,
                status_df=status,
                output_dir=destino,
            )

            self.assertEqual(
                ruta,
                destino / "_metadata.json",
            )

            self.assertTrue(ruta.is_file())

            metadata = json.loads(
                ruta.read_text(encoding="utf-8")
            )

            self.assertEqual(
                metadata["fuente"],
                "Watchmode",
            )

            self.assertEqual(
                metadata["capa"],
                "silver",
            )

            self.assertEqual(
                metadata["formato"],
                "parquet",
            )

            self.assertEqual(
                metadata["datasets"]["availability"]["filas"],
                2,
            )

            self.assertEqual(
                metadata["datasets"]["availability_status"]["filas"],
                2,
            )

            self.assertEqual(
                metadata["estado"],
                "success",
            )

    
    def test_tmdb_rechaza_identificadores_duplicados(self):

        # Dos registros diferentes con el mismo ID.
        registros = [
            {
                "id": 550,
                "title": "Pelicula A",
            },
            {
                "id": 550,
                "title": "Pelicula C",
            },
        ]

        # Convertir los JSON en una tabla.
        df = transform_tmdb_movies(
            registros
        )

        # La limpieza debe detectar el conflicto.
        with self.assertRaisesRegex(
            ValueError,
            "tmdb_id duplicados",
        ):
            clean_tmdb_movies(df)

    
    def test_links_tmdb_id_debe_ser_positivo(self):

        links = pd.DataFrame({
            "movieId": [1, 2, 3, 4],
            "imdbId": [
                "0111161",
                "0123456",
                "0234567",
                "0345678",
            ],
            "tmdbId": [862, 0, -50, None],
        })

        resultado = transform_links(links)

        # No debemos perder películas.
        self.assertEqual(
            len(resultado),
            4,
        )

        # El identificador válido se conserva.
        self.assertEqual(
            resultado.iloc[0]["tmdb_id"],
            862,
        )

        # Los identificadores inválidos o
        # ausentes quedan como desconocidos.
        self.assertTrue(
            pd.isna(resultado.iloc[1]["tmdb_id"])
        )

        self.assertTrue(
            pd.isna(resultado.iloc[2]["tmdb_id"])
        )

        self.assertTrue(
            pd.isna(resultado.iloc[3]["tmdb_id"])
        )

        # Conservamos los identificadores MovieLens.
        self.assertEqual(
            resultado["movie_id"].tolist(),
            [1, 2, 3, 4],
        )

    
    def test_links_rechaza_identificadores_conflictivos(self):

        links = pd.DataFrame({
            "movieId": [1, 1, 2],
            "imdbId": [
                "0111161",
                "0111161",
                "0123456",
            ],
            "tmdbId": [862, 999, 200],
        })

        with self.assertRaisesRegex(
            ValueError,
            "movieId duplicados",
        ):
            transform_links(links)

    
    def test_watchmode_solo_carga_region_verificada(self):

        with TemporaryDirectory() as temporal:

            with patch(
                "src.ingestion.watchmode."
                "BRONZE_WATCHMODE_DIR",
                Path(temporal),
            ):

                # Película válida y con procedencia verificada
                save_watchmode_raw(
                    tmdb_id=550,
                    data=[{"name": "Netflix"}],
                    region="PE",
                )

                save_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

                # Película sin metadatos de procedencia
                save_watchmode_raw(
                    tmdb_id=600,
                    data=[{"name": "Prime Video"}],
                    region="PE",
                )

                resultado = load_watchmode_bronze(
                    region="PE"
                )

        self.assertEqual(len(resultado), 1)

        self.assertEqual(
            resultado[0]["tmdb_id"],
            550,
        )

        self.assertEqual(
            resultado[0]["sources"],
            [{"name": "Netflix"}],
        )

    
    def test_imdb_conserva_ceros_iniciales(self):

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal) / "ml-32m"
            carpeta.mkdir(parents=True)

            # Crear archivos mínimos para la prueba.
            (carpeta / "movies.csv").write_text(
                "movieId,title,genres\n"
                "1,Pelicula A,Drama\n",
                encoding="utf-8",
            )

            (carpeta / "ratings.csv").write_text(
                "userId,movieId,rating,timestamp\n"
                "10,1,4.5,1000000000\n",
                encoding="utf-8",
            )

            (carpeta / "tags.csv").write_text(
                "userId,movieId,tag,timestamp\n"
                "10,1,Drama,1000000000\n",
                encoding="utf-8",
            )

            (carpeta / "links.csv").write_text(
                "movieId,imdbId,tmdbId\n"
                "1,0111161,862\n",
                encoding="utf-8",
            )

            with patch(
                "src.transformation.bronze_to_silver."
                "BRONZE_MOVIELENS_DIR",
                Path(temporal),
            ):
                datasets = load_movielens_bronze()

            resultado = transform_links(
                datasets["links"]
            )

        self.assertEqual(
            resultado.iloc[0]["imdb_id"],
            "0111161",
        )

        self.assertEqual(
            resultado.iloc[0]["movie_id"],
            1,
        )

        self.assertEqual(
            resultado.iloc[0]["tmdb_id"],
            862,
        )

        self.assertTrue(
            pd.api.types.is_string_dtype(
                resultado["imdb_id"]
            )
        )


if __name__ == "__main__":
    unittest.main()
