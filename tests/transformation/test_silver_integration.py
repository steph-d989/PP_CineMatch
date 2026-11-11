import unittest
import pandas as pd
import json

from pathlib import Path
from tempfile import TemporaryDirectory

from src.transformation.silver_integration import (
    integrate_movielens,
    integrate_tmdb,
    integrate_watchmode_status,
    build_integrated_movies,
    run_silver_integration,
    validate_integrated_catalog,
    save_integrated_metadata,
    validate_watchmode_source_counts,
)

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.utils.silver_runs import (
    build_silver_run_directories,
)

class SilverIntegrationTests(unittest.TestCase):

    
    def test_integracion_completa_silver(self):

        # 1. Películas originales de MovieLens.
        movies = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
            "genres": [
                "Action",
                "Comedy",
                "Drama",
            ],
        })

        # 2. Identificadores que relacionan
        # MovieLens con TMDb.
        links = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "tmdb_id": pd.Series(
                [550, 600, pd.NA],
                dtype="Int64",
            ),
            "imdb_id": [
                "0111161",
                "0123456",
                "0234567",
            ],
        })

        # 3. Información de TMDb.
        tmdb = pd.DataFrame({
            "tmdb_id": pd.Series(
                [550, 600],
                dtype="Int64",
            ),
            "title": [
                "Titulo TMDb A",
                "Titulo TMDb B",
            ],
            "imdb_id": [
                "tt0111161",
                "tt0123456",
            ],
            "genres": [
                ["Action"],
                ["Comedy"],
            ],
        })

        # 4. Disponibilidad consultada en Watchmode.
        watchmode_status = pd.DataFrame({
            "tmdb_id": pd.Series(
                [550, 600],
                dtype="Int64",
            ),
            "has_availability": [True, False],
            "sources_count": [2, 0],
        })

        datasets = {
            "movies": movies,
            "links": links,
            "tmdb": tmdb,
            "watchmode_status": watchmode_status,
        }

        # Ejecutar la integración completa.
        resultado = build_integrated_movies(
            datasets
        )

        # Debemos conservar las tres películas.
        self.assertEqual(len(resultado), 3)

        self.assertEqual(
            resultado["movie_id"].nunique(),
            3,
        )

        pelicula_1 = resultado.loc[
            resultado["movie_id"] == 1
        ].iloc[0]

        pelicula_2 = resultado.loc[
            resultado["movie_id"] == 2
        ].iloc[0]

        pelicula_3 = resultado.loc[
            resultado["movie_id"] == 3
        ].iloc[0]

        # Película 1: TMDb + disponibilidad.
        self.assertEqual(
            pelicula_1["canonical_title"],
            "Titulo TMDb A",
        )
        self.assertTrue(
            pelicula_1["watchmode_checked"]
        )
        self.assertTrue(
            pelicula_1["has_availability"]
        )
        self.assertEqual(
            pelicula_1["sources_count"],
            2,
        )

        # Película 2: consultada sin plataformas.
        self.assertTrue(
            pelicula_2["watchmode_checked"]
        )
        self.assertFalse(
            pelicula_2["has_availability"]
        )
        self.assertEqual(
            pelicula_2["sources_count"],
            0,
        )

        # Película 3: no tiene TMDb ni Watchmode.
        self.assertEqual(
            pelicula_3["canonical_title"],
            "Pelicula C",
        )
        self.assertFalse(
            pelicula_3["watchmode_checked"]
        )
        self.assertTrue(
            pd.isna(
                pelicula_3["has_availability"]
            )
        )
        self.assertTrue(
            pd.isna(
                pelicula_3["sources_count"]
            )
        )

    
    def test_silver_integracion_parquet_completa(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from src.transformation.silver_integration import (
            load_silver_datasets,
            build_integrated_movies,
            save_integrated_movies,
        )

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            movielens_dir = raiz / "movielens"
            tmdb_dir = raiz / "tmdb"
            watchmode_dir = raiz / "watchmode"
            integrated_dir = raiz / "integrated"

            for carpeta in (
                movielens_dir,
                tmdb_dir,
                watchmode_dir,
                integrated_dir,
            ):
                carpeta.mkdir(parents=True)

            # 1. Crear datos Silver simulados.
            movies = pd.DataFrame({
                "movie_id": [1, 2, 3],
                "title": [
                    "Pelicula A",
                    "Pelicula B",
                    "Pelicula C",
                ],
                "genres": [
                    "Drama",
                    "Comedy",
                    "Action",
                ],
            })

            links = pd.DataFrame({
                "movie_id": [1, 2, 3],
                "tmdb_id": pd.Series(
                    [550, 600, pd.NA],
                    dtype="Int64",
                ),
            })

            tmdb = pd.DataFrame({
                "tmdb_id": pd.Series(
                    [550, 600],
                    dtype="Int64",
                ),
                "title": [
                    "Titulo TMDb A",
                    "Titulo TMDb B",
                ],
                "imdb_id": [
                    "tt0111161",
                    "tt0123456",
                ],
                "genres": [
                    ["Drama"],
                    ["Comedy"],
                ],
            })

            watchmode_status = pd.DataFrame({
                "tmdb_id": pd.Series(
                    [550, 600],
                    dtype="Int64",
                ),
                "has_availability": [True, False],
                "sources_count": [2, 0],
            })

            # 2. Guardar archivos Parquet temporales.
            movies.to_parquet(
                movielens_dir / "movies.parquet",
                index=False,
            )

            links.to_parquet(
                movielens_dir / "links.parquet",
                index=False,
            )

            tmdb.to_parquet(
                tmdb_dir / "movies_tmdb.parquet",
                index=False,
            )

            watchmode_status.to_parquet(
                watchmode_dir / "availability_status.parquet",
                index=False,
            )

            # 3. Usar las funciones reales con
            # rutas temporales, no con el proyecto.
            with (
                patch(
                    "src.transformation.silver_integration."
                    "SILVER_MOVIELENS_DIR",
                    movielens_dir,
                ),
                patch(
                    "src.transformation.silver_integration."
                    "SILVER_TMDB_DIR",
                    tmdb_dir,
                ),
                patch(
                    "src.transformation.silver_integration."
                    "SILVER_WATCHMODE_DIR",
                    watchmode_dir,
                ),
                patch(
                    "src.transformation.silver_integration."
                    "SILVER_INTEGRATED_DIR",
                    integrated_dir,
                ),
            ):

                datasets = load_silver_datasets()

                integrado = build_integrated_movies(
                    datasets
                )

                ruta = save_integrated_movies(
                    integrado
                )

                # 4. Leer el archivo que se generó.
                recuperado = pd.read_parquet(ruta)

            # 5. Verificar el resultado.
            self.assertTrue(ruta.is_file())

            self.assertEqual(len(recuperado), 3)

            self.assertEqual(
                recuperado["movie_id"].nunique(),
                3,
            )

            pelicula_1 = recuperado.loc[
                recuperado["movie_id"] == 1
            ].iloc[0]

            pelicula_2 = recuperado.loc[
                recuperado["movie_id"] == 2
            ].iloc[0]

            pelicula_3 = recuperado.loc[
                recuperado["movie_id"] == 3
            ].iloc[0]

            self.assertEqual(
                pelicula_1["canonical_title"],
                "Titulo TMDb A",
            )

            self.assertTrue(
                pelicula_1["has_availability"]
            )

            self.assertFalse(
                pelicula_2["has_availability"]
            )

            self.assertEqual(
                pelicula_3["canonical_title"],
                "Pelicula C",
            )

            self.assertTrue(
                pd.isna(
                    pelicula_3["has_availability"]
                )
            )

            self.assertFalse(
                pelicula_3["watchmode_checked"]
            )
    
    def test_integrated_guarda_parquet_por_ejecucion(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.utils.silver_runs import (
            build_silver_run_directories,
        )

        from src.transformation.silver_integration import (
            save_integrated_movies,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "tmdb_id": pd.Series(
                [550, 600, pd.NA],
                dtype="Int64",
            ),
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
            "has_availability": pd.Series(
                [True, False, pd.NA],
                dtype="boolean",
            ),
        })

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["integrated"]

            ruta = save_integrated_movies(
                df=peliculas,
                output_dir=destino,
            )

            # Verificar la ubicación del archivo.
            self.assertEqual(
                ruta,
                destino / "movies_integrated.parquet",
            )

            self.assertTrue(ruta.is_file())

            # Leer nuevamente el archivo Parquet.
            recuperado = pd.read_parquet(ruta)

            self.assertEqual(len(recuperado), 3)

            self.assertEqual(
                recuperado["movie_id"].tolist(),
                [1, 2, 3],
            )

            self.assertEqual(
                recuperado["canonical_title"].tolist(),
                [
                    "Pelicula A",
                    "Pelicula B",
                    "Pelicula C",
                ],
            )

            # La película 3 conserva disponibilidad
            # desconocida, no False.
            pelicula_3 = recuperado.loc[
                recuperado["movie_id"] == 3
            ].iloc[0]

            self.assertTrue(
                pd.isna(pelicula_3["has_availability"])
            )

            # No se crean carpetas de otras fuentes.
            self.assertFalse(
                carpetas["watchmode"].exists()
            )

    
    def test_integrated_metadata_guarda_por_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        peliculas = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
            "tmdb_title": [
                "Titulo TMDb A",
                "Titulo TMDb B",
                None,
            ],
            "watchmode_checked": [
                True,
                True,
                False,
            ],
            "has_availability": pd.Series(
                [True, False, pd.NA],
                dtype="boolean",
            ),
        })

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["integrated"]

            ruta = save_integrated_metadata(
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
                metadata["dataset"],
                "movies_integrated",
            )

            self.assertEqual(
                metadata["capa"],
                "silver",
            )

            self.assertEqual(
                metadata["filas"],
                3,
            )

            self.assertEqual(
                metadata["con_tmdb"],
                2,
            )

            self.assertEqual(
                metadata["watchmode_consultadas"],
                2,
            )

            self.assertEqual(
                metadata["con_disponibilidad"],
                1,
            )

            self.assertEqual(
                metadata["estado"],
                "success",
            )

    
    def test_pipeline_integrated_usa_misma_carpeta_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )


        peliculas = pd.DataFrame({
            "movie_id": [1, 2],
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
            ],
            "tmdb_title": [
                "Titulo TMDb A",
                None,
            ],
            "watchmode_checked": pd.Series(
                [True, False],
                dtype="boolean",
            ),
            "has_availability": pd.Series(
                [True, pd.NA],
                dtype="boolean",
            ),
            "sources_count": pd.Series(
                [1, pd.NA],
                dtype="Int64",
            ),
        })


        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["integrated"]

            with (
                patch(
                    "src.transformation.silver_integration."
                    "load_silver_datasets",
                    return_value={"datos_simulados": True},
                ) as mock_carga,
                patch(
                    "src.transformation.silver_integration."
                    "build_integrated_movies",
                    return_value=peliculas,
                ) as mock_integracion,
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_movies",
                    return_value=destino / "movies_integrated.parquet",
                ) as mock_parquet,
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_metadata",
                    return_value=destino / "_metadata.json",
                ) as mock_metadata,
            ):

                resultado = run_silver_integration(
                    output_dir=destino,
                )

        mock_carga.assert_called_once()
        mock_integracion.assert_called_once()
        mock_parquet.assert_called_once()
        mock_metadata.assert_called_once()

        # Ambos guardados reciben el mismo directorio.
        self.assertEqual(
            mock_parquet.call_args.kwargs["output_dir"],
            destino,
        )

        self.assertEqual(
            mock_metadata.call_args.kwargs["output_dir"],
            destino,
        )

        # El catálogo sigue teniendo los registros.
        self.assertEqual(len(resultado), 2)

        self.assertEqual(
            resultado.iloc[0]["canonical_title"],
            "Pelicula A",
        )

        self.assertTrue(
            pd.isna(
                resultado.iloc[1]["has_availability"]
            )
        )

    def test_watchmode_sources_count_coincide_con_disponibilidad(self):

        availability = pd.DataFrame({
            "tmdb_id": [550, 550],
            "provider_name": [
                "Plataforma A",
                "Plataforma B",
            ],
        })

        status = pd.DataFrame({
            "tmdb_id": [550, 600],
            "sources_count": [2, 0],
        })

        # Escenario correcto: 2 plataformas
        # para 550 y ninguna para 600.
        validate_watchmode_source_counts(
            availability=availability,
            status=status,
        )

        # Escenario incorrecto: declara 3
        # plataformas cuando solo existen 2.
        status_incorrecto = status.copy()

        status_incorrecto.loc[
            status_incorrecto["tmdb_id"] == 550,
            "sources_count",
        ] = 3

        with self.assertRaisesRegex(
            ValueError,
            "sources_count no coincide",
        ):
            validate_watchmode_source_counts(
                availability=availability,
                status=status_incorrecto,
            )

        # Tampoco se permiten plataformas
        # de películas sin registro de estado.
        availability_huerfana = pd.concat(
            [
                availability,
                pd.DataFrame({
                    "tmdb_id": [700],
                    "provider_name": ["Plataforma C"],
                }),
            ],
            ignore_index=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "plataformas sin estado",
        ):
            validate_watchmode_source_counts(
                availability=availability_huerfana,
                status=status,
            )

    def test_pipeline_integrated_lee_y_guarda_misma_ejecucion(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        
        peliculas = pd.DataFrame({
            "movie_id": [1],
            "canonical_title": ["Pelicula A"],
            "tmdb_title": ["Titulo TMDb A"],
            "watchmode_checked": pd.Series(
                [True],
                dtype="boolean",
            ),
            "has_availability": pd.Series(
                [True],
                dtype="boolean",
            ),
            "sources_count": pd.Series(
                [1],
                dtype="Int64",
            ),
        })


        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            destino = carpetas["integrated"]

            with (
                patch(
                    "src.transformation.silver_integration."
                    "load_silver_datasets",
                    return_value={"datos_simulados": True},
                ) as mock_carga,
                patch(
                    "src.transformation.silver_integration."
                    "build_integrated_movies",
                    return_value=peliculas,
                ),
                patch(
                    "src.transformation.silver_integration."
                    "build_integrated_movies",
                    return_value=peliculas,
                ),
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_movies",
                    return_value=destino / "movies_integrated.parquet",
                ) as mock_parquet,
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_metadata",
                    return_value=destino / "_metadata.json",
                ) as mock_metadata,
            ):

                resultado = run_silver_integration(
                    run_dirs=carpetas,
                )


            mock_carga.assert_called_once_with(
                run_dirs=carpetas,
                include_availability=True,
            )


            # Ambos archivos se guardan en Integrated
            # de la misma ejecución.
            self.assertEqual(
                mock_parquet.call_args.kwargs["output_dir"],
                destino,
            )

            self.assertEqual(
                mock_metadata.call_args.kwargs["output_dir"],
                destino,
            )

            self.assertEqual(
                len(resultado),
                1,
            )

            self.assertEqual(
                resultado.iloc[0]["canonical_title"],
                "Pelicula A",
            )


    
    def test_pipeline_integrated_exige_availability(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from src.utils.silver_runs import (
            build_silver_run_directories,
        )

        from src.transformation.silver_integration import (
            run_silver_integration,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            # La carga simulada no consigue encontrar
            # el archivo obligatorio de disponibilidad.
            with (
                patch(
                    "src.transformation.silver_integration."
                    "load_silver_datasets",
                    side_effect=FileNotFoundError(
                        "No se encontró availability.parquet"
                    ),
                ) as mock_carga,
                patch(
                    "src.transformation.silver_integration."
                    "build_integrated_movies",
                ) as mock_integracion,
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_movies",
                ) as mock_parquet,
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_metadata",
                ) as mock_metadata,
            ):

                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "availability.parquet",
                ):
                    run_silver_integration(
                        run_dirs=carpetas,
                    )

            # Integrated solicita los cinco datasets.
            mock_carga.assert_called_once_with(
                run_dirs=carpetas,
                include_availability=True,
            )

            # Si falta un archivo obligatorio,
            # no se construye ni guarda el catálogo.
            mock_integracion.assert_not_called()
            mock_parquet.assert_not_called()
            mock_metadata.assert_not_called()

            self.assertFalse(
                carpetas["integrated"].exists()
            )

    
    def test_integrated_rechaza_movie_id_duplicados_antes_guardar(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from src.transformation.silver_integration import (
            run_silver_integration,
        )

        peliculas_duplicadas = pd.DataFrame({
            "movie_id": [1, 2, 2],
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        with TemporaryDirectory() as temporal:

            destino = Path(temporal) / "integrated"

            with (
                patch(
                    "src.transformation.silver_integration."
                    "load_silver_datasets",
                    return_value={},
                ),
                patch(
                    "src.transformation.silver_integration."
                    "build_integrated_movies",
                    return_value=peliculas_duplicadas,
                ),
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_movies",
                ) as mock_parquet,
                patch(
                    "src.transformation.silver_integration."
                    "save_integrated_metadata",
                ) as mock_metadata,
            ):

                with self.assertRaisesRegex(
                    ValueError,
                    "movie_id duplicados",
                ):
                    run_silver_integration(
                        output_dir=destino,
                    )

            # No debe guardar resultados inválidos.
            mock_parquet.assert_not_called()
            mock_metadata.assert_not_called()

            self.assertFalse(
                destino.exists()
            )

    
    def test_integrated_rechaza_movie_id_invalidos(self):

        from src.transformation.silver_integration import (
            validate_integrated_catalog,
        )

        casos_invalidos = [
            0,
            -5,
            2.5,
            "ABC",
        ]

        for identificador in casos_invalidos:

            with self.subTest(movie_id=identificador):

                peliculas = pd.DataFrame({
                    "movie_id": [1, identificador],
                    "canonical_title": [
                        "Pelicula A",
                        "Pelicula B",
                    ],
                })

                with self.assertRaisesRegex(
                    ValueError,
                    "movie_id inválidos",
                ):
                    validate_integrated_catalog(
                        peliculas
                    )

        # Confirmar que los identificadores
        # válidos continúan siendo aceptados.

        peliculas_validas = pd.DataFrame({
            "movie_id": [1, 250],
            "canonical_title": [
                "Pelicula A",
                "Pelicula B",
            ],
            "watchmode_checked": pd.Series(
                [True, False],
                dtype="boolean",
            ),
            "has_availability": pd.Series(
                [True, pd.NA],
                dtype="boolean",
            ),
            "sources_count": pd.Series(
                [2, pd.NA],
                dtype="Int64",
            ),
        })


        resultado = validate_integrated_catalog(
            peliculas_validas
        )

        self.assertEqual(
            resultado["movie_id"].tolist(),
            [1, 250],
        )

    
    def test_integrated_valida_coherencia_watchmode(self):

        casos_invalidos = [
            # Declara ausencia, pero registra 3 fuentes.
            (True, False, 3),

            # Declara disponibilidad, pero no hay fuentes.
            (True, True, 0),

            # No consultada, pero registra disponibilidad.
            (False, True, 1),

            # No consultada, pero declara ausencia.
            (False, False, 0),

            # Consultada, pero se desconoce el resultado.
            (True, pd.NA, pd.NA),

            # Conteo negativo.
            (True, True, -1),
        ]

        for checked, availability, count in casos_invalidos:

            with self.subTest(
                checked=checked,
                availability=str(availability),
                count=str(count),
            ):

                df = pd.DataFrame({
                    "movie_id": [1],
                    "watchmode_checked": pd.Series(
                        [checked],
                        dtype="boolean",
                    ),
                    "has_availability": pd.Series(
                        [availability],
                        dtype="boolean",
                    ),
                    "sources_count": pd.Series(
                        [count],
                        dtype="Int64",
                    ),
                })

                with self.assertRaisesRegex(
                    ValueError,
                    "estados Watchmode incoherentes",
                ):
                    validate_integrated_catalog(df)

        # Los tres estados correctos deben aceptarse.
        validos = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "watchmode_checked": pd.Series(
                [True, True, False],
                dtype="boolean",
            ),
            "has_availability": pd.Series(
                [True, False, pd.NA],
                dtype="boolean",
            ),
            "sources_count": pd.Series(
                [2, 0, pd.NA],
                dtype="Int64",
            ),
        })

        resultado = validate_integrated_catalog(validos)

        self.assertEqual(len(resultado), 3)

        self.assertTrue(
            pd.isna(
                resultado.loc[2, "has_availability"]
            )
        )

        self.assertTrue(
            pd.isna(
                resultado.loc[2, "sources_count"]
            )
        )

    
    def test_integrated_lee_fuentes_de_misma_ejecucion(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from src.utils.silver_runs import (
            build_silver_run_directories,
        )

        from src.transformation.silver_integration import (
            load_silver_datasets,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            rutas_esperadas = {
                carpetas["movielens"] / "movies.parquet",
                carpetas["movielens"] / "links.parquet",
                carpetas["tmdb"] / "movies_tmdb.parquet",
                carpetas["watchmode"]
                / "availability_status.parquet",
            }

            # Crear archivos temporales reales para
            # superar la validación de existencia.
            for ruta in rutas_esperadas:

                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pd.DataFrame({
                    "dato_prueba": [1]
                }).to_parquet(
                    ruta,
                    index=False,
                )

            # Simular la lectura de los archivos.
            with patch(
                "src.transformation.silver_integration."
                "pd.read_parquet",
                return_value=pd.DataFrame(),
            ) as mock_lectura:

                datasets = load_silver_datasets(
                    run_dirs=carpetas,
                )

            rutas_leidas = {
                Path(llamada.args[0])
                for llamada in mock_lectura.call_args_list
            }

            self.assertEqual(
                rutas_leidas,
                rutas_esperadas,
            )

            self.assertEqual(
                mock_lectura.call_count,
                4,
            )

            self.assertIsInstance(
                datasets,
                dict,
            )

    
    def test_integracion_conserva_peliculas_sin_tmdb(self):

        # Tres películas originales de MovieLens
        movies = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        # La película 2 no tiene identificador TMDb
        links = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "tmdb_id": pd.Series(
                [862, pd.NA, 550],
                dtype="Int64",
            ),
        })

        # Solo dos películas tienen información TMDb
        tmdb = pd.DataFrame({
            "tmdb_id": pd.Series(
                [862, 550],
                dtype="Int64",
            ),
            "title": [
                "Titulo TMDb A",
                "Titulo TMDb C",
            ],
            "imdb_id": [
                "0111161",
                "0137523",
            ],
            "genres": [
                ["Drama"],
                ["Accion"],
            ],
        })

        # Primera integración: MovieLens + Links
        resultado = integrate_movielens(
            movies,
            links,
        )

        # Segunda integración: agregar TMDb
        resultado = integrate_tmdb(
            resultado,
            tmdb,
        )

        # Ninguna película debe desaparecer
        self.assertEqual(
            len(resultado),
            3,
        )

        self.assertEqual(
            resultado["movie_id"].nunique(),
            3,
        )

        pelicula_2 = resultado.loc[
            resultado["movie_id"] == 2
        ].iloc[0]

        # Conservamos la película de MovieLens
        self.assertEqual(
            pelicula_2["title"],
            "Pelicula B",
        )

        # Pero reconocemos que falta TMDb
        self.assertTrue(
            pd.isna(pelicula_2["tmdb_id"])
        )

        self.assertTrue(
            pd.isna(pelicula_2["tmdb_title"])
        )

        # Las otras películas sí reciben sus datos
        pelicula_1 = resultado.loc[
            resultado["movie_id"] == 1
        ].iloc[0]

        self.assertEqual(
            pelicula_1["tmdb_title"],
            "Titulo TMDb A",
        )

    
    def test_integracion_tmdb_rechaza_id_nulo(self):

        movies = pd.DataFrame({
            "movie_id": [1, 2],
            "tmdb_id": pd.Series(
                [862, pd.NA],
                dtype="Int64",
            ),
            "title": [
                "Pelicula A",
                "Pelicula B",
            ],
        })

        tmdb = pd.DataFrame({
            "tmdb_id": pd.Series(
                [862, pd.NA],
                dtype="Int64",
            ),
            "title": [
                "Titulo TMDb A",
                "Pelicula desconocida",
            ],
            "imdb_id": [
                "0111161",
                None,
            ],
            "genres": [
                ["Drama"],
                [],
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "tmdb_id nulos",
        ):
            integrate_tmdb(
                movies,
                tmdb,
            )

    
    def test_diferenciar_disponibilidad_desconocida(self):

        movies = pd.DataFrame({
            "tmdb_id": [550, 600, 700]
        })

        status = pd.DataFrame({
            "tmdb_id": [550, 600],
            "has_availability": [True, False],
            "sources_count": [2, 0],
        })

        resultado = integrate_watchmode_status(
            movies,
            status,
        )

        pelicula_550 = resultado.loc[
            resultado["tmdb_id"] == 550
        ].iloc[0]

        pelicula_600 = resultado.loc[
            resultado["tmdb_id"] == 600
        ].iloc[0]

        pelicula_700 = resultado.loc[
            resultado["tmdb_id"] == 700
        ].iloc[0]

        # Consultada y con plataformas
        self.assertTrue(
            pelicula_550["has_availability"]
        )

        # Consultada, pero sin plataformas
        self.assertFalse(
            pelicula_600["has_availability"]
        )

        # Nunca consultada: dato desconocido
        self.assertTrue(
            pd.isna(pelicula_700["has_availability"])
        )

        self.assertTrue(
            pd.isna(pelicula_700["sources_count"])
        )


    def test_integracion_rechaza_tmdb_id_duplicados(self):

        movies = pd.DataFrame({
            "tmdb_id": [550, 600]
        })

        status = pd.DataFrame({
            "tmdb_id": [550, 550, 600],
            "has_availability": [
                True, False, True
            ],
            "sources_count": [2, 0, 1],
        })

        with self.assertRaisesRegex(
            ValueError,
            "tmdb_id duplicados",
        ):
            integrate_watchmode_status(
                movies,
                status,
            )

    
    def test_integrated_carga_availability_misma_ejecucion(self):

        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.utils.silver_runs import (
            build_silver_run_directories,
        )

        from src.transformation.silver_integration import (
            load_silver_datasets,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpetas = build_silver_run_directories(
                silver_root=Path(temporal),
                run_id=run_id,
            )

            rutas = {
                "movies": (
                    carpetas["movielens"] / "movies.parquet"
                ),
                "links": (
                    carpetas["movielens"] / "links.parquet"
                ),
                "tmdb": (
                    carpetas["tmdb"] / "movies_tmdb.parquet"
                ),
                "watchmode_status": (
                    carpetas["watchmode"]
                    / "availability_status.parquet"
                ),
                "watchmode_availability": (
                    carpetas["watchmode"]
                    / "availability.parquet"
                ),
            }

            # Crear cinco archivos Parquet pequeños.
            for nombre, ruta in rutas.items():

                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                pd.DataFrame({
                    "origen": [nombre],
                }).to_parquet(
                    ruta,
                    index=False,
                )

            # Ejecutar la carga real.
            datasets = load_silver_datasets(
                run_dirs=carpetas,
                include_availability=True,
            )

            self.assertEqual(
                set(datasets),
                set(rutas),
            )

            # Comprobar que cada dataset procede
            # del archivo esperado.
            for nombre in rutas:

                self.assertEqual(
                    datasets[nombre].iloc[0]["origen"],
                    nombre,
                )

            self.assertEqual(
                len(datasets),
                5,
            )

    def test_integrated_bloquea_conteos_watchmode_incorrectos(self):

        # Watchmode declara 3 plataformas,
        # pero availability contiene solamente 2.
        datasets = {
            "watchmode_availability": pd.DataFrame({
                "tmdb_id": [550, 550],
                "provider_name": [
                    "Plataforma A",
                    "Plataforma B",
                ],
            }),
            "watchmode_status": pd.DataFrame({
                "tmdb_id": [550],
                "sources_count": [3],
                "has_availability": [True],
            }),
        }

        # Si el conteo es incorrecto, no debe
        # comenzar la integración MovieLens.
        with patch(
            "src.transformation.silver_integration."
            "integrate_movielens"
        ) as mock_integracion:

            with self.assertRaisesRegex(
                ValueError,
                "sources_count no coincide",
            ):
                build_integrated_movies(datasets)

        mock_integracion.assert_not_called()

        # Corregir el conteo a 2.
        datasets["watchmode_status"].loc[
            0, "sources_count"
        ] = 2

        # Corregir el conteo: ahora coincide
        # con las dos plataformas registradas.
        datasets["watchmode_status"].loc[
            0, "sources_count"
        ] = 2

        # Añadir datos mínimos de MovieLens para
        # comprobar que se supera la validación.
        datasets["movies"] = pd.DataFrame({
            "movie_id": [1],
            "title": ["Pelicula A"],
            "genres": ["Drama"],
        })

        datasets["links"] = pd.DataFrame({
            "movie_id": [1],
            "tmdb_id": pd.Series(
                [550],
                dtype="Int64",
            ),
            "imdb_id": ["0000550"],
        })

        # Comprobar que, una vez corregido
        # Watchmode, comienza la integración.
        with patch(
            "src.transformation.silver_integration."
            "integrate_movielens",
            side_effect=RuntimeError(
                "Validacion superada"
            ),
        ) as mock_integracion:

            with self.assertRaisesRegex(
                RuntimeError,
                "Validacion superada",
            ):
                build_integrated_movies(datasets)

        mock_integracion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
