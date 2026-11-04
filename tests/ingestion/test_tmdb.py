import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from src.ingestion.tmdb import (
    extract_tmdb_movie,
    extract_tmdb_movies,
    upload_tmdb_to_supabase,
)

from src.utils.config import TMDB_BASE_URL

class TMDbTests(unittest.TestCase):

    def test_consulta_pelicula_correcta(self):

        tmdb_id = 550
        token = "token_simulado"

        respuesta = MagicMock()
        respuesta.json.return_value = {
            "id": 550,
            "title": "Fight Club"
        }

        with patch(
            "src.ingestion.tmdb.requests.get",
            return_value=respuesta
        ) as mock_get:

            resultado = extract_tmdb_movie(
                tmdb_id=tmdb_id,
                token=token
            )

        mock_get.assert_called_once()

        argumentos = mock_get.call_args

        self.assertEqual(
            argumentos.args[0],
            f"{TMDB_BASE_URL}/movie/{tmdb_id}"
        )

        self.assertEqual(
            argumentos.kwargs["headers"]["Authorization"],
            f"Bearer {token}"
        )

        self.assertEqual(resultado["id"], tmdb_id)


    def test_registra_solo_archivos_nuevos(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            archivo_existente = (
                directorio / "movie_100.json"
            )
            archivo_existente.write_text(
                '{"id": 100}',
                encoding="utf-8"
            )

    
    def test_registra_solo_archivos_nuevos(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            archivo_existente = (
                directorio / "movie_100.json"
            )
            archivo_existente.write_text(
                '{"id": 100}',
                encoding="utf-8"
            )

            def guardar_simulado(tmdb_id, data):
                ruta = directorio / f"movie_{tmdb_id}.json"
                ruta.write_text(
                    '{"id": 200}',
                    encoding="utf-8"
                )
                return ruta

            with (
                patch(
                    "src.ingestion.tmdb.BRONZE_TMDB_DIR",
                    directorio
                ),
                patch(
                    "src.ingestion.tmdb.extract_tmdb_movie",
                    return_value={"id": 200}
                ) as mock_extract,
                patch(
                    "src.ingestion.tmdb.save_tmdb_raw",
                    side_effect=guardar_simulado
                ),
                patch(
                    "src.ingestion.tmdb.time.sleep"
                ),
            ):

                resultado = extract_tmdb_movies(
                    tmdb_ids=[100, 200],
                    token="token_simulado"
                )

            self.assertEqual(
                resultado["existentes"], 1
            )

            self.assertEqual(
                resultado["exitosos"], 1
            )

            self.assertEqual(
                len(resultado["archivos_nuevos"]), 1
            )

            self.assertEqual(
                resultado["archivos_nuevos"][0].name,
                "movie_200.json"
            )

            mock_extract.assert_called_once_with(
                200,
                "token_simulado"
            )

    
    def test_sincronizacion_supabase(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            archivo = directorio / "movie_550.json"
            archivo.write_text(
                '{"id": 550}',
                encoding="utf-8"
            )

            metadata = directorio / "_metadata.json"
            metadata.write_text(
                '{"fuente": "TMDb"}',
                encoding="utf-8"
            )

            with (
                patch(
                    "src.ingestion.tmdb.list_storage_filenames",
                    return_value={"movie_550.json"}
                ) as mock_exists,
                patch(
                    "src.ingestion.tmdb.upload_file_to_storage"
                ) as mock_upload,
            ):

                resultado = upload_tmdb_to_supabase(
                    archivos=[archivo],
                    ruta_metadata=metadata,
                )

            self.assertEqual(
                resultado["archivos_existentes"], 1
            )

            self.assertEqual(
                resultado["archivos_subidos"], 0
            )

            self.assertTrue(
                resultado["metadata_subida"]
            )

            mock_exists.assert_called_once()

            # Solo se actualizan los metadatos
            mock_upload.assert_called_once()

            self.assertEqual(
                mock_upload.call_args.kwargs["local_path"],
                metadata
            )

    
    def test_subida_archivo_pendiente(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            archivo = directorio / "movie_550.json"
            archivo.write_text(
                '{"id": 550}',
                encoding="utf-8"
            )

            metadata = directorio / "_metadata.json"
            metadata.write_text(
                '{"fuente": "TMDb"}',
                encoding="utf-8"
            )

            with (
                patch(
                    "src.ingestion.tmdb.list_storage_filenames",
                    return_value=set()
                ) as mock_exists,
                patch(
                    "src.ingestion.tmdb.upload_file_to_storage"
                ) as mock_upload,
            ):

                resultado = upload_tmdb_to_supabase(
                    archivos=[archivo],
                    ruta_metadata=metadata,
                )

            self.assertEqual(
                resultado["archivos_subidos"], 1
            )

            self.assertEqual(
                resultado["archivos_existentes"], 0
            )

            self.assertTrue(
                resultado["metadata_subida"]
            )

            self.assertEqual(
                resultado["errores"], []
            )

            mock_exists.assert_called_once()

            # Una subida de película y otra de metadatos
            self.assertEqual(
                mock_upload.call_count, 2
            )

            primera_subida = mock_upload.call_args_list[0]

            self.assertEqual(
                primera_subida.kwargs["local_path"],
                archivo
            )

            self.assertFalse(
                primera_subida.kwargs["upsert"]
            )
    
    def test_sincronizacion_por_lotes(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            archivos = []

            for tmdb_id in [100, 200, 300]:
                archivo = (
                    directorio / f"movie_{tmdb_id}.json"
                )
                archivo.write_text(
                    f'{{"id": {tmdb_id}}}',
                    encoding="utf-8"
                )
                archivos.append(archivo)

            metadata = directorio / "_metadata.json"
            metadata.write_text(
                '{"fuente": "TMDb"}',
                encoding="utf-8"
            )

            with (
                patch(
                    "src.ingestion.tmdb.list_storage_filenames",
                    return_value={"movie_100.json"}
                ) as mock_list,
                patch(
                    "src.ingestion.tmdb.upload_file_to_storage"
                ) as mock_upload,
            ):

                resultado = upload_tmdb_to_supabase(
                    archivos=archivos,
                    ruta_metadata=metadata,
                )

            self.assertEqual(
                resultado["archivos_existentes"], 1
            )

            self.assertEqual(
                resultado["archivos_subidos"], 2
            )

            self.assertTrue(
                resultado["metadata_subida"]
            )

            self.assertEqual(
                resultado["errores"], []
            )

            # Listado remoto: una sola llamada
            mock_list.assert_called_once()
            
            self.assertEqual(
                mock_list.call_args.kwargs["folder"],
                "tmdb/movies"
            )

            rutas_subidas = [
                llamada.kwargs["remote_path"]
                for llamada in mock_upload.call_args_list
                if llamada.kwargs["local_path"].name.startswith("movie_")
            ]

            self.assertEqual(
                set(rutas_subidas),
                {
                    "tmdb/movies/movie_200.json",
                    "tmdb/movies/movie_300.json",
                }
            )

            # Dos películas y los metadatos
            self.assertEqual(
                mock_upload.call_count, 3
            )


if __name__ == "__main__":
    unittest.main()
