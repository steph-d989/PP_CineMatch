import unittest
from unittest.mock import patch, MagicMock

from src.utils.supabase_client import (
    storage_file_exists,
    list_storage_filenames,
    build_tmdb_movie_path,
    list_tmdb_legacy_files
)


class SupabaseClientTests(unittest.TestCase):

    def test_archivo_existente(self):

        cliente = MagicMock()

        storage = cliente.storage.from_.return_value

        storage.list.return_value = [
            {"name": "movie_550.json"},
            {"name": "movie_100.json"},
        ]

        with patch(
            "src.utils.supabase_client.get_supabase_client",
            return_value=cliente
        ):
            resultado = storage_file_exists(
                bucket="Bronze",
                remote_path=(
                    "tmdb/ingestion_date=2026-11-04/"
                    "movie_550.json"
                )
            )

        self.assertTrue(resultado)

        cliente.storage.from_.assert_called_once_with(
            "Bronze"
        )

        storage.list.assert_called_once_with(
            path="tmdb/ingestion_date=2026-11-04",
            options={
                "limit": 100,
                "offset": 0,
                "sortBy": {
                    "column": "name",
                    "order": "asc"
                }
            }
        )

    def test_archivo_no_existente(self):

        cliente = MagicMock()

        storage = cliente.storage.from_.return_value

        storage.list.return_value = [
            {"name": "movie_100.json"},
        ]

        with patch(
            "src.utils.supabase_client.get_supabase_client",
            return_value=cliente
        ):
            resultado = storage_file_exists(
                bucket="Bronze",
                remote_path=(
                    "tmdb/ingestion_date=2026-11-04/"
                    "movie_550.json"
                )
            )

        self.assertFalse(resultado)

    def test_carpeta_vacia(self):

        cliente = MagicMock()

        storage = cliente.storage.from_.return_value

        storage.list.return_value = []

        with patch(
            "src.utils.supabase_client.get_supabase_client",
            return_value=cliente
        ):
            resultado = storage_file_exists(
                bucket="Bronze",
                remote_path=(
                    "tmdb/ingestion_date=2026-11-04/"
                    "movie_550.json"
                )
            )

        self.assertFalse(resultado)

    
    
    def test_archivo_en_segunda_pagina(self):

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        primera_pagina = [
            {"name": f"movie_{i}.json"}
            for i in range(100)
        ]

        segunda_pagina = [
            {"name": "movie_550.json"}
        ]

        storage.list.side_effect = [
            primera_pagina,
            segunda_pagina,
        ]

        with patch(
            "src.utils.supabase_client.get_supabase_client",
            return_value=cliente
        ):
            resultado = storage_file_exists(
                bucket="Bronze",
                remote_path=(
                    "tmdb/ingestion_date=2026-11-04/"
                    "movie_550.json"
                )
            )

        self.assertTrue(resultado)
        self.assertEqual(storage.list.call_count, 2)
        self.assertEqual(
            storage.list.call_args_list[1].kwargs[
                "options"
            ]["offset"],
            100
        )

    
    def test_listar_archivos_por_lotes(self):

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.list.side_effect = [
            [
                {"id": "1", "name": "movie_100.json"},
                {"id": "2", "name": "movie_200.json"},
            ],
            [
                {"id": "3", "name": "movie_550.json"},
            ],
        ]

        with patch(
            "src.utils.supabase_client.get_supabase_client",
            return_value=cliente
        ):
            nombres = list_storage_filenames(
                bucket="Bronze",
                folder="tmdb/ingestion_date=2026-11-04",
                page_size=2,
            )

        self.assertEqual(
            nombres,
            {
                "movie_100.json",
                "movie_200.json",
                "movie_550.json",
            }
        )

        self.assertEqual(storage.list.call_count, 2)

        self.assertEqual(
            storage.list.call_args_list[1].kwargs[
                "options"
            ]["offset"],
            2
        )
    
    def test_ruta_estable_tmdb(self):

        ruta = build_tmdb_movie_path(550)

        self.assertEqual(
            ruta,
            "tmdb/movies/movie_550.json"
        )

        with self.assertRaises(ValueError):
            build_tmdb_movie_path(-1)

        with self.assertRaises(TypeError):
            build_tmdb_movie_path("550")

        with self.assertRaises(TypeError):
            build_tmdb_movie_path(True)

    
    def test_inventario_tmdb_legacy(self):

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.list.return_value = [
            {"name": "ingestion_date=2026-11-04"},
            {"name": "movies"},
        ]

        with (
            patch(
                "src.utils.supabase_client.get_supabase_client",
                return_value=cliente
            ),
            patch(
                "src.utils.supabase_client.list_storage_filenames",
                return_value={
                    "movie_550.json",
                    "_metadata.json",
                }
            ),
        ):

            inventario = list_tmdb_legacy_files()

        self.assertEqual(len(inventario), 1)

        self.assertEqual(
            inventario[0]["ruta_antigua"],
            (
                "tmdb/ingestion_date=2026-11-04/"
                "movie_550.json"
            )
        )


if __name__ == "__main__":
    unittest.main()
