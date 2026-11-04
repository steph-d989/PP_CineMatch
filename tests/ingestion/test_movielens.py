import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import io
import requests
from unittest.mock import MagicMock

from src.ingestion.movielens import (
    download_movielens,
    extract_movielens,
    validate_movielens,
    upload_movielens_to_supabase,
    run_movielens_ingestion,
)


ARCHIVOS_ESPERADOS = [
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
    "genome-scores.csv",
    "genome-tags.csv",
]


class MovieLensExtractionTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.base = Path(self.temp.name)
        self.dataset = self.base / "ml-32m"
        self.zip_path = self.base / "ml-32m.zip"

        parche = patch(
            "src.ingestion.movielens.BRONZE_MOVIELENS_DIR",
            self.base
        )
        parche.start()
        self.addCleanup(parche.stop)

    def crear_zip(self, archivos=None):
        if archivos is None:
            archivos = ARCHIVOS_ESPERADOS

        with zipfile.ZipFile(self.zip_path, "w") as z:
            for nombre in archivos:
                z.writestr(
                    f"ml-32m/{nombre}",
                    "columna1,columna2\n1,2\n"
                )

    def test_extraccion_correcta(self):
        self.crear_zip()

        resultado = extract_movielens(self.zip_path)

        self.assertEqual(resultado, self.dataset)

        for nombre in ARCHIVOS_ESPERADOS:
            self.assertTrue(
                (self.dataset / nombre).is_file()
            )

    def test_no_reextraer_archivos_existentes(self):
        self.dataset.mkdir()

        for nombre in ARCHIVOS_ESPERADOS:
            (self.dataset / nombre).write_text(
                "archivo existente",
                encoding="utf-8"
            )

        resultado = extract_movielens(
            self.zip_path,
            force_extract=False
        )

        self.assertEqual(resultado, self.dataset)

    def test_zip_invalido(self):
        self.zip_path.write_text(
            "No soy un ZIP",
            encoding="utf-8"
        )

        with self.assertRaises(ValueError):
            extract_movielens(self.zip_path)

    def test_archivos_faltantes(self):
        self.crear_zip(["movies.csv"])

        with self.assertRaises(FileNotFoundError):
            extract_movielens(self.zip_path)

    
    def test_validacion_csv_correctos(self):
        self.dataset.mkdir(parents=True, exist_ok=True)

        contenidos = {
            "movies.csv": "movieId,title,genres\n1,Toy Story,Comedy\n",
            "ratings.csv": "userId,movieId,rating,timestamp\n1,1,4.0,1000\n",
            "tags.csv": "userId,movieId,tag,timestamp\n1,1,fun,1000\n",
            "links.csv": "movieId,imdbId,tmdbId\n1,114709,862\n",
            "genome-scores.csv": "movieId,tagId,relevance\n1,1,0.8\n",
            "genome-tags.csv": "tagId,tag\n1,action\n",
        }

        for nombre, contenido in contenidos.items():
            (self.dataset / nombre).write_text(
                contenido, encoding="utf-8"
            )

        resultado = validate_movielens(self.dataset)

        for nombre in ARCHIVOS_ESPERADOS:
            self.assertTrue(resultado[nombre]["existe"])
            self.assertTrue(resultado[nombre]["csv_valido"])
            self.assertEqual(resultado[nombre]["filas"], 1)

    def test_validacion_archivos_faltantes(self):
        self.dataset.mkdir(parents=True, exist_ok=True)

        resultado = validate_movielens(self.dataset)

        for nombre in ARCHIVOS_ESPERADOS:
            self.assertFalse(resultado[nombre]["existe"])
            self.assertFalse(resultado[nombre]["csv_valido"])


    def test_descarga_correcta(self):
        # Creamos un ZIP simulado en memoria
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("ml-32m/movies.csv", "movieId,title\n1,Test\n")

        respuesta = MagicMock()
        respuesta.__enter__.return_value = respuesta
        respuesta.iter_content.return_value = [buffer.getvalue()]

        with patch(
            "src.ingestion.movielens.requests.get",
            return_value=respuesta
        ) as mock_get:

            resultado = download_movielens()

        self.assertTrue(resultado.exists())
        self.assertTrue(zipfile.is_zipfile(resultado))
        self.assertFalse(
            (self.base / "ml-32m.zip.part").exists()
        )
        mock_get.assert_called_once()

    def test_no_descargar_zip_existente(self):
        self.crear_zip()

        with patch(
            "src.ingestion.movielens.requests.get"
        ) as mock_get:

            resultado = download_movielens()

        self.assertEqual(resultado, self.zip_path)
        mock_get.assert_not_called()

    def test_descarga_interrumpida(self):
        with patch(
            "src.ingestion.movielens.requests.get",
            side_effect=requests.ConnectionError(
                "Conexión interrumpida"
            )
        ):

            with self.assertRaises(requests.ConnectionError):
                download_movielens()

        self.assertFalse(self.zip_path.exists())
        self.assertFalse(
            (self.base / "ml-32m.zip.part").exists()
        )
        
    def test_subida_metadata_supabase(self):

        metadata = self.base / "_metadata.json"

        metadata.write_text(
            '{"fuente": "MovieLens"}',
            encoding="utf-8"
        )

        with patch(
            "src.ingestion.movielens.upload_file_to_storage"
        ) as mock_upload:

            resultado = upload_movielens_to_supabase(
                metadata
            )

        mock_upload.assert_called_once()

        argumentos = mock_upload.call_args.kwargs

        self.assertEqual(
            argumentos["bucket"],
            "Bronze"
        )

        self.assertTrue(
            argumentos["upsert"]
        )

        self.assertEqual(
            argumentos["local_path"],
            metadata
        )

        self.assertEqual(
            resultado["remote_path"],
            argumentos["remote_path"]
        )
    
    def test_pipeline_detiene_validacion_fallida(self):

        validacion_invalida = {
            "ratings.csv": {
                "existe": False,
                "csv_valido": False,
            }
        }

        with (
            patch(
                "src.ingestion.movielens.download_movielens",
                return_value=self.zip_path
            ),
            patch(
                "src.ingestion.movielens.extract_movielens",
                return_value=self.dataset
            ),
            patch(
                "src.ingestion.movielens.validate_movielens",
                return_value=validacion_invalida
            ),
            patch(
                "src.ingestion.movielens.upload_movielens_to_supabase"
            ) as mock_upload,
        ):

            with self.assertRaises(RuntimeError):
                run_movielens_ingestion()

            mock_upload.assert_not_called()



if __name__ == "__main__":
    unittest.main()
