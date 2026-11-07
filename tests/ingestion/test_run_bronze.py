
import unittest
from unittest.mock import patch

from src.ingestion.run_bronze import (
    run_bronze_pipeline,
    evaluate_bronze_status,
    execute_bronze_source,
    diagnose_watchmode_sync,
)

class BronzePipelineTests(unittest.TestCase):

    
    def test_pipeline_bloquea_ingesta_no_autorizada(self):

        from src.ingestion.run_bronze import (
            run_bronze_pipeline,
        )

        from unittest.mock import patch

        with (
            patch(
                "src.ingestion.run_bronze."
                "run_movielens_ingestion"
            ) as mock_movielens,
            patch(
                "src.ingestion.run_bronze."
                "run_tmdb_ingestion"
            ) as mock_tmdb,
            patch(
                "src.ingestion.run_bronze."
                "run_watchmode_ingestion"
            ) as mock_watchmode,
            patch(
                "src.ingestion.run_bronze."
                "diagnose_watchmode_sync"
            ) as mock_diagnostico,
            patch(
                "src.ingestion.run_bronze."
                "sync_watchmode_bundles"
            ) as mock_sync,
        ):

            resultado = run_bronze_pipeline()

        self.assertEqual(
            resultado["estado"],
            "skipped"
        )

        self.assertEqual(
            resultado["motivo"],
            "ingestion_not_authorized"
        )

        mock_movielens.assert_not_called()
        mock_tmdb.assert_not_called()
        mock_watchmode.assert_not_called()
        mock_diagnostico.assert_not_called()
        mock_sync.assert_not_called()

    def test_sync_independiente_sin_links(self):

        with (
            patch(
                "src.ingestion.run_bronze.run_movielens_ingestion",
                return_value={"estado": "failed"},
            ),
            patch(
                "src.ingestion.run_bronze.movielens_links_available",
                return_value=False,
            ),
            patch(
                "src.ingestion.run_bronze.run_tmdb_ingestion"
            ) as mock_tmdb,
            patch(
                "src.ingestion.run_bronze.run_watchmode_ingestion"
            ) as mock_watchmode,
            patch(
                "src.ingestion.run_bronze.diagnose_watchmode_sync",
                return_value={
                    "estado": "success",
                    "modo": "dry_run",
                    "pendientes": 2,
                },
            ) as mock_sync,
            patch(
                "src.ingestion.run_bronze.sync_watchmode_bundles"
            ) as mock_subida,
        ):

            resultado = run_bronze_pipeline(
                sync_dry_run=True,
                execute_ingestion=True
            )

        mock_tmdb.assert_not_called()
        mock_watchmode.assert_not_called()

        mock_sync.assert_called_once_with(
            region="PE"
        )

        mock_subida.assert_not_called()

        self.assertEqual(
            resultado["resultados"]["tmdb"]["estado"],
            "skipped"
        )

        self.assertEqual(
            resultado["resultados"]["watchmode"]["estado"],
            "skipped"
        )

        self.assertEqual(
            resultado["resultados"]["watchmode_sync"]["estado"],
            "success"
        )

        self.assertEqual(
            resultado["estado"],
            "partial"
        )

    
    def test_pipeline_usa_diagnostico_por_defecto(self):

        with (
            patch(
                "src.ingestion.run_bronze.run_movielens_ingestion",
                return_value={"estado": "success"},
            ),
            patch(
                "src.ingestion.run_bronze.movielens_links_available",
                return_value=True,
            ),
            patch(
                "src.ingestion.run_bronze.run_tmdb_ingestion",
                return_value={"estado": "success"},
            ),
            patch(
                "src.ingestion.run_bronze.run_watchmode_ingestion",
                return_value={"estado": "skipped"},
            ),
            patch(
                "src.ingestion.run_bronze.diagnose_watchmode_sync",
                return_value={
                    "estado": "success",
                    "modo": "dry_run",
                },
            ) as mock_diagnostico,
            patch(
                "src.ingestion.run_bronze.sync_watchmode_bundles"
            ) as mock_subida,
        ):
            resultado = run_bronze_pipeline(
                execute_ingestion=True
            )

        mock_diagnostico.assert_called_once_with(
            region="PE"
        )

        mock_subida.assert_not_called()

        self.assertEqual(
            resultado["resultados"]["watchmode_sync"]["modo"],
            "dry_run",
        )

        self.assertEqual(
            resultado["estado"],
            "partial",
        )

    
    def test_omitir_fuentes_sin_movielens_links(self):

        with (
            patch(
                "src.ingestion.run_bronze.diagnose_watchmode_sync",
                return_value={
                    "estado": "success",
                    "modo": "dry_run",
                },
            ) as mock_diagnostico,
            patch(
                "src.ingestion.run_bronze.run_movielens_ingestion",
                return_value={"estado": "failed"},
            ),
            patch(
                "src.ingestion.run_bronze.movielens_links_available",
                return_value=False,
            ),
            patch(
                "src.ingestion.run_bronze.run_tmdb_ingestion"
            ) as mock_tmdb,
            patch(
                "src.ingestion.run_bronze.run_watchmode_ingestion"
            ) as mock_watchmode,
        ):

            resultado = run_bronze_pipeline(
                execute_ingestion=True
            )

        self.assertEqual(
            resultado["resultados"]["tmdb"]["estado"],
            "skipped",
        )

        self.assertEqual(
            resultado["resultados"]["watchmode"]["estado"],
            "skipped",
        )

        mock_tmdb.assert_not_called()
        mock_watchmode.assert_not_called()

        self.assertEqual(
            resultado["estado"],
            "partial",
        )

    
    def test_error_independiente_de_fuente(self):

        def fuente_fallida():
            raise RuntimeError(
                "Error simulado de conexión"
            )

        resultado = execute_bronze_source(
            "TMDb",
            fuente_fallida,
        )

        self.assertEqual(
            resultado["estado"],
            "failed",
        )

        self.assertEqual(
            resultado["fuente"],
            "TMDb",
        )

        self.assertIn(
            "Error simulado",
            resultado["error"],
        )

    
    def test_estados_pipeline_bronze(self):

        casos = [
            (
                {
                    "movielens": {"estado": "success"},
                    "tmdb": {"estado": "success"},
                    "watchmode": {"estado": "success"},
                },
                "success",
            ),
            (
                {
                    "movielens": {"estado": "success"},
                    "tmdb": {"estado": "partial"},
                    "watchmode": {"estado": "success"},
                },
                "partial",
            ),
            (
                {
                    "movielens": {"estado": "success"},
                    "tmdb": {"estado": "success"},
                    "watchmode": {"estado": "skipped"},
                },
                "partial",
            ),
            (
                {
                    "movielens": {"estado": "failed"},
                    "tmdb": {"estado": "failed"},
                    "watchmode": {"estado": "failed"},
                },
                "failed",
            ),
        ]

        for resultados, esperado in casos:
            with self.subTest(esperado=esperado):

                estado = evaluate_bronze_status(
                    resultados
                )

                self.assertEqual(
                    estado,
                    esperado,
                )

    
    def test_ejecucion_completa_bronze(self):

        with (
            patch(
                "src.ingestion.run_bronze.diagnose_watchmode_sync",
                return_value={"estado": "success"},
            ) as mock_sync,
            patch(
                "src.ingestion.run_bronze.movielens_links_available",
                return_value=True,
            ),
            patch(
                "src.ingestion.run_bronze.run_movielens_ingestion",
                return_value={"estado": "success"},
            ) as mock_movielens,
            patch(
                "src.ingestion.run_bronze.run_tmdb_ingestion",
                return_value={"estado": "success"},
            ) as mock_tmdb,
            patch(
                "src.ingestion.run_bronze.run_watchmode_ingestion",
                return_value={"estado": "success"},
            ) as mock_watchmode,
        ):

            resultado = run_bronze_pipeline(
                execute_ingestion=True
            )

        self.assertEqual(
            resultado["estado"],
            "success",
        )

        self.assertEqual(
            len(resultado["resultados"]),
            4,
        )

        mock_sync.assert_called_once_with(region="PE")
        mock_movielens.assert_called_once()
        mock_tmdb.assert_called_once()
        mock_watchmode.assert_called_once()


if __name__ == "__main__":
    unittest.main()
