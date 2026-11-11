
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from src.transformation.silver_pipeline import (
    prepare_silver_run,
)


class SilverPipelineTests(unittest.TestCase):

    def test_orquestador_prepara_cuatro_fuentes_mismo_run_id(
        self,
    ):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            silver_root = Path(temporal) / "silver"

            plan = prepare_silver_run(
                silver_root=silver_root,
                run_id=run_id,
            )

            self.assertEqual(
                plan["run_id"],
                run_id,
            )

            self.assertEqual(
                plan["estado"],
                "planned",
            )

            self.assertEqual(
                plan["orden"],
                (
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                ),
            )

            run_root = (
                silver_root
                / "runs"
                / f"run_id={run_id}"
            )

            self.assertEqual(
                plan["directorios"]["root"],
                run_root,
            )

            for fuente in plan["orden"]:

                self.assertEqual(
                    plan["directorios"][fuente],
                    run_root / fuente,
                )

            # La preparación no debe escribir archivos
            # ni crear carpetas.
            self.assertFalse(
                run_root.exists()
            )


    def test_orquestador_ejecuta_cuatro_fuentes_en_orden(self):

        from unittest.mock import Mock

        from src.transformation.silver_pipeline import (
            execute_silver_run,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            llamadas = []

            def simular_transformacion(fuente):

                def ejecutar(**kwargs):
                    llamadas.append((fuente, kwargs))
                    return {"estado": "simulado"}

                return ejecutar

            transformers = {
                fuente: Mock(
                    side_effect=simular_transformacion(fuente)
                )
                for fuente in plan["orden"]
            }

            resultado = execute_silver_run(
                plan=plan,
                transformers=transformers,
            )

            self.assertEqual(
                resultado["estado"],
                "transformed",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertEqual(
                resultado["fuentes_completadas"],
                plan["orden"],
            )

            self.assertEqual(
                [fuente for fuente, _ in llamadas],
                [
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                ],
            )

            for fuente in (
                "movielens",
                "tmdb",
                "watchmode",
            ):

                transformers[fuente].assert_called_once_with(
                    output_dir=plan["directorios"][fuente],
                )

            transformers["integrated"].assert_called_once_with(
                run_dirs=plan["directorios"],
            )

            # No se crean carpetas ni archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )
    
    def test_orquestador_detiene_ejecucion_si_tmdb_falla(self):

        from unittest.mock import Mock

        from src.transformation.silver_pipeline import (
            execute_silver_run,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            plan = prepare_silver_run(
                silver_root=Path(temporal) / "silver",
                run_id=run_id,
            )

            # MovieLens termina correctamente.
            mock_movielens = Mock(
                return_value={"estado": "success"}
            )

            # TMDb simula un error de transformación.
            mock_tmdb = Mock(
                side_effect=RuntimeError(
                    "Error simulado en TMDb"
                )
            )

            # Estas funciones nunca deben ejecutarse.
            mock_watchmode = Mock()
            mock_integrated = Mock()

            transformers = {
                "movielens": mock_movielens,
                "tmdb": mock_tmdb,
                "watchmode": mock_watchmode,
                "integrated": mock_integrated,
            }

            resultado = execute_silver_run(
                plan=plan,
                transformers=transformers,
            )

            # El orquestador reconoce el error.
            self.assertEqual(
                resultado["estado"],
                "failed",
            )

            self.assertEqual(
                resultado["run_id"],
                run_id,
            )

            self.assertEqual(
                resultado["fuente_fallida"],
                "tmdb",
            )

            self.assertIn(
                "Error simulado en TMDb",
                resultado["error"],
            )

            # Solo MovieLens completó su ejecución.
            self.assertEqual(
                resultado["fuentes_completadas"],
                ("movielens",),
            )

            mock_movielens.assert_called_once_with(
                output_dir=plan["directorios"]["movielens"],
            )

            mock_tmdb.assert_called_once_with(
                output_dir=plan["directorios"]["tmdb"],
            )

            # No debe continuar después del fallo.
            mock_watchmode.assert_not_called()
            mock_integrated.assert_not_called()

            # No se generan archivos reales.
            self.assertFalse(
                plan["directorios"]["root"].exists()
            )


if __name__ == "__main__":
    unittest.main()
