
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.silver_runs import (
    build_silver_run_directories,
)


class SilverRunsTests(unittest.TestCase):

    def test_ejecuciones_silver_tienen_carpetas_independientes(self):

        run_id_1 = (
            "12345678-1234-4234-8234-123456789abc"
        )

        run_id_2 = (
            "87654321-4321-4321-8321-cba987654321"
        )

        with TemporaryDirectory() as temporal:

            raiz = Path(temporal)

            primera = build_silver_run_directories(
                silver_root=raiz,
                run_id=run_id_1,
            )

            segunda = build_silver_run_directories(
                silver_root=raiz,
                run_id=run_id_2,
            )

            # Cada ejecución tiene rutas distintas.
            self.assertNotEqual(
                primera["root"],
                segunda["root"],
            )

            # Cada fuente tiene su propia carpeta.
            self.assertEqual(
                primera["movielens"],
                raiz / "runs"
                / f"run_id={run_id_1}"
                / "movielens",
            )

            self.assertEqual(
                primera["integrated"],
                raiz / "runs"
                / f"run_id={run_id_1}"
                / "integrated",
            )

            self.assertEqual(
                set(primera.keys()),
                {
                    "root",
                    "movielens",
                    "tmdb",
                    "watchmode",
                    "integrated",
                },
            )

            # No se crea nada en el disco.
            self.assertFalse(
                primera["root"].exists()
            )

            self.assertFalse(
                segunda["root"].exists()
            )


if __name__ == "__main__":
    unittest.main()

