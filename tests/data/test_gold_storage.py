
import unittest

from src.load.gold_storage import (
    build_gold_versioned_path,
)


class GoldStorageTests(unittest.TestCase):

    def test_gold_construye_ruta_versionada(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        ruta = build_gold_versioned_path(
            relative_path="dimensions/dim_movie.parquet",
            run_id=run_id,
        )

        self.assertEqual(
            ruta,
            (
                f"run_id={run_id}/"
                "dimensions/dim_movie.parquet"
            ),
        )

        # Aceptar separadores Windows.
        ruta_windows = build_gold_versioned_path(
            relative_path=r"facts\fact_rating\part_000.parquet",
            run_id=run_id,
        )

        self.assertEqual(
            ruta_windows,
            (
                f"run_id={run_id}/"
                "facts/fact_rating/part_000.parquet"
            ),
        )

        # Rechazar rutas que escapan del directorio.
        with self.assertRaises(ValueError):
            build_gold_versioned_path(
                relative_path="../archivo.parquet",
                run_id=run_id,
            )

        # Rechazar identificadores inválidos.
        with self.assertRaises(ValueError):
            build_gold_versioned_path(
                relative_path="dimensions/dim_movie.parquet",
                run_id="identificador-invalido",
            )


if __name__ == "__main__":
    unittest.main()
