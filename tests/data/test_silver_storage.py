
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from src.load.silver_storage import (
    upload_silver_directory,
    upload_silver_to_supabase,
    compare_silver_storage_file,
)


class SilverStorageTests(unittest.TestCase):

    
    def test_comparacion_silver_archivos_identicos(self):

        contenido = b"contenido_parquet_simulado"

        with TemporaryDirectory() as temporal:

            archivo = (
                Path(temporal) / "movies.parquet"
            )

            archivo.write_bytes(contenido)

            cliente = MagicMock()
            storage = cliente.storage.from_.return_value

            # Supabase devuelve exactamente
            # los mismos bytes que el archivo local.
            storage.download.return_value = contenido

            with (
                patch(
                    "src.load.silver_storage."
                    "get_supabase_client",
                    return_value=cliente,
                ),
                patch(
                    "src.load.silver_storage."
                    "upload_file_to_storage"
                ) as mock_upload,
            ):

                resultado = compare_silver_storage_file(
                    local_path=archivo,
                    remote_path=(
                        "movielens/"
                        "ingestion_date=2026-11-09/"
                        "movies.parquet"
                    ),
                )

        self.assertTrue(
            resultado["iguales"]
        )

        self.assertEqual(
            resultado["estado"],
            "identical"
        )

        self.assertEqual(
            resultado["hash_local"],
            resultado["hash_remoto"],
        )

        storage.download.assert_called_once_with(
            "movielens/"
            "ingestion_date=2026-11-09/"
            "movies.parquet"
        )

        mock_upload.assert_not_called()

    
    def test_comparacion_silver_detecta_contenido_diferente(self):

        contenido_local = b"catalogo_actualizado"
        contenido_remoto = b"catalogo_anterior"

        with TemporaryDirectory() as temporal:

            archivo = Path(temporal) / "movies.parquet"
            archivo.write_bytes(contenido_local)

            cliente = MagicMock()
            storage = cliente.storage.from_.return_value

            # Simular una versión remota diferente.
            storage.download.return_value = (
                contenido_remoto
            )

            with (
                patch(
                    "src.load.silver_storage."
                    "get_supabase_client",
                    return_value=cliente,
                ),
                patch(
                    "src.load.silver_storage."
                    "upload_file_to_storage"
                ) as mock_upload,
            ):

                resultado = compare_silver_storage_file(
                    local_path=archivo,
                    remote_path=(
                        "movielens/"
                        "ingestion_date=2026-11-09/"
                        "movies.parquet"
                    ),
                )

        self.assertFalse(
            resultado["iguales"]
        )

        self.assertEqual(
            resultado["estado"],
            "conflict"
        )

        self.assertNotEqual(
            resultado["hash_local"],
            resultado["hash_remoto"],
        )

        storage.download.assert_called_once()

        mock_upload.assert_not_called()

    
    def test_publicacion_silver_detecta_fallo_parcial(self):

        from src.load.silver_storage import (
            evaluate_silver_publication,
        )

        resultados = {
            "movielens": {
                "subidos": 4,
                "existentes": 0,
                "errores": [],
            },
            "tmdb": {
                "subidos": 1,
                "existentes": 0,
                "errores": [],
            },
            "watchmode": {
                "subidos": 0,
                "existentes": 0,
                "errores": [
                    {
                        "archivo": "availability.parquet",
                        "error": "Error simulado",
                    }
                ],
            },
            "integrated": {
                "subidos": 1,
                "existentes": 0,
                "errores": [],
            },
        }

        evaluacion = evaluate_silver_publication(
            resultados
        )

        self.assertEqual(
            evaluacion["estado"],
            "partial",
        )

        self.assertEqual(
            evaluacion["errores"],
            1,
        )

        self.assertEqual(
            evaluacion["detalle"]["watchmode"],
            "failed",
        )

        self.assertEqual(
            evaluacion["archivos_procesados"],
            6,
        )

        self.assertEqual(
            evaluacion["fuentes_faltantes"],
            [],
        )

    
    def test_subida_silver_utiliza_run_id(self):

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            (carpeta / "movies.parquet").write_bytes(
                b"peliculas_simuladas"
            )

            ratings_dir = carpeta / "ratings"
            ratings_dir.mkdir()

            (ratings_dir / "part_000.parquet").write_bytes(
                b"ratings_simulados"
            )

            with patch(
                "src.load.silver_storage."
                "upload_file_to_storage"
            ) as mock_upload:

                resultado = upload_silver_directory(
                    local_dir=carpeta,
                    source="movielens",
                    run_id=run_id,
                )

        # Se intentaron subir dos archivos.
        self.assertEqual(
            mock_upload.call_count,
            2,
        )

        rutas = {
            llamada.kwargs["remote_path"]
            for llamada in mock_upload.call_args_list
        }

        self.assertEqual(
            rutas,
            {
                (
                    f"movielens/run_id={run_id}/"
                    "movies.parquet"
                ),
                (
                    f"movielens/run_id={run_id}/"
                    "ratings/part_000.parquet"
                ),
            },
        )

        # No se permiten sobrescrituras.
        for llamada in mock_upload.call_args_list:
            self.assertFalse(
                llamada.kwargs["upsert"]
            )

        self.assertEqual(
            resultado["run_id"],
            run_id,
        )

        self.assertEqual(
            resultado["subidos"],
            2,
        )

        self.assertEqual(
            resultado["errores"],
            [],
        )

    
    def test_publicacion_silver_comparte_run_id(self):


        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        def subida_simulada(
            local_dir,
            source,
            run_id,
        ):
            return {
                "source": source,
                "run_id": run_id,
                "subidos": 1,
                "existentes": 0,
                "errores": [],
            }

        with patch(
            "src.load.silver_storage."
            "upload_silver_directory",
            side_effect=subida_simulada,
        ) as mock_subida:

            resultado = upload_silver_to_supabase(
                run_id=run_id
            )
            patch(
                "src.load.silver_storage.build_silver_manifest",
                return_value={
                    "estado": "complete",
                },
            ),
            patch(
                "src.load.silver_storage.validate_silver_manifests",
                return_value={
                    "estado": "ready",
                    "autorizado": True,
                },
            ),


        # Las cuatro fuentes deben procesarse.
        self.assertEqual(
            mock_subida.call_count,
            4,
        )

        fuentes = {
            "movielens",
            "tmdb",
            "watchmode",
            "integrated",
        }


        self.assertEqual(
            set(resultado["fuentes"].keys()),
            fuentes,
        )



        # Las cuatro fuentes deben estar presentes.
        fuentes = {
            "movielens",
            "tmdb",
            "watchmode",
            "integrated",
        }

        self.assertEqual(
            set(resultado["fuentes"].keys()),
            fuentes,
        )

        # Todas las respuestas conservan el run_id.
        for fuente in fuentes:

            self.assertEqual(
                resultado["fuentes"][fuente]["run_id"],
                run_id,
            )

            self.assertEqual(
                resultado["fuentes"][fuente]["subidos"],
                1,
            )

            self.assertEqual(
                resultado["fuentes"][fuente]["errores"],
                [],
            )

    
    def test_verificacion_remota_detecta_hash_incorrecto(self):

        import hashlib
        from unittest.mock import MagicMock

        from src.load.silver_storage import (
            verify_silver_remote_manifest,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        contenido_local = b"peliculas_correctas"
        contenido_remoto = b"peliculas_diferentes"

        hash_esperado = hashlib.sha256(
            contenido_local
        ).hexdigest()

        ruta_remota = (
            f"movielens/run_id={run_id}/movies.parquet"
        )

        manifest = {
            "source": "movielens",
            "run_id": run_id,
            "estado": "complete",
            "archivos": [
                {
                    "archivo": "movies.parquet",
                    "remote_path": ruta_remota,
                    "sha256": hash_esperado,
                    "size_bytes": len(contenido_local),
                }
            ],
        }

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.download.return_value = contenido_remoto

        with (
            patch(
                "src.load.silver_storage.get_supabase_client",
                return_value=cliente,
            ),
            patch(
                "src.load.silver_storage.upload_file_to_storage"
            ) as mock_upload,
        ):
            resultado = verify_silver_remote_manifest(
                manifest
            )

        self.assertEqual(resultado["estado"], "failed")

        self.assertEqual(resultado["verificados"], [])

        self.assertEqual(len(resultado["conflictos"]), 1)

        self.assertEqual(
            resultado["conflictos"][0]["motivo"],
            "sha256_mismatch",
        )

        self.assertEqual(resultado["errores"], [])

        storage.download.assert_called_once_with(
            ruta_remota
        )

        mock_upload.assert_not_called()

    
    def test_verificacion_remota_silver_completa(self):

        import hashlib
        from unittest.mock import MagicMock

        from src.load.silver_storage import (
            verify_silver_remote_manifest,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        archivos = {
            "movies_tmdb.parquet": b"peliculas_tmdb",
            "_metadata.json": b'{"estado":"success"}',
        }

        registros = []

        for nombre, contenido in archivos.items():

            registros.append({
                "archivo": nombre,
                "remote_path": (
                    f"tmdb/run_id={run_id}/{nombre}"
                ),
                "sha256": hashlib.sha256(
                    contenido
                ).hexdigest(),
                "size_bytes": len(contenido),
            })

        manifest = {
            "source": "tmdb",
            "run_id": run_id,
            "estado": "complete",
            "archivos_esperados": 2,
            "archivos": registros,
            "faltantes": [],
        }

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        # Simular las descargas de Supabase.
        storage.download.side_effect = [
            archivos["movies_tmdb.parquet"],
            archivos["_metadata.json"],
        ]

        with (
            patch(
                "src.load.silver_storage.get_supabase_client",
                return_value=cliente,
            ),
            patch(
                "src.load.silver_storage.upload_file_to_storage"
            ) as mock_upload,
        ):

            resultado = verify_silver_remote_manifest(
                manifest
            )

        self.assertEqual(
            resultado["estado"],
            "verified",
        )

        self.assertEqual(
            len(resultado["verificados"]),
            2,
        )

        self.assertEqual(
            resultado["conflictos"],
            [],
        )

        self.assertEqual(
            resultado["errores"],
            [],
        )

        self.assertEqual(
            storage.download.call_count,
            2,
        )

        mock_upload.assert_not_called()

    
    def test_silver_registra_error_general_de_subida(self):

        from src.load.silver_storage import (
            upload_silver_directory,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            archivo = carpeta / "movies.parquet"
            archivo.write_bytes(b"datos_simulados")

            with (
                patch(
                    "src.load.silver_storage."
                    "upload_file_to_storage",
                    side_effect=ConnectionError(
                        "Conexion simulada interrumpida"
                    ),
                ) as mock_upload,
                patch(
                    "src.load.silver_storage."
                    "compare_silver_storage_file"
                ) as mock_comparacion,
            ):

                resultado = upload_silver_directory(
                    local_dir=carpeta,
                    source="movielens",
                    run_id=run_id,
                )

        mock_upload.assert_called_once()
        mock_comparacion.assert_not_called()

        self.assertEqual(
            resultado["subidos"],
            0,
        )

        self.assertEqual(
            resultado["existentes"],
            0,
        )

        self.assertEqual(
            len(resultado["errores"]),
            1,
        )

        self.assertEqual(
            resultado["errores"][0]["error"],
            "storage_upload_failed",
        )

        self.assertIn(
            "Conexion simulada interrumpida",
            resultado["errores"][0]["detalle"],
        )

        self.assertFalse(
            mock_upload.call_args.kwargs["upsert"]
        )

    
    def test_manifiesto_detecta_archivo_local_modificado(self):

        from src.load.silver_storage import (
            build_silver_manifest,
            verify_silver_local_manifest,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            archivos = {
                "movies_tmdb.parquet": b"version_original",
                "_metadata.json": b'{"estado":"success"}',
            }

            for nombre, contenido in archivos.items():
                (carpeta / nombre).write_bytes(contenido)

            # Crear el manifiesto de la versión original.
            manifiesto = build_silver_manifest(
                local_dir=carpeta,
                source="tmdb",
                run_id=run_id,
            )

            self.assertEqual(
                manifiesto["estado"],
                "complete",
            )

            # Simular una modificación posterior.
            (carpeta / "movies_tmdb.parquet").write_bytes(
                b"version_modificada"
            )

            with patch(
                "src.load.silver_storage."
                "upload_file_to_storage"
            ) as mock_upload:

                resultado = verify_silver_local_manifest(
                    manifest=manifiesto,
                    local_dir=carpeta,
                )

        self.assertEqual(
            resultado["estado"],
            "failed",
        )

        self.assertEqual(
            len(resultado["conflictos"]),
            1,
        )

        self.assertEqual(
            resultado["conflictos"][0]["archivo"],
            "movies_tmdb.parquet",
        )

        self.assertEqual(
            resultado["conflictos"][0]["motivo"],
            "local_file_changed",
        )

        self.assertEqual(
            resultado["verificados"],
            ["_metadata.json"],
        )

        mock_upload.assert_not_called()

    
    def test_manifiesto_silver_detecta_archivos_faltantes(self):

        from src.load.silver_storage import (
            build_silver_manifest,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            # Crear solamente dos archivos.
            (carpeta / "movies.parquet").write_bytes(
                b"peliculas_simuladas"
            )

            (carpeta / "links.parquet").write_bytes(
                b"identificadores_simulados"
            )

            manifiesto = build_silver_manifest(
                local_dir=carpeta,
                source="movielens",
                run_id=run_id,
            )

        # La publicación no está completa.
        self.assertEqual(
            manifiesto["estado"],
            "incomplete",
        )

        self.assertEqual(
            manifiesto["run_id"],
            run_id,
        )

        self.assertEqual(
            manifiesto["archivos_esperados"],
            2,
        )

        # Identificar los archivos faltantes.
        self.assertEqual(
            set(manifiesto["faltantes"]),
            {
                "tags.parquet",
                "_metadata.json",
                "ratings/*.parquet",
            },
        )

        # Todos los archivos encontrados
        # deben tener SHA-256 y ruta versionada.
        for archivo in manifiesto["archivos"]:

            self.assertEqual(
                len(archivo["sha256"]),
                64,
            )

            self.assertTrue(
                archivo["remote_path"].startswith(
                    f"movielens/run_id={run_id}/"
                )
            )

            self.assertGreater(
                archivo["size_bytes"],
                0,
            )

    
    def test_manifiesto_silver_completo(self):

        from src.load.silver_storage import (
            build_silver_manifest,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            # Crear los archivos obligatorios.
            archivos = {
                "movies.parquet": b"peliculas",
                "links.parquet": b"identificadores",
                "tags.parquet": b"etiquetas",
                "_metadata.json": b'{"estado":"success"}',
                "ratings/part_000.parquet": b"ratings_1",
                "ratings/part_001.parquet": b"ratings_2",
            }

            for nombre, contenido in archivos.items():

                ruta = carpeta / nombre

                ruta.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                ruta.write_bytes(contenido)

            manifiesto = build_silver_manifest(
                local_dir=carpeta,
                source="movielens",
                run_id=run_id,
            )

            self.assertEqual(
                manifiesto["estado"],
                "complete",
            )

            self.assertEqual(
                manifiesto["faltantes"],
                [],
            )

            self.assertEqual(
                manifiesto["archivos_esperados"],
                6,
            )

            nombres = {
                item["archivo"]
                for item in manifiesto["archivos"]
            }

            self.assertEqual(
                nombres,
                set(archivos.keys()),
            )

            # Verificar los hashes calculados.
            import hashlib

            for item in manifiesto["archivos"]:

                contenido_original = archivos[
                    item["archivo"]
                ]

                hash_esperado = hashlib.sha256(
                    contenido_original
                ).hexdigest()

                self.assertEqual(
                    item["sha256"],
                    hash_esperado,
                )

                self.assertEqual(
                    item["size_bytes"],
                    len(contenido_original),
                )

                self.assertTrue(
                    item["remote_path"].startswith(
                        f"movielens/run_id={run_id}/"
                    )
                )

    
    def test_publicacion_bloqueada_por_manifest_incompleto(self):

        from src.load.silver_storage import (
            validate_silver_manifests,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        def manifiesto_simulado(fuente):
            archivo = "archivo.parquet"

            return {
                "source": fuente,
                "run_id": run_id,
                "estado": "complete",
                "faltantes": [],
                "archivos_esperados": 1,
                "archivos": [
                    {
                        "archivo": archivo,
                        "remote_path": (
                            f"{fuente}/run_id={run_id}/"
                            f"{archivo}"
                        ),
                        "sha256": "a" * 64,
                        "size_bytes": 100,
                    }
                ],
            }

        manifests = {
            fuente: manifiesto_simulado(fuente)
            for fuente in (
                "movielens",
                "tmdb",
                "watchmode",
                "integrated",
            )
        }

        # Simular que Integrated está incompleto.
        manifests["integrated"]["estado"] = "incomplete"
        manifests["integrated"]["faltantes"] = [
            "movies_integrated.parquet"
        ]

        with patch(
            "src.load.silver_storage."
            "upload_file_to_storage"
        ) as mock_upload:

            resultado = validate_silver_manifests(
                manifests=manifests,
                run_id=run_id,
            )

        self.assertFalse(resultado["autorizado"])

        self.assertEqual(
            resultado["estado"],
            "blocked",
        )

        self.assertEqual(
            resultado["bloqueados"]["integrated"],
            "incomplete_manifest",
        )

        self.assertEqual(
            len(resultado["fuentes_validadas"]),
            3,
        )

        mock_upload.assert_not_called()


    
    def test_publicacion_silver_devuelve_estado_global(self):

        from src.load.silver_storage import (
            upload_silver_to_supabase,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        def subida_simulada(local_dir, source, run_id):

            resultado = {
                "source": source,
                "run_id": run_id,
                "subidos": 1,
                "existentes": 0,
                "errores": [],
            }

            # Simular fallo de publicación Watchmode.
            if source == "watchmode":
                resultado["subidos"] = 0
                resultado["errores"] = [
                    {
                        "archivo": "availability.parquet",
                        "error": "Error simulado",
                    }
                ]

            return resultado

        with patch(
            "src.load.silver_storage."
            "upload_silver_directory",
            side_effect=subida_simulada,
        ) as mock_subida:

            resultado = upload_silver_to_supabase(
                run_id=run_id
            )
        
            patch(
                "src.load.silver_storage.build_silver_manifest",
                return_value={
                    "estado": "complete",
                },
            ),
            patch(
                "src.load.silver_storage.validate_silver_manifests",
                return_value={
                    "estado": "ready",
                    "autorizado": True,
                },
            ),

        self.assertEqual(
            mock_subida.call_count,
            4,
        )

        self.assertEqual(
            resultado["run_id"],
            run_id,
        )

        self.assertEqual(
            resultado["estado"],
            "partial",
        )

        self.assertEqual(
            resultado["evaluacion"]["errores"],
            1,
        )

        self.assertEqual(
            resultado["evaluacion"]["detalle"]["watchmode"],
            "failed",
        )

        self.assertEqual(
            len(resultado["fuentes"]),
            4,
        )

        for fuente in resultado["fuentes"]:
            self.assertEqual(
                resultado["fuentes"][fuente]["run_id"],
                run_id,
            )

    
    def test_publicacion_silver_bloqueada_antes_de_subir(self):

        from src.load.silver_storage import (
            upload_silver_to_supabase,
        )

        run_id = (
            "12345678-1234-4234-8234-123456789abc"
        )

        def manifiesto_simulado(local_dir, source, run_id):

            estado = (
                "incomplete"
                if source == "integrated"
                else "complete"
            )

            return {
                "source": source,
                "run_id": run_id,
                "estado": estado,
                "faltantes": (
                    ["movies_integrated.parquet"]
                    if source == "integrated"
                    else []
                ),
                "archivos_esperados": 1,
                "archivos": [{
                    "archivo": "archivo.parquet",
                    "remote_path": (
                        f"{source}/run_id={run_id}/"
                        "archivo.parquet"
                    ),
                    "sha256": "a" * 64,
                    "size_bytes": 100,
                }],
            }

        with (
            patch(
                "src.load.silver_storage.build_silver_manifest",
                side_effect=manifiesto_simulado,
            ) as mock_manifest,
            patch(
                "src.load.silver_storage.upload_silver_directory"
            ) as mock_subida,
            patch(
                "src.load.silver_storage.upload_file_to_storage"
            ) as mock_upload,
        ):

            resultado = upload_silver_to_supabase(
                run_id=run_id,
            )

        self.assertEqual(
            mock_manifest.call_count,
            4,
        )

        self.assertEqual(
            resultado["estado"],
            "blocked",
        )

        self.assertEqual(
            resultado["run_id"],
            run_id,
        )

        self.assertFalse(
            resultado["preflight"]["autorizado"]
        )

        self.assertEqual(
            resultado["preflight"]["bloqueados"]["integrated"],
            "incomplete_manifest",
        )

        self.assertEqual(
            resultado["fuentes"],
            {},
        )

        # No se permite ninguna subida.
        mock_subida.assert_not_called()
        mock_upload.assert_not_called()


    def test_archivo_existente_identico_se_reutiliza(self):

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            archivo = carpeta / "movies.parquet"
            archivo.write_bytes(b"contenido_simulado")

            with (
                patch(
                    "src.load.silver_storage."
                    "upload_file_to_storage",
                    side_effect=RuntimeError(
                        "The resource already exists"
                    ),
                ) as mock_upload,
                patch(
                    "src.load.silver_storage."
                    "compare_silver_storage_file",
                    return_value={
                        "estado": "identical",
                        "iguales": True,
                    },
                ) as mock_comparacion,
            ):

                resultado = upload_silver_directory(
                    local_dir=carpeta,
                    source="movielens",
                )

        # La subida inicial se intentó una sola vez.
        mock_upload.assert_called_once()

        # Se realizó una comparación.
        mock_comparacion.assert_called_once()

        self.assertEqual(
            mock_comparacion.call_args.kwargs["bucket"],
            "Silver",
        )

        self.assertEqual(
            mock_comparacion.call_args.kwargs["local_path"],
            archivo,
        )

        # El archivo no se subió nuevamente.
        self.assertEqual(resultado["subidos"], 0)

        # Se reconoce como existente e idéntico.
        self.assertEqual(resultado["existentes"], 1)

        # No hay conflicto.
        self.assertEqual(resultado["errores"], [])


    def test_ruta_silver_versionada_por_ejecucion(self):

        from src.load.silver_storage import (
            build_silver_versioned_path,
        )

        run_id_1 = (
            "12345678-1234-4234-8234-123456789abc"
        )

        run_id_2 = (
            "87654321-4321-4321-8321-cba987654321"
        )

        ruta_1 = build_silver_versioned_path(
            source="movielens",
            relative_path="ratings/part_000.parquet",
            run_id=run_id_1,
        )

        ruta_2 = build_silver_versioned_path(
            source="movielens",
            relative_path="ratings/part_000.parquet",
            run_id=run_id_2,
        )

        self.assertEqual(
            ruta_1,
            (
                f"movielens/run_id={run_id_1}/"
                "ratings/part_000.parquet"
            ),
        )

        # Dos ejecuciones tienen rutas diferentes.
        self.assertNotEqual(ruta_1, ruta_2)

        # No permitir rutas que salgan de la carpeta.
        with self.assertRaisesRegex(
            ValueError,
            "Ruta relativa inválida",
        ):
            build_silver_versioned_path(
                source="movielens",
                relative_path="../movies.parquet",
                run_id=run_id_1,
            )

    
    def test_error_conexion_al_comparar_silver(self):

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            archivo = carpeta / "movies.parquet"
            archivo.write_bytes(b"contenido_local")

            with (
                patch(
                    "src.load.silver_storage."
                    "upload_file_to_storage",
                    side_effect=RuntimeError(
                        "The resource already exists"
                    ),
                ) as mock_upload,
                patch(
                    "src.load.silver_storage."
                    "compare_silver_storage_file",
                    side_effect=ConnectionError(
                        "Conexion simulada interrumpida"
                    ),
                ) as mock_comparacion,
            ):

                resultado = upload_silver_directory(
                    local_dir=carpeta,
                    source="movielens",
                )

        # Una sola subida, sin reintentos peligrosos.
        mock_upload.assert_called_once()

        # Se intentó verificar el archivo remoto.
        mock_comparacion.assert_called_once()

        # No se confirma ninguna subida ni coincidencia.
        self.assertEqual(resultado["subidos"], 0)
        self.assertEqual(resultado["existentes"], 0)

        # Se registra el error de comparación.
        self.assertEqual(len(resultado["errores"]), 1)

        self.assertEqual(
            resultado["errores"][0]["error"],
            "remote_comparison_failed",
        )

        self.assertIn(
            "Conexion simulada interrumpida",
            resultado["errores"][0]["detalle"],
        )

        # La subida nunca autoriza sobrescritura.
        self.assertFalse(
            mock_upload.call_args.kwargs["upsert"]
        )

    
    def test_archivo_existente_no_se_acepta_sin_verificar(self):

        with TemporaryDirectory() as temporal:

            carpeta = Path(temporal)

            archivo = carpeta / "movies.parquet"
            archivo.write_bytes(b"contenido_simulado")

            with (
                patch(
                    "src.load.silver_storage."
                    "upload_file_to_storage",
                    side_effect=RuntimeError(
                        "The resource already exists"
                    ),
                ) as mock_upload,
                patch(
                    "src.load.silver_storage."
                    "compare_silver_storage_file",
                    return_value={
                        "estado": "conflict",
                        "iguales": False,
                    },
                ) as mock_comparacion,
            ):

                resultado = upload_silver_directory(
                    local_dir=carpeta,
                    source="movielens",
                )

            # Comprobar que se utilizaron los mocks.
            mock_upload.assert_called_once()
            mock_comparacion.assert_called_once()

            # No se subió ningún archivo.
            self.assertEqual(
                resultado["subidos"],
                0,
            )

            # No se aceptó como idéntico.
            self.assertEqual(
                resultado["existentes"],
                0,
            )

            # El contenido diferente genera conflicto.
            self.assertEqual(
                len(resultado["errores"]),
                1,
            )

            self.assertEqual(
                resultado["errores"][0]["error"],
                "remote_object_exists_unverified",
            )

            self.assertEqual(
                resultado["errores"][0]["archivo"],
                "movies.parquet",
            )

            # Nunca debemos permitir sobrescrituras.
            self.assertFalse(
                mock_upload.call_args.kwargs["upsert"]
            )


if __name__ == "__main__":
    unittest.main()
