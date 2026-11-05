import unittest
import json
import hashlib
from unittest.mock import patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone, timedelta

from src.ingestion.watchmode import (
    run_watchmode_ingestion,
    MIN_CREDITS_RESERVE,
    calculate_watchmode_budget,
    get_pending_watchmode_ids,
    plan_watchmode_requests,
    get_local_watchmode_files,
    validate_watchmode_local_file,
    upload_watchmode_to_supabase,
    build_local_watchmode_path,
    save_watchmode_raw,
    extract_watchmode_movies,
    validate_watchmode,
    inventory_legacy_watchmode_files,
    get_watchmode_provenance,
    save_watchmode_provenance,
    check_watchmode_freshness,
    update_watchmode_safely,
    plan_watchmode_sync,
    prepare_watchmode_sync_bundles,
    sync_watchmode_bundles,
    inspect_incomplete_watchmode_bundles,
    recover_incomplete_watchmode_bundles,
    audit_complete_watchmode_bundles,
    compare_watchmode_versions,
    backup_watchmode_remote_version,
    WatchmodeBackupError,
    inspect_watchmode_backup,
)


class WatchmodeTests(unittest.TestCase):

    
    def test_respaldo_error_conexion(self):

        version_id = (
            "123e4567-e89b-42d3-a456-426614174000"
        )

        error = WatchmodeBackupError(
            message="Respaldo incompleto",
            tmdb_id=550,
            region="PE",
            version_id=version_id,
            rutas_historicas={
                "json": "history/sources.json",
                "metadata": "history/metadata.json",
            },
        )

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.download.side_effect = RuntimeError(
            "Error de conexión"
        )

        with patch(
            "src.ingestion.watchmode.get_supabase_client",
            return_value=cliente,
        ):
            resultado = inspect_watchmode_backup(
                error
            )

        self.assertEqual(
            resultado["estado"],
            "error"
        )

        self.assertFalse(
            resultado["integridad_verificada"]
        )

        self.assertEqual(
            len(resultado["errores"]),
            2
        )

    def test_inspeccionar_respaldo_incompleto(self):

        version_id = (
            "123e4567-e89b-42d3-a456-426614174000"
        )

        error = WatchmodeBackupError(
            message="Respaldo incompleto",
            tmdb_id=550,
            region="PE",
            version_id=version_id,
            rutas_historicas={
                "json": "history/sources.json",
                "metadata": "history/metadata.json",
            },
        )

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        def descargar(ruta):
            if ruta == "history/sources.json":
                return b"[]"

            class ObjetoNoEncontrado(Exception):
                status_code = 404

            raise ObjetoNoEncontrado(
                "Objeto no encontrado"
            )

        storage.download.side_effect = descargar

        with patch(
            "src.ingestion.watchmode.get_supabase_client",
            return_value=cliente,
        ):
            resultado = inspect_watchmode_backup(
                error
            )

        self.assertTrue(resultado["json_existe"])
        self.assertFalse(resultado["metadata_existe"])

        self.assertFalse(
            resultado["integridad_verificada"]
        )

        self.assertEqual(
            resultado["estado"],
            "incomplete"
        )

    
    def test_respaldo_historico_watchmode(self):

        import hashlib

        contenido_json = b'[{"name": "Netflix"}]'

        hash_json = hashlib.sha256(
            contenido_json
        ).hexdigest()

        contenido_metadata = json.dumps({
            "tmdb_id": 550,
            "region": "PE",
            "region_verified": True,
            "sha256": hash_json,
        }).encode("utf-8")

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.download.side_effect = [
            contenido_json,
            contenido_metadata,
            contenido_json,
            contenido_metadata,
        ]

        with (
            patch(
                "src.ingestion.watchmode.get_supabase_client",
                return_value=cliente,
            ),
            patch(
                "src.ingestion.watchmode.uuid4",
                return_value=(
                    "123e4567-e89b-42d3-a456-426614174000"
                ),
            ),
        ):
            resultado = backup_watchmode_remote_version(
                tmdb_id=550,
                region="PE",
            )

        self.assertEqual(
            resultado["estado"],
            "verified",
        )

        self.assertEqual(
            storage.copy.call_count,
            2,
        )

        self.assertEqual(
            storage.download.call_count,
            4,
        )

        self.assertEqual(
            resultado["sha256"],
            hash_json,
        )

    
    def test_respaldo_historico_incompleto(self):

        contenido_json = b'[{"name": "Netflix"}]'

        hash_json = hashlib.sha256(
            contenido_json
        ).hexdigest()

        contenido_metadata = json.dumps({
            "tmdb_id": 550,
            "region": "PE",
            "region_verified": True,
            "sha256": hash_json,
        }).encode("utf-8")

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        # Lectura correcta de la versión vigente
        storage.download.side_effect = [
            contenido_json,
            contenido_metadata,
        ]

        # Primera copia correcta, segunda fallida
        storage.copy.side_effect = [
            {"status": "ok"},
            RuntimeError(
                "Error al copiar metadatos"
            ),
        ]

        with patch(
            "src.ingestion.watchmode.get_supabase_client",
            return_value=cliente,
        ):
            with self.assertRaises(
                WatchmodeBackupError
            ) as contexto:
                backup_watchmode_remote_version(
                    tmdb_id=550,
                    region="PE",
                )

        self.assertIn(
            "Error al copiar metadatos",
            str(contexto.exception),
        )

        # Se intentaron las dos copias
        self.assertEqual(
            storage.copy.call_count,
            2,
        )

        # No hubo verificación posterior
        self.assertEqual(
            storage.download.call_count,
            2,
        )
        
        
        error = contexto.exception

        self.assertEqual(error.tmdb_id, 550)
        self.assertEqual(error.region, "PE")

        self.assertIn(
            "json",
            error.rutas_historicas,
        )

        self.assertIn(
            "metadata",
            error.rutas_historicas,
        )

        self.assertIsInstance(
            error.__cause__,
            RuntimeError,
        )



    def test_comparar_versiones_watchmode(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                self.crear_pelicula_vigente(550, "PE")

                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                metadata_local = archivo.with_name(
                    archivo.stem + "_metadata.json"
                )

                datos_locales = json.loads(
                    metadata_local.read_text(
                        encoding="utf-8"
                    )
                )

                fecha_local = datos_locales["retrieved_at"]

                contenido_remoto = b'[{"name": "Prime Video"}]'

                import hashlib

                hash_remoto = hashlib.sha256(
                    contenido_remoto
                ).hexdigest()

                metadata_remota = json.dumps({
                    "retrieved_at": "2026-10-20T12:00:00+00:00",
                    "sha256": hash_remoto,
                }).encode("utf-8")

                plan = {
                    "region": "PE",
                    "existentes": [{
                        "archivo": archivo.name,
                        "origen": str(archivo),
                        "destino": (
                            "watchmode/regions/PE/"
                            "movie_550_sources.json"
                        ),
                    }],
                }

                auditoria = {
                    "region": "PE",
                    "verificados": [{
                        "archivo": archivo.name,
                        "sha256": hash_remoto,
                    }],
                }

                cliente = MagicMock()
                storage = cliente.storage.from_.return_value
                storage.download.return_value = metadata_remota

                with (
                    patch(
                        "src.ingestion.watchmode.plan_watchmode_sync",
                        return_value=plan,
                    ),
                    patch(
                        "src.ingestion.watchmode.audit_complete_watchmode_bundles",
                        return_value=auditoria,
                    ),
                    patch(
                        "src.ingestion.watchmode.get_supabase_client",
                        return_value=cliente,
                    ),
                ):
                    resultado = compare_watchmode_versions(
                        region="PE"
                    )

        self.assertEqual(len(resultado["diferentes"]), 1)

        self.assertTrue(
            resultado["diferentes"][0]["local_mas_reciente"]
        )

        self.assertEqual(resultado["errores"], [])


    
    def test_auditoria_detecta_identidad_incorrecta(self):

        contenido_json = b"[]"

        import hashlib

        hash_json = hashlib.sha256(
            contenido_json
        ).hexdigest()

        metadata = json.dumps({
            "tmdb_id": 999,
            "region": "CL",
            "region_verified": True,
            "sha256": hash_json,
        }).encode("utf-8")

        plan = {
            "region": "PE",
            "existentes": [{
                "archivo": "movie_550_sources.json",
                "destino": (
                    "watchmode/regions/PE/"
                    "movie_550_sources.json"
                ),
            }],
        }

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.download.side_effect = [
            contenido_json,
            metadata,
        ]

        with (
            patch(
                "src.ingestion.watchmode.plan_watchmode_sync",
                return_value=plan,
            ),
            patch(
                "src.ingestion.watchmode.get_supabase_client",
                return_value=cliente,
            ),
        ):
            resultado = audit_complete_watchmode_bundles(
                region="PE"
            )

        self.assertEqual(
            len(resultado["verificados"]), 0
        )

        self.assertEqual(
            len(resultado["conflictos"]), 1
        )

        self.assertEqual(
            resultado["conflictos"][0]["motivo"],
            "metadata_identity_mismatch"
        )

    
    def test_recuperacion_detecta_hash_diferente(self):

        with TemporaryDirectory() as temporal:
            archivo = Path(temporal) / "metadata.json"
            archivo.write_bytes(b"contenido local")

            diagnostico = {
                "region": "PE",
                "recuperables": [{
                    "archivo": "movie_550_sources.json",
                    "tipo": "metadata_missing",
                    "local_path": str(archivo),
                    "remote_path": (
                        "watchmode/regions/PE/"
                        "movie_550_sources_metadata.json"
                    ),
                }],
                "conflictos": [],
                "errores": [],
            }

            cliente = MagicMock()
            storage = cliente.storage.from_.return_value

            storage.download.return_value = (
                b"contenido remoto diferente"
            )

            with (
                patch(
                    "src.ingestion.watchmode."
                    "inspect_incomplete_watchmode_bundles",
                    return_value=diagnostico,
                ),
                patch(
                    "src.ingestion.watchmode.upload_file_to_storage"
                ),
                patch(
                    "src.ingestion.watchmode.get_supabase_client",
                    return_value=cliente,
                ),
            ):
                resultado = recover_incomplete_watchmode_bundles(
                    region="PE"
                )

        self.assertEqual(resultado["recuperados"], 0)
        self.assertEqual(resultado["fallidos"], 1)
        self.assertEqual(len(resultado["errores"]), 1)
        self.assertIn(
            "SHA-256",
            resultado["errores"][0]["error"]
        )

    
    def test_auditar_paquete_completo_watchmode(self):

        contenido_json = b'[{"name": "Netflix"}]'

        import hashlib

        hash_json = hashlib.sha256(
            contenido_json
        ).hexdigest()

        metadata = json.dumps({
            "tmdb_id": 550,
            "region": "PE",
            "region_verified": True,
            "sha256": hash_json,
        }).encode("utf-8")

        plan = {
            "region": "PE",
            "existentes": [{
                "archivo": "movie_550_sources.json",
                "destino": (
                    "watchmode/regions/PE/"
                    "movie_550_sources.json"
                ),
            }],
        }

        cliente = MagicMock()
        storage = cliente.storage.from_.return_value

        storage.download.side_effect = [
            contenido_json,
            metadata,
        ]

        with (
            patch(
                "src.ingestion.watchmode.plan_watchmode_sync",
                return_value=plan,
            ),
            patch(
                "src.ingestion.watchmode.get_supabase_client",
                return_value=cliente,
            ),
        ):

            resultado = audit_complete_watchmode_bundles(
                region="PE"
            )

        self.assertEqual(
            len(resultado["verificados"]),
            1
        )

        self.assertEqual(
            resultado["conflictos"],
            []
        )

        self.assertEqual(
            resultado["errores"],
            []
        )

    
    def test_recuperar_paquete_incompleto(self):

        diagnostico = {
            "region": "PE",
            "recuperables": [
                {
                    "archivo": "movie_550_sources.json",
                    "tipo": "metadata_missing",
                    "local_path": (
                        "data/bronze/watchmode/regions/PE/"
                        "movie_550_sources_metadata.json"
                    ),
                    "remote_path": (
                        "watchmode/regions/PE/"
                        "movie_550_sources_metadata.json"
                    ),
                }
            ],
            "conflictos": [],
            "errores": [],
        }

        with TemporaryDirectory() as temporal:
            archivo = (
                Path(temporal)
                / "movie_550_sources_metadata.json"
            )

            contenido = b'{"region": "PE"}'
            archivo.write_bytes(contenido)

            diagnostico["recuperables"][0][
                "local_path"
            ] = str(archivo)

            cliente = MagicMock()

            storage = cliente.storage.from_.return_value
            storage.download.return_value = contenido

            with (
                patch(
                    "src.ingestion.watchmode."
                    "inspect_incomplete_watchmode_bundles",
                    return_value=diagnostico,
                ),
                patch(
                    "src.ingestion.watchmode.upload_file_to_storage"
                ) as mock_upload,
                patch(
                    "src.ingestion.watchmode.get_supabase_client",
                    return_value=cliente,
                ),
            ):
                resultado = recover_incomplete_watchmode_bundles(
                    region="PE"
                )

        self.assertEqual(
            resultado["recuperados"], 1
        )

        self.assertEqual(
            resultado["fallidos"], 0
        )

        self.assertEqual(
            resultado["errores"], []
        )

        mock_upload.assert_called_once()

        argumentos = mock_upload.call_args.kwargs

        self.assertEqual(
            argumentos["bucket"],
            "Bronze"
        )

        self.assertEqual(
            argumentos["remote_path"],
            (
                "watchmode/regions/PE/"
                "movie_550_sources_metadata.json"
            )
        )

        self.assertFalse(
            argumentos["upsert"]
        )

    
    def test_inspeccionar_paquete_incompleto(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio,
                ),
                patch(
                    "src.ingestion.watchmode.list_storage_filenames",
                    return_value={
                        "movie_550_sources.json"
                    },
                ),
            ):

                self.crear_pelicula_vigente(550, "PE")

                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                cliente = MagicMock()

                storage = cliente.storage.from_.return_value
                storage.download.return_value = (
                    archivo.read_bytes()
                )

                with patch(
                    "src.ingestion.watchmode.get_supabase_client",
                    return_value=cliente,
                ):
                    resultado = (
                        inspect_incomplete_watchmode_bundles(
                            region="PE"
                        )
                    )

        self.assertEqual(
            len(resultado["recuperables"]), 1
        )

        self.assertEqual(
            resultado["recuperables"][0]["tipo"],
            "metadata_missing"
        )

        self.assertEqual(
            resultado["conflictos"], []
        )

        self.assertEqual(
            resultado["errores"], []
        )

    
    def test_sincronizar_paquete_watchmode(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                self.crear_pelicula_vigente(550, "PE")

                with (
                    patch(
                        "src.ingestion.watchmode.list_storage_filenames",
                        return_value=set(),
                    ),
                    patch(
                        "src.ingestion.watchmode.upload_file_to_storage"
                    ) as mock_upload,
                ):
                    resultado = sync_watchmode_bundles(
                        region="PE"
                    )

        self.assertEqual(
            resultado["paquetes_subidos"], 1
        )

        self.assertEqual(
            resultado["paquetes_fallidos"], 0
        )

        self.assertEqual(
            resultado["errores"], []
        )

        self.assertEqual(
            mock_upload.call_count, 2
        )

        rutas = [
            llamada.kwargs["remote_path"]
            for llamada in mock_upload.call_args_list
        ]

        self.assertEqual(
            rutas,
            [
                "watchmode/regions/PE/movie_550_sources.json",
                "watchmode/regions/PE/"
                "movie_550_sources_metadata.json",
            ]
        )

    
    def test_plan_sincronizacion_watchmode(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio,
                ),
                patch(
                    "src.ingestion.watchmode.list_storage_filenames",
                    return_value={
                        "movie_100_sources.json",
                        "movie_100_sources_metadata.json",
                    },
                ) as mock_list,
            ):

                self.crear_pelicula_vigente(100, "PE")
                self.crear_pelicula_vigente(200, "PE")

                plan = plan_watchmode_sync(
                    region="PE"
                )

            self.assertEqual(
                plan["archivos_locales"], 2
            )

            self.assertEqual(
                len(plan["existentes"]), 1
            )

            self.assertEqual(
                len(plan["pendientes"]), 1
            )

            self.assertEqual(
                len(plan["invalidos"]), 0
            )

            self.assertEqual(
                plan["pendientes"][0]["destino"],
                "watchmode/regions/PE/movie_200_sources.json"
            )

            mock_list.assert_called_once_with(
                bucket="Bronze",
                folder="watchmode/regions/PE",
            )

    
    def test_detectar_paquete_incompleto(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio,
                ),
                patch(
                    "src.ingestion.watchmode.list_storage_filenames",
                    return_value={
                        "movie_550_sources.json"
                    },
                ),
            ):

                self.crear_pelicula_vigente(
                    tmdb_id=550,
                    region="PE",
                )

                resultado = plan_watchmode_sync(
                    region="PE"
                )

        self.assertEqual(
            len(resultado["incompletos"]), 1
        )

        self.assertEqual(
            len(resultado["pendientes"]), 0
        )

        self.assertEqual(
            len(resultado["existentes"]), 0
        )

        registro = resultado["incompletos"][0]

        self.assertTrue(
            registro["json_existe"]
        )

        self.assertFalse(
            registro["metadata_existe"]
        )

    
    def test_actualizacion_segura_ante_error(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                self.crear_pelicula_vigente(550, "PE")

                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                metadata = archivo.with_name(
                    archivo.stem + "_metadata.json"
                )

                json_anterior = archivo.read_bytes()
                metadata_anterior = metadata.read_bytes()

                with patch(
                    "src.ingestion.watchmode.save_watchmode_provenance",
                    side_effect=OSError(
                        "Error simulado de metadatos"
                    ),
                ):
                    with self.assertRaises(OSError):
                        update_watchmode_safely(
                            tmdb_id=550,
                            data=[{"name": "Prime Video"}],
                            region="PE",
                        )

                self.assertEqual(
                    archivo.read_bytes(),
                    json_anterior,
                )

                self.assertEqual(
                    metadata.read_bytes(),
                    metadata_anterior,
                )

    
    def test_escritura_metadata_fallida_conserva_original(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                self.crear_pelicula_vigente(550, "PE")

                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                metadata = archivo.with_name(
                    archivo.stem + "_metadata.json"
                )

                contenido_original = metadata.read_bytes()

                with patch(
                    "src.ingestion.watchmode.json.dump",
                    side_effect=OSError(
                        "Error simulado de escritura"
                    ),
                ):
                    with self.assertRaises(OSError):
                        save_watchmode_provenance(
                            tmdb_id=550,
                            region="PE",
                        )

                self.assertEqual(
                    metadata.read_bytes(),
                    contenido_original,
                )

                temporales = list(
                    metadata.parent.glob(
                        metadata.name + ".*.tmp"
                    )
                )

                self.assertEqual(temporales, [])

    
    def test_error_durante_restauracion(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                self.crear_pelicula_vigente(550, "PE")

                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                with (
                    patch(
                        "src.ingestion.watchmode.save_watchmode_provenance",
                        side_effect=OSError(
                            "Error de actualización"
                        ),
                    ),
                    patch.object(
                        Path,
                        "write_bytes",
                        side_effect=OSError(
                            "Error de restauración"
                        ),
                    ),
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        update_watchmode_safely(
                            tmdb_id=550,
                            data=[{"name": "Prime Video"}],
                            region="PE",
                        )

                self.assertIn(
                    "restauración",
                    str(ctx.exception)
                )

                self.assertIsInstance(
                    ctx.exception.__cause__,
                    OSError,
                )


    def crear_pelicula_vigente(
        self,
        tmdb_id,
        region="PE",
    ):
        save_watchmode_raw(
            tmdb_id=tmdb_id,
            data=[],
            region=region,
        )

        save_watchmode_provenance(
            tmdb_id=tmdb_id,
            region=region,
        )

    
    def test_vigencia_watchmode(self):

        ahora = datetime(
            2026, 11, 4, 18, 0,
            tzinfo=timezone.utc
        )

        casos = [
            (
                "fresh",
                ahora - timedelta(days=2),
                True,
            ),
            (
                "stale",
                ahora - timedelta(days=10),
                False,
            ),
        ]

        for estado, fecha, esperado in casos:

            with self.subTest(estado=estado):

                procedencia = {
                    "exists": True,
                    "region_verified": True,
                    "status": "verified",
                    "retrieved_at": fecha.isoformat(),
                }

                with patch(
                    "src.ingestion.watchmode.get_watchmode_provenance",
                    return_value=procedencia,
                ):

                    resultado = check_watchmode_freshness(
                        tmdb_id=550,
                        region="PE",
                        max_age_days=7,
                        now=ahora,
                    )

                self.assertEqual(
                    resultado["status"],
                    estado
                )

                self.assertEqual(
                    resultado["fresh"],
                    esperado
                )


    def test_guardar_procedencia_nueva_consulta(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                save_watchmode_raw(
                    tmdb_id=550,
                    data=[{"name": "Netflix"}],
                    region="PE",
                )

                ruta = save_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

            self.assertTrue(ruta.is_file())

            metadata = json.loads(
                ruta.read_text(encoding="utf-8")
            )

            self.assertEqual(metadata["tmdb_id"], 550)
            self.assertEqual(metadata["region"], "PE")
            self.assertTrue(metadata["region_verified"])
            self.assertEqual(metadata["sources_count"], 1)
            self.assertIn("retrieved_at", metadata)

    
    def test_procedencia_prioriza_consulta_nueva(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                save_watchmode_raw(
                    tmdb_id=550,
                    data=[{"name": "Netflix"}],
                    region="PE",
                )

                # Primero, registrar migración histórica
                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                migracion = {
                    "records": [
                        {
                            "archivo": archivo.name,
                            "region_verified": False,
                        }
                    ]
                }

                (
                    archivo.parent / "_migration_metadata.json"
                ).write_text(
                    json.dumps(migracion),
                    encoding="utf-8",
                )

                # Registrar una consulta nueva verificada
                save_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

                resultado = get_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

        self.assertTrue(resultado["exists"])
        self.assertTrue(resultado["region_verified"])
        self.assertEqual(
            resultado["status"],
            "verified"
        )
        self.assertIsNotNone(resultado["retrieved_at"])

    
    def test_extraccion_registra_procedencia(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio
                ),
                patch(
                    "src.ingestion.watchmode.extract_watchmode_sources",
                    return_value=[{"name": "Netflix"}]
                ) as mock_api,
                patch(
                    "src.ingestion.watchmode.time.sleep"
                ),
            ):

                resultado = extract_watchmode_movies(
                    tmdb_ids=[550],
                    api_key="api_simulada",
                    region="PE",
                )

                procedencia = get_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

            self.assertEqual(resultado["exitosos"], 1)

            self.assertEqual(resultado["errores"], [])

            self.assertTrue(procedencia["exists"])

            self.assertTrue(
                procedencia["region_verified"]
            )

            self.assertEqual(
                procedencia["status"],
                "verified"
            )

            self.assertIsNotNone(
                procedencia["retrieved_at"]
            )

            mock_api.assert_called_once()

    
    
    def test_procedencia_detecta_json_modificado(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):
                archivo = save_watchmode_raw(
                    tmdb_id=550,
                    data=[{"name": "Netflix"}],
                    region="PE",
                )

                save_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

                # Verificar procedencia inicial
                resultado_inicial = get_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

                self.assertEqual(
                    resultado_inicial["status"],
                    "verified"
                )

                # Modificar el JSON sin actualizar metadatos
                archivo.write_text(
                    '[{"name": "Prime Video"}]',
                    encoding="utf-8",
                )

                resultado_final = get_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

        self.assertFalse(
            resultado_final["region_verified"]
        )

        self.assertEqual(
            resultado_final["status"],
            "unknown_provenance"
        )

    
    def test_procedencia_watchmode_migrado(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            carpeta_pe = directorio / "regions" / "PE"
            carpeta_pe.mkdir(parents=True)

            archivo = (
                carpeta_pe / "movie_550_sources.json"
            )
            archivo.write_text(
                "[]",
                encoding="utf-8"
            )

            metadata = {
                "records": [
                    {
                        "archivo": "movie_550_sources.json",
                        "region_verified": False,
                    }
                ]
            }

            (
                carpeta_pe / "_migration_metadata.json"
            ).write_text(
                json.dumps(metadata),
                encoding="utf-8"
            )

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):
                resultado = get_watchmode_provenance(
                    tmdb_id=550,
                    region="PE",
                )

        self.assertTrue(resultado["exists"])
        self.assertFalse(resultado["region_verified"])
        self.assertEqual(
            resultado["status"],
            "region_unverified"
        )

    
    def test_inventario_watchmode_antiguo(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            # Archivo antiguo válido
            (directorio / "movie_100_sources.json").write_text(
                "[]",
                encoding="utf-8"
            )

            # Archivo antiguo inválido
            (directorio / "movie_200_sources.json").write_text(
                '{"error": "incorrecto"}',
                encoding="utf-8"
            )

            # Archivo regional: no debe incluirse
            carpeta_pe = directorio / "regions" / "PE"
            carpeta_pe.mkdir(parents=True)

            (carpeta_pe / "movie_300_sources.json").write_text(
                "[]",
                encoding="utf-8"
            )

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):
                inventario = inventory_legacy_watchmode_files()

        self.assertEqual(inventario["total"], 2)
        self.assertEqual(len(inventario["validos"]), 1)
        self.assertEqual(len(inventario["invalidos"]), 1)

        self.assertEqual(
            inventario["validos"][0]["archivo"],
            "movie_100_sources.json"
        )

    
    def test_metadata_migracion_watchmode(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)
            carpeta_pe = directorio / "regions" / "PE"
            carpeta_pe.mkdir(parents=True)

            metadata = {
                "region_assumed": "PE",
                "region_verified": False,
                "records": [
                    {
                        "archivo": "movie_550_sources.json",
                        "sha256": "hash_simulado",
                        "region_assumed": "PE",
                        "region_verified": False,
                    }
                ],
            }

            ruta_metadata = (
                carpeta_pe / "_migration_metadata.json"
            )
            ruta_metadata.write_text(
                json.dumps(metadata),
                encoding="utf-8",
            )

            datos = json.loads(
                ruta_metadata.read_text(encoding="utf-8")
            )

            self.assertEqual(
                datos["region_assumed"], "PE"
            )
            self.assertFalse(
                datos["region_verified"]
            )
            self.assertEqual(len(datos["records"]), 1)

    
    def test_validacion_watchmode_por_region(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            carpeta_pe = directorio / "regions" / "PE"
            carpeta_cl = directorio / "regions" / "CL"

            carpeta_pe.mkdir(parents=True)
            carpeta_cl.mkdir(parents=True)

            # Perú: una respuesta con fuentes
            (carpeta_pe / "movie_100_sources.json").write_text(
                '[{"name": "Netflix"}]',
                encoding="utf-8"
            )

            # Perú: una respuesta vacía válida
            (carpeta_pe / "movie_200_sources.json").write_text(
                "[]",
                encoding="utf-8"
            )

            # Chile: JSON inválido, no debe afectar a Perú
            (carpeta_cl / "movie_300_sources.json").write_text(
                '{"error": "incorrecto"}',
                encoding="utf-8"
            )

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):
                resultado = validate_watchmode(region="PE")

        self.assertEqual(resultado["region"], "PE")
        self.assertEqual(resultado["archivos_totales"], 2)
        self.assertEqual(resultado["archivos_validos"], 2)
        self.assertEqual(resultado["archivos_vacios"], 1)
        self.assertEqual(
            resultado["archivos_invalidos"], []
        )

    def test_extraccion_watchmode_por_region(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio,
                ),
                patch(
                    "src.ingestion.watchmode.extract_watchmode_sources",
                    return_value=[{"name": "Netflix"}],
                ) as mock_api,
                patch(
                    "src.ingestion.watchmode.time.sleep"
                ),
            ):

                # Crear película 100 con datos vigentes
                self.crear_pelicula_vigente(
                    tmdb_id=100,
                    region="PE",
                )

                # 100 existe; 200 debe consultarse
                resultado = extract_watchmode_movies(
                    tmdb_ids=[100, 200],
                    api_key="api_simulada",
                    region="PE",
                )

                archivo_nuevo = build_local_watchmode_path(
                    tmdb_id=200,
                    region="PE",
                )

                # La película 100 debe omitirse
                self.assertEqual(
                    resultado["existentes"],
                    1,
                )

                # La película 200 debe descargarse
                self.assertEqual(
                    resultado["exitosos"],
                    1,
                )

                # Solo se consulta la película 200
                mock_api.assert_called_once_with(
                    200,
                    "api_simulada",
                    region="PE",
                )

                # Comprobar archivo regional
                self.assertTrue(
                    archivo_nuevo.is_file()
                )

                datos = json.loads(
                    archivo_nuevo.read_text(
                        encoding="utf-8"
                    )
                )

                self.assertEqual(
                    datos,
                    [{"name": "Netflix"}],
                )

    
    def test_pendientes_independientes_por_region(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):

                # Crear película vigente solo para Perú
                self.crear_pelicula_vigente(
                    tmdb_id=550,
                    region="PE",
                )

                # Consultar películas pendientes en Perú
                resultado_pe = get_pending_watchmode_ids(
                    tmdb_ids=[550, 862],
                    region="PE",
                )

                # Consultar películas pendientes en Chile
                resultado_cl = get_pending_watchmode_ids(
                    tmdb_ids=[550, 862],
                    region="CL",
                )

        # PERÚ: 550 existe y está vigente
        self.assertEqual(
            resultado_pe["seleccionados"],
            [862],
        )

        self.assertEqual(
            resultado_pe["existentes"],
            1,
        )

        # CHILE: ninguna película está descargada
        self.assertEqual(
            resultado_cl["seleccionados"],
            [550, 862],
        )

        self.assertEqual(
            resultado_cl["existentes"],
            0,
        )

    def test_pendientes_segun_vigencia(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            carpeta_pe = directorio / "regions" / "PE"
            carpeta_pe.mkdir(parents=True)

            for tmdb_id in [100, 200, 300]:
                (
                    carpeta_pe /
                    f"movie_{tmdb_id}_sources.json"
                ).write_text(
                    "[]",
                    encoding="utf-8"
                )

            def vigencia_simulada(tmdb_id, region, max_age_days):
                estados = {
                    100: ("fresh", True),
                    200: ("stale", False),
                    300: ("region_unverified", False),
                }
                estado, vigente = estados[tmdb_id]

                return {
                    "status": estado,
                    "fresh": vigente,
                    "age_days": None,
                }

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio
                ),
                patch(
                    "src.ingestion.watchmode.check_watchmode_freshness",
                    side_effect=vigencia_simulada
                ),
            ):

                resultado = get_pending_watchmode_ids(
                    tmdb_ids=[100, 200, 300, 400],
                    region="PE",
                )

        self.assertEqual(resultado["existentes"], 1)
        self.assertEqual(
            resultado["seleccionados"],
            [200, 300, 400]
        )
        self.assertEqual(resultado["estados"]["fresh"], 1)
        self.assertEqual(resultado["estados"]["stale"], 1)
        self.assertEqual(
            resultado["estados"]["region_unverified"],
            1
        )
        self.assertEqual(resultado["estados"]["missing"], 1)
    
    def test_guardar_watchmode_por_region(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):

                ruta = save_watchmode_raw(
                    tmdb_id=550,
                    data=[{"name": "Netflix"}],
                    region="PE",
                )

            ruta_esperada = (
                directorio
                / "regions"
                / "PE"
                / "movie_550_sources.json"
            )

            self.assertEqual(ruta, ruta_esperada)
            self.assertTrue(ruta.is_file())

            with open(
                ruta,
                "r",
                encoding="utf-8",
            ) as archivo:
                contenido = json.load(archivo)

            self.assertEqual(
                contenido,
                [{"name": "Netflix"}],
            )

    
    def test_ruta_local_regional_watchmode(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):
                ruta = build_local_watchmode_path(
                    tmdb_id=550,
                    region="pe",
                )

            self.assertEqual(
                ruta,
                directorio
                / "regions"
                / "PE"
                / "movie_550_sources.json"
            )

    
    def test_validacion_json_watchmode(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            archivo_valido = (
                directorio / "movie_100_sources.json"
            )
            archivo_valido.write_text(
                '[{"name": "Netflix"}]',
                encoding="utf-8"
            )

            archivo_vacio = (
                directorio / "movie_200_sources.json"
            )
            archivo_vacio.write_text(
                "[]",
                encoding="utf-8"
            )

            archivo_invalido = (
                directorio / "movie_300_sources.json"
            )
            archivo_invalido.write_text(
                '{"error": "respuesta incorrecta"}',
                encoding="utf-8"
            )

            resultado_valido = validate_watchmode_local_file(
                archivo_valido
            )

            resultado_vacio = validate_watchmode_local_file(
                archivo_vacio
            )

            resultado_invalido = validate_watchmode_local_file(
                archivo_invalido
            )

        self.assertTrue(resultado_valido["valid"])
        self.assertEqual(
            resultado_valido["sources_count"], 1
        )

        self.assertTrue(resultado_vacio["valid"])
        self.assertTrue(resultado_vacio["empty"])

        self.assertFalse(resultado_invalido["valid"])

        self.assertEqual(
            resultado_invalido["error"],
            "invalid_json_structure"
        )
    
    def test_detener_ingesta_por_creditos(self):

        cuota_simulada = {
            "quota": 1000,
            "quotaUsed": 750,
        }

        with (
            patch(
                "src.ingestion.watchmode.load_watchmode_api_key",
                return_value="api_key_simulada"
            ),
            patch(
                "src.ingestion.watchmode.get_watchmode_quota",
                return_value=cuota_simulada
            ),
            patch(
                "src.ingestion.watchmode.load_tmdb_ids_from_movielens",
                return_value=[100, 200, 300]
            ),
            patch(
                "src.ingestion.watchmode.extract_watchmode_movies"
            ) as mock_extract,
            patch(
                "src.ingestion.watchmode.upload_watchmode_to_supabase"
            ) as mock_upload,
        ):

            resultado = run_watchmode_ingestion(
                limit=10
            )

        self.assertEqual(
            resultado["estado"],
            "skipped"
        )

        self.assertEqual(
            resultado["motivo"],
            "sin_consultas_autorizadas"
        )

        self.assertEqual(
            resultado["plan"]["consultas_planificadas"],
            0
        )

        mock_extract.assert_not_called()
        mock_upload.assert_not_called()

    
    def test_presupuesto_creditos(self):

        resultado = calculate_watchmode_budget(
            quota_total=1000,
            quota_used=690,
            requested_movies=20,
        )

        self.assertEqual(
            resultado["remaining_credits"], 310
        )

        self.assertEqual(
            resultado["available_credits"], 10
        )

        self.assertEqual(
            resultado["max_requests"], 5
        )

        self.assertEqual(
            resultado["allowed_requests"], 5
        )

    def test_seleccionar_peliculas_pendientes(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):

                self.crear_pelicula_vigente(100, "PE")
                self.crear_pelicula_vigente(200, "PE")
                
                vigencia_100 = check_watchmode_freshness(
                    tmdb_id=100,
                    region="PE",
                )

                print("Vigencia 100:", vigencia_100)

                self.assertTrue(vigencia_100["fresh"])

                resultado = get_pending_watchmode_ids(
                    tmdb_ids=[100, 200, 300, 400, 500],
                    limit=2,
                    region="PE",
                )

        self.assertEqual(resultado["existentes"], 2)

        self.assertEqual(
            resultado["pendientes_totales"], 3
        )

        self.assertEqual(
            resultado["seleccionados"],
            [300, 400]
        )

    
    def test_planificar_consultas_segun_creditos(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            # Crear carpeta regional de Perú
            carpeta_pe = directorio / "regions" / "PE"
            carpeta_pe.mkdir(parents=True, exist_ok=True)

            # Simular dos películas ya descargadas
            for tmdb_id in [100, 200]:

                archivo = (
                    carpeta_pe
                    / f"movie_{tmdb_id}_sources.json"
                )

            # Simular la carpeta Bronze de Watchmode
            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):

                self.crear_pelicula_vigente(100, "PE")
                self.crear_pelicula_vigente(200, "PE")
                
                resultado = plan_watchmode_requests(
                    tmdb_ids=[
                        100, 200, 300, 400,
                        500, 600, 700, 800,
                    ],
                    quota_total=1000,
                    quota_used=690,
                    requested_limit=20,
                    region="PE",
                )

        # Dos películas ya existen localmente
        self.assertEqual(
            resultado["existentes"], 2
        )

        # Seis películas están pendientes
        self.assertEqual(
            resultado["pendientes_totales"], 6
        )

        # Solo podemos consultar cinco
        self.assertEqual(
            resultado["consultas_planificadas"], 5
        )

        self.assertEqual(
            resultado["ids_seleccionados"],
            [300, 400, 500, 600, 700]
        )

        # 310 restantes - 300 de reserva = 10
        self.assertEqual(
            resultado["creditos_disponibles"], 10
        )

    
    def test_pipeline_respeta_plan_creditos(self):

        plan_simulado = {
            "existentes": 2,
            "pendientes_totales": 6,
            "ids_seleccionados": [300, 400],
            "consultas_planificadas": 2,
            "creditos_restantes": 310,
            "creditos_disponibles": 10,
        }

        extraccion_simulada = {
            "solicitados": 2,
            "exitosos": 2,
            "existentes": 0,
            "sin_disponibilidad": 0,
            "errores": [],
            "region": "PE",
        }

        validacion_simulada = {
            "archivos_totales": 2,
            "archivos_validos": 2,
            "archivos_vacios": 0,
            "archivos_invalidos": [],
        }

        subida_simulada = {
            "archivos_subidos": 2,
            "metadata_subida": True,
            "errores": [],
        }

        with (
            patch(
                "src.ingestion.watchmode.load_watchmode_api_key",
                return_value="api_simulada"
            ),
            patch(
                "src.ingestion.watchmode.get_watchmode_quota",
                return_value={
                    "quota": 1000,
                    "quotaUsed": 690,
                }
            ),
            patch(
                "src.ingestion.watchmode.load_tmdb_ids_from_movielens",
                return_value=[100, 200, 300, 400]
            ),
            patch(
                "src.ingestion.watchmode.plan_watchmode_requests",
                return_value=plan_simulado
            ),
            patch(
                "src.ingestion.watchmode.extract_watchmode_movies",
                return_value=extraccion_simulada
            ) as mock_extract,
            patch(
                "src.ingestion.watchmode.validate_watchmode",
                return_value=validacion_simulada
            ),
            patch(
                "src.ingestion.watchmode.save_watchmode_metadata"
            ),
            patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR"
            ) as mock_directorio,
            patch(
                "src.ingestion.watchmode.upload_watchmode_to_supabase",
                return_value=subida_simulada
            ),
        ):
            mock_directorio.glob.return_value = []

            run_watchmode_ingestion(limit=10, region="PE")

        mock_extract.assert_called_once_with(
            tmdb_ids=[300, 400],
            api_key="api_simulada",
            region="PE",
            limit=None,
        )

    def test_listar_archivos_locales_watchmode(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            carpeta_pe = directorio / "regions" / "PE"
            carpeta_cl = directorio / "regions" / "CL"

            carpeta_pe.mkdir(parents=True)
            carpeta_cl.mkdir(parents=True)

            # Tres películas de Perú
            for tmdb_id in [100, 200, 300]:
                archivo = (
                    carpeta_pe
                    / f"movie_{tmdb_id}_sources.json"
                )
                archivo.write_text(
                    "[]",
                    encoding="utf-8"
                )

            # Una película de Chile
            (carpeta_cl / "movie_400_sources.json").write_text(
                "[]",
                encoding="utf-8"
            )

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio
            ):
                archivos_pe = get_local_watchmode_files(
                    region="PE"
                )

                archivos_cl = get_local_watchmode_files(
                    region="CL"
                )

        self.assertEqual(len(archivos_pe), 3)
        self.assertEqual(len(archivos_cl), 1)

        self.assertEqual(
            [archivo.name for archivo in archivos_pe],
            [
                "movie_100_sources.json",
                "movie_200_sources.json",
                "movie_300_sources.json",
            ]
        )

        self.assertEqual(
            archivos_cl[0].name,
            "movie_400_sources.json"
        )
    
    
    def test_no_subir_json_watchmode_invalido(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            # Archivo con estructura incorrecta
            archivo = (
                directorio / "movie_550_sources.json"
            )

            archivo.write_text(
                '{"error": "respuesta incorrecta"}',
                encoding="utf-8"
            )

            # Metadatos de la ejecución
            metadata = directorio / "_metadata.json"

            metadata.write_text(
                '{"fuente": "Watchmode"}',
                encoding="utf-8"
            )

            with patch(
                "src.ingestion.watchmode.upload_file_to_storage"
            ) as mock_upload:

                resultado = upload_watchmode_to_supabase(
                    archivos=[archivo],
                    ruta_metadata=metadata,
                )

            # El archivo inválido no se subió
            self.assertEqual(
                resultado["archivos_subidos"], 0
            )

            # Se registró el error
            self.assertEqual(
                len(resultado["errores"]), 1
            )

            self.assertEqual(
                resultado["errores"][0]["error"],
                "invalid_json_structure"
            )

            # Solo se subieron los metadatos
            mock_upload.assert_called_once()

            self.assertEqual(
                mock_upload.call_args.kwargs["local_path"],
                metadata
            )
    
    def test_json_corrupto_se_considera_pendiente(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):

                # Crear película 100 con JSON corrupto
                carpeta_pe = directorio / "regions" / "PE"
                carpeta_pe.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                archivo_corrupto = (
                    carpeta_pe / "movie_100_sources.json"
                )

                archivo_corrupto.write_text(
                    '{"datos": incorrectos',
                    encoding="utf-8",
                )

                # Película 200 válida, verificada y vigente
                self.crear_pelicula_vigente(
                    tmdb_id=200,
                    region="PE",
                )

                # Película 300 no existe
                resultado = get_pending_watchmode_ids(
                    tmdb_ids=[100, 200, 300],
                    region="PE",
                )

        # Solo la película 200 está vigente
        self.assertEqual(
            resultado["existentes"],
            1,
        )

        # Las películas 100 y 300 requieren consulta
        self.assertEqual(
            resultado["pendientes_totales"],
            2,
        )

        self.assertEqual(
            resultado["seleccionados"],
            [100, 300],
        )

        # Verificar clasificación
        self.assertEqual(
            resultado["estados"]["corrupt"],
            1,
        )

        self.assertEqual(
            resultado["estados"]["fresh"],
            1,
        )

        self.assertEqual(
            resultado["estados"]["missing"],
            1,
        )

    
    
    def test_actualizar_pelicula_vencida(self):

        with TemporaryDirectory() as temporal:
            directorio = Path(temporal)

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio,
                ),
            ):

                # Crear película válida y con procedencia
                self.crear_pelicula_vigente(550, "PE")

                archivo = build_local_watchmode_path(
                    550, "PE"
                )

                # Simular que la película está vencida
                with (
                    patch(
                        "src.ingestion.watchmode.check_watchmode_freshness",
                        return_value={
                            "status": "stale",
                            "fresh": False,
                            "age_days": 10,
                        },
                    ),
                    patch(
                        "src.ingestion.watchmode.extract_watchmode_sources",
                        return_value=[{"name": "Prime Video"}],
                    ) as mock_api,
                    patch(
                        "src.ingestion.watchmode.time.sleep"
                    ),
                ):

                    resultado = extract_watchmode_movies(
                        tmdb_ids=[550],
                        api_key="api_simulada",
                        region="PE",
                    )

                self.assertEqual(
                    resultado["exitosos"], 1
                )

                mock_api.assert_called_once()

                # Verificar contenido actualizado
                datos = json.loads(
                    archivo.read_text(encoding="utf-8")
                )

                self.assertEqual(
                    datos,
                    [{"name": "Prime Video"}]
                )

                # Verificar respaldo histórico
                respaldos = list(
                    archivo.parent.glob(
                        "movie_550_sources.json.history.*.bak"
                    )
                )

                self.assertEqual(len(respaldos), 1)

                # La nueva procedencia debe coincidir
                procedencia = get_watchmode_provenance(
                    550, "PE"
                )

                self.assertEqual(
                    procedencia["status"],
                    "verified"
                )

    
    def test_recuperar_json_corrupto_watchmode(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            carpeta_pe = directorio / "regions" / "PE"
            carpeta_pe.mkdir(parents=True)

            archivo = (
                carpeta_pe / "movie_550_sources.json"
            )

            archivo.write_text(
                '{"datos": incorrectos',
                encoding="utf-8",
            )

            with (
                patch(
                    "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                    directorio,
                ),
                patch(
                    "src.ingestion.watchmode.extract_watchmode_sources",
                    return_value=[{"name": "Netflix"}],
                ) as mock_api,
                patch(
                    "src.ingestion.watchmode.time.sleep"
                ),
            ):

                resultado = extract_watchmode_movies(
                    tmdb_ids=[550],
                    api_key="api_simulada",
                    region="PE",
                )

            self.assertEqual(resultado["exitosos"], 1)
            self.assertEqual(resultado["existentes"], 0)
            self.assertEqual(resultado["errores"], [])

            mock_api.assert_called_once()

            # El archivo recuperado contiene datos válidos
            with open(archivo, encoding="utf-8") as f:
                contenido = json.load(f)

            self.assertEqual(
                contenido,
                [{"name": "Netflix"}],
            )

            # La copia corrupta se conserva
            respaldos = list(
                carpeta_pe.glob(
                    "movie_550_sources.json.corrupt.*.bak"
                )
            )

            self.assertEqual(len(respaldos), 1)

            self.assertEqual(
                respaldos[0].read_text(encoding="utf-8"),
                '{"datos": incorrectos',
            )

    
    def test_escritura_fallida_conserva_original(self):

        with TemporaryDirectory() as temporal:

            directorio = Path(temporal)

            with patch(
                "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                directorio,
            ):

                ruta = save_watchmode_raw(
                    tmdb_id=550,
                    data=[{"name": "Netflix"}],
                    region="PE",
                )

                contenido_original = ruta.read_bytes()

                with patch(
                    "src.ingestion.watchmode.json.dump",
                    side_effect=OSError(
                        "Error simulado de escritura"
                    ),
                ):

                    with self.assertRaises(OSError):
                        save_watchmode_raw(
                            tmdb_id=550,
                            data=[{"name": "Prime Video"}],
                            region="PE",
                        )

                self.assertEqual(
                    ruta.read_bytes(),
                    contenido_original,
                )

                # No deben quedar temporales
                temporales = list(
                    ruta.parent.glob(
                        "movie_550_sources.json.*.tmp"
                    )
                )

                self.assertEqual(temporales, [])

    def test_preparar_paquete_sincronizacion(self):
    
            with TemporaryDirectory() as temporal:
                directorio = Path(temporal)
    
                with (
                    patch(
                        "src.ingestion.watchmode.BRONZE_WATCHMODE_DIR",
                        directorio,
                    ),
                    patch(
                        "src.ingestion.watchmode.list_storage_filenames",
                        return_value=set(),
                    ) as mock_list,
                ):
    
                    self.crear_pelicula_vigente(550, "PE")
    
                    resultado = prepare_watchmode_sync_bundles(
                        region="PE"
                    )
    
                self.assertEqual(
                    len(resultado["preparados"]), 1
                )
    
                self.assertEqual(
                    resultado["errores"], []
                )
    
                paquete = resultado["preparados"][0]
    
                self.assertTrue(
                    paquete["json_local"].is_file()
                )
    
                self.assertTrue(
                    paquete["metadata_local"].is_file()
                )
    
                self.assertEqual(
                    paquete["json_remoto"],
                    "watchmode/regions/PE/movie_550_sources.json"
                )
    
                self.assertEqual(
                    paquete["metadata_remoto"],
                    "watchmode/regions/PE/"
                    "movie_550_sources_metadata.json"
                )
    
                mock_list.assert_called_once()
    


if __name__ == "__main__":
    unittest.main()
