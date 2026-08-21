# src/core/analytics.py
import os
import json
import gzip
from pathlib import Path
import shutil
from uuid import uuid4
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_access.config import (
    DATA_FOLDER,
    FORENSIC_LOG_ARCHIVE_KEEP,
    FORENSIC_LOG_ARCHIVE_PATH,
    FORENSIC_LOG_MAX_BYTES,
    FORENSIC_LOG_PATH,
)

# Intentar importar cupy para soporte de hardware acelerado
try:
    import cupy as cp
except ImportError:
    cp = None


class SniperJSONEncoder(json.JSONEncoder):
    """Codificador de élite para asegurar compatibilidad total GPU/CPU."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if cp and isinstance(obj, cp.ndarray):
            return obj.get().tolist()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return super(SniperJSONEncoder, self).default(obj)


class PerformanceTracker:
    """Sistema de Registro y Telemetría V15 (Full Spectrum Log + JSON)."""

    COLUMNS_ORDER = (
        "timestamp",
        "tag",
        "draw_id",
        "hits",
        "univ_size",
        "rank",
        "proximity",
        "logloss",
        "brier",
        "ece",
        "ai_score",
        "ai_score_kind",
        "ai_percentile_rank",
        "ai_weight_effective",
        "geo_weight_effective",
        "ai_signal_validated",
        "temporal_holdout_auc",
        "geo_score",
        "hybrid_score",
        "sniper_log",
        "event_id",
        "profile_code",
        "dataset_hash",
        "model_version",
        "seed",
        "split_id",
        "metrics_json",
    )

    def __init__(
        self,
        *,
        log_path=None,
        json_path=None,
        archive_directory=None,
        max_log_bytes=FORENSIC_LOG_MAX_BYTES,
        archive_keep=FORENSIC_LOG_ARCHIVE_KEEP,
    ):
        self.log_path = str(log_path or FORENSIC_LOG_PATH)
        self.json_path = str(
            json_path or os.path.join(DATA_FOLDER, "backtest_results.json")
        )
        self.archive_directory = Path(
            archive_directory or FORENSIC_LOG_ARCHIVE_PATH
        )
        self.max_log_bytes = max(0, int(max_log_bytes or 0))
        self.archive_keep = max(0, int(archive_keep or 0))
        self._ensure_log_exists()

    def _ensure_log_exists(self):
        columns_order = list(self.COLUMNS_ORDER)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(self.log_path):
            self._atomic_write_csv(pd.DataFrame(columns=columns_order))
            return

        try:
            existing = pd.read_csv(self.log_path)
            changed = False
            for col in columns_order:
                if col not in existing.columns:
                    existing[col] = ""
                    changed = True
            if changed:
                existing = existing[columns_order]
                self._atomic_write_csv(existing)
        except Exception:
            # Preserva incluso un CSV ilegible antes de reconstruir el activo.
            try:
                archived = self._archive_current_log()
                if archived is not None:
                    print(f"⚠️ Log forense ilegible archivado en {archived.name}")
            except Exception:
                # Si tampoco puede archivarse, no sobrescribimos el original.
                return
            self._atomic_write_csv(pd.DataFrame(columns=columns_order))

    def _atomic_write_csv(self, dataframe):
        target = Path(self.log_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            dataframe.to_csv(temporary, index=False)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _archive_current_log(self):
        source = Path(self.log_path)
        if not source.exists() or source.stat().st_size == 0:
            return None
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive = self.archive_directory / (
            f"{source.stem}_{stamp}_{uuid4().hex[:8]}.csv.gz"
        )
        temporary = archive.with_name(f".{archive.name}.tmp")
        try:
            with source.open("rb") as input_file, gzip.open(
                temporary, "wb"
            ) as output_file:
                shutil.copyfileobj(input_file, output_file)
            os.replace(temporary, archive)
        finally:
            if temporary.exists():
                temporary.unlink()
        return archive

    def _prune_archives(self):
        if self.archive_keep <= 0 or not self.archive_directory.exists():
            return []
        stem = Path(self.log_path).stem
        archives = sorted(
            self.archive_directory.glob(f"{stem}_*.csv.gz"),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        removed = []
        for archive in archives[: max(0, len(archives) - self.archive_keep)]:
            archive.unlink()
            removed.append(archive)
        return removed

    def _should_rotate(self, new_rows):
        if self.max_log_bytes <= 0 or not os.path.exists(self.log_path):
            return False
        current_size = os.path.getsize(self.log_path)
        header_size = len(",".join(self.COLUMNS_ORDER).encode("utf-8")) + 1
        if current_size <= header_size:
            return False
        incoming_size = len(
            new_rows.to_csv(index=False, header=False).encode("utf-8")
        )
        return current_size + incoming_size > self.max_log_bytes

    def log_run(self, result_dto, tag, forensic_data):
        """
        Punto de Persistencia Dual: Guarda en CSV para histórico y JSON para el modelo estático.
        """
        forensic_data = forensic_data or []

        # --- FASE 1: PERSISTENCIA CSV (Forense) ---
        self._save_to_csv(tag, forensic_data)

        # --- FASE 2: PERSISTENCIA JSON (Resumen de Misión) ---
        self._save_to_json(result_dto, tag, forensic_data)

    def _save_to_csv(self, tag, forensic_data):
        try:
            if not forensic_data:
                return

            new_rows = pd.DataFrame(forensic_data)
            new_rows["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_rows["tag"] = tag

            columns_order = list(self.COLUMNS_ORDER)

            for col in columns_order:
                if col not in new_rows.columns:
                    new_rows[col] = ""

            if "metrics_json" in new_rows.columns:
                def _compact_json(value):
                    if isinstance(value, dict):
                        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if isinstance(value, list):
                        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                    if value is None or value == "":
                        return ""
                    return str(value)

                new_rows["metrics_json"] = new_rows["metrics_json"].apply(_compact_json)

            new_rows = new_rows[columns_order]

            rotated_archive = None
            if self._should_rotate(new_rows):
                # El archivo activo no se toca hasta que el gzip se haya escrito
                # correctamente. La escritura atómica posterior reemplaza el CSV.
                rotated_archive = self._archive_current_log()

            if os.path.exists(self.log_path) and rotated_archive is None:
                existing = pd.read_csv(self.log_path)
                for col in columns_order:
                    if col not in existing.columns:
                        existing[col] = ""
                existing = existing[columns_order]
                if existing.empty:
                    combined = new_rows
                else:
                    combined = pd.concat([existing, new_rows], ignore_index=True)
            else:
                combined = new_rows

            self._atomic_write_csv(combined)
            if rotated_archive is not None:
                self._prune_archives()
                print(
                    "🗜️ Log forense rotado: "
                    f"{rotated_archive.name} | activo={len(new_rows):,} filas"
                )
        except Exception as e:
            print(f"⚠️ Error guardando CSV: {e}")

    def _save_to_json(self, result_dto, tag, forensic_data):
        """Genera el reporte de inteligencia JSON necesario para el sistema."""
        temporary = None
        try:
            target = Path(self.json_path)
            target.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "timestamp": datetime.now().isoformat(),
                "version": tag,
                "summary": result_dto,  # El codificador maneja el DTO
                "forensic_details": forensic_data,
            }

            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            with temporary.open("w", encoding="utf-8") as f:
                json.dump(payload, f, cls=SniperJSONEncoder, indent=4)
            os.replace(temporary, target)

            # print(f"✅ Inteligencia JSON generada: {self.json_path}")
        except Exception as e:
            print(f"❌ FALLO CRÍTICO EN SERIALIZACIÓN JSON: {e}")
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def get_summary(self, include_archives=True):
        """Read the active ledger and, by default, retained gzip archives."""
        frames = []
        if include_archives and self.archive_directory.exists():
            stem = Path(self.log_path).stem
            for archive in sorted(
                self.archive_directory.glob(f"{stem}_*.csv.gz")
            ):
                try:
                    frames.append(pd.read_csv(archive, compression="gzip"))
                except Exception:
                    continue
        if os.path.exists(self.log_path):
            frames.append(pd.read_csv(self.log_path))
        if not frames:
            return pd.DataFrame(columns=list(self.COLUMNS_ORDER))
        return pd.concat(frames, ignore_index=True)
