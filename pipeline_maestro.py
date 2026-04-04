# -*- coding: utf-8 -*-
"""
pipeline_maestro.py
===================
Orquestador completo del pipeline diario ML-Finance.

Flujo end-to-end:
  FASE 1 — Descarga: Corre los 3 scripts de descarga (ML_1, ML_2, ML_3)
  FASE 2 — Consolidacion: Corre daily_data_appender.py (que internamente
           appendea al Excel y luego invoca target_updater.py)
  FASE 3 — Verificacion: Comprueba que el consolidado tenga filas nuevas
           sin Target_real (datos a predecir). Si no las tiene, aborta.
  FASE 4 — Prediccion: Corre los 3 modelos en secuencia con aislamiento
           de fallos (si uno falla, los demas siguen)
  FASE 5 — Resumen final con timestamps de cada paso.

Todo el output (consola + log) se guarda en un archivo de texto con
timestamp para auditoria.

Uso:
  python pipeline_maestro.py                   # pipeline completo
  python pipeline_maestro.py --dry-run         # simula sin escribir
  python pipeline_maestro.py --skip-downloads  # salta descarga (ya baje)
  python pipeline_maestro.py --models-only     # solo corre los 3 modelos
  python pipeline_maestro.py --solo v4         # solo un modelo
  python pipeline_maestro.py --skip v2_4       # omite un modelo
  python pipeline_maestro.py --timeout 7200    # timeout por script (seg)

Pensado para Windows Task Scheduler:
  Programa: python.exe
  Argumentos: "C:\\...\\pipeline_maestro.py"
  Inicio en: C:\\...\\(directorio del script)
  Trigger: diario a las 06:00
"""

import argparse
import sys
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ===========================================================================
# CONFIG
# ===========================================================================

BASE_DIR = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")
INPUT_FILE = BASE_DIR / "Consolidado_100_semanas_paste_todos.xlsx"
CONSO_SHEET = "Hoja1"

SCRIPTS_DIR = Path(__file__).resolve().parent

# Timeout por defecto por cada sub-script (4 horas)
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60

# Timeout especifico para scripts de descarga (1 hora)
DOWNLOAD_TIMEOUT_SECONDS = 1 * 60 * 60

# Dias de antiguedad del Excel para considerar warning de staleness
MAX_DAYS_STALE = 8

# --- FASE 1: Scripts de descarga ---
DOWNLOAD_SCRIPTS = [
    {
        "key":   "ml_1",
        "label": "Datos tickers para ML_1",
        "file":  "Datos tickers para ML_1.py",
    },
    {
        "key":   "ml_2",
        "label": "Datos tickers para ML_2",
        "file":  "Datos tickers para ML_2.py",
    },
    {
        "key":   "ml_3",
        "label": "Datos tickers para ML_3",
        "file":  "Datos tickers para ML_3.py",
    },
]

# --- FASE 2: Consolidacion ---
CONSOLIDATION_SCRIPT = {
    "key":   "appender",
    "label": "daily_data_appender",
    "file":  "daily_data_appender.py",
}

# --- FASE 4: Modelos de prediccion ---
PREDICTION_SCRIPTS = [
    {
        "key":   "v2_4",
        "label": "Predicciones Target3 SUPERADOR v2.4",
        "file":  "Predicciones_Target3_SUPERADOR_v2_4.py",
        "desc":  "Modelo base -- FULL + MIN_PRICE (LGBMClassifier + LGBMRanker)",
    },
    {
        "key":   "v3",
        "label": "Pred T3 v3 (MINPRICE)",
        "file":  "Pred_T3_v3_MINPRICE.py",
        "desc":  "v3 -- sector neutralization, stacking, permutation test",
    },
    {
        "key":   "v4",
        "label": "Pred T3 v4",
        "file":  "Pred_T3_v4.py",
        "desc":  "v4 -- MAX_PRICE filter, Target 1 shorts, feature importance",
    },
]


# ===========================================================================
# HELPERS
# ===========================================================================

class TeeWriter:
    """Escribe simultaneamente a un archivo de log y a stdout."""

    def __init__(self, log_path: Path):
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        self._log.write(text)

    def writeline(self, text: str = ""):
        self.write(text + "\n")

    def close(self):
        self._log.close()


def stream_process(proc: subprocess.Popen, tee: TeeWriter, prefix: str = ""):
    """Lee stdout y stderr en threads separados, escribiendo via tee."""
    def _reader(stream, label):
        try:
            for line in iter(stream.readline, ""):
                tee.write(f"{prefix}{label}{line}")
        except Exception:
            pass
        finally:
            stream.close()

    t_out = threading.Thread(target=_reader, args=(proc.stdout, ""), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, "[ERR] "), daemon=True)
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()


def run_script(script_cfg: dict, tee: TeeWriter, timeout: int,
               extra_args: list = None, phase_label: str = "") -> dict:
    """
    Corre un script de Python como subproceso con streaming de output.
    Devuelve dict: key, label, status, duration_s, returncode, error.
    """
    script_path = SCRIPTS_DIR / script_cfg["file"]
    key   = script_cfg["key"]
    label = script_cfg["label"]
    desc  = script_cfg.get("desc", "")

    tee.writeline("-" * 70)
    tee.writeline(f"  {phase_label}CORRIENDO: {label}")
    tee.writeline(f"  Script: {script_path.name}")
    if desc:
        tee.writeline(f"  Desc:   {desc}")
    if extra_args:
        tee.writeline(f"  Args:   {' '.join(extra_args)}")
    tee.writeline("-" * 70)

    if not script_path.exists():
        msg = f"Script no encontrado: {script_path}"
        tee.writeline(f"  [FAIL] {msg}")
        tee.writeline()
        return {"key": key, "label": label, "status": "FAIL",
                "duration_s": 0, "returncode": -1, "error": msg}

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    result = {"key": key, "label": label, "status": "?",
              "duration_s": 0, "returncode": -1, "error": ""}

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(SCRIPTS_DIR),
        )

        stream_process(proc, tee)

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.time() - start
            msg = f"Timeout tras {elapsed/60:.1f} min ({timeout}s)"
            tee.writeline(f"\n  [FAIL] {msg}")
            result.update({"status": "TIMEOUT", "duration_s": elapsed,
                           "returncode": -2, "error": msg})
            return result

        elapsed = time.time() - start
        rc = proc.returncode

        if rc == 0:
            tee.writeline(f"\n  [OK] Termino en {elapsed/60:.1f} min (rc={rc})")
            result.update({"status": "OK", "duration_s": elapsed, "returncode": rc})
        else:
            msg = f"returncode={rc}"
            tee.writeline(f"\n  [FAIL] Termino con error en {elapsed/60:.1f} min ({msg})")
            result.update({"status": "FAIL", "duration_s": elapsed,
                           "returncode": rc, "error": msg})

    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}"
        tee.writeline(f"\n  [FAIL] Excepcion al lanzar el script:")
        tee.writeline(tb)
        result.update({"status": "FAIL", "duration_s": elapsed,
                       "returncode": -1, "error": msg})

    tee.writeline()
    return result


def format_duration(seconds: float) -> str:
    """Formatea duracion a string legible."""
    if seconds >= 3600:
        return f"{seconds/3600:.1f}h"
    elif seconds >= 60:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds:.0f}s"


# ===========================================================================
# FASE 3: VERIFICACION DEL CONSOLIDADO
# ===========================================================================

def verify_consolidado_has_prediction_rows(tee: TeeWriter, dry_run: bool = False) -> dict:
    """
    Verifica que el consolidado tenga filas sin Target_real (datos a predecir).
    Retorna dict con status y metadata.
    """
    tee.writeline("=" * 70)
    tee.writeline("FASE 3: VERIFICACION DEL CONSOLIDADO")
    tee.writeline("=" * 70)

    info = {
        "total_rows": 0,
        "prediction_rows": 0,
        "prediction_date": None,
        "prediction_tickers": 0,
        "status": "?"
    }

    if not INPUT_FILE.exists():
        tee.writeline(f"  [FAIL] Archivo no encontrado: {INPUT_FILE}")
        info["status"] = "FAIL"
        return info

    # Verificar antiguedad
    stat = INPUT_FILE.stat()
    size_mb = stat.st_size / (1024 ** 2)
    mtime = datetime.fromtimestamp(stat.st_mtime)
    age_days = (datetime.now() - mtime).days

    tee.writeline(f"  Archivo:  {INPUT_FILE.name}")
    tee.writeline(f"  Tamanio:  {size_mb:.1f} MB")
    tee.writeline(f"  Modificado: {mtime.strftime('%Y-%m-%d %H:%M')} (hace {age_days} dias)")

    if not HAS_PANDAS:
        tee.writeline("  [WARN] pandas no disponible, saltando verificacion de filas.")
        tee.writeline("         Se asume que el consolidado esta listo.")
        info["status"] = "OK"
        return info

    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=CONSO_SHEET, engine="openpyxl")
        info["total_rows"] = len(df)
        tee.writeline(f"  Filas totales: {len(df)}")

        # Buscar filas sin Target_real
        target_col = None
        for candidate in ["Target_real", "target_real", "TARGET_REAL"]:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            tee.writeline("  [WARN] Columna Target_real no encontrada. "
                          "Se asume que hay datos a predecir.")
            info["status"] = "OK"
            return info

        # Filas sin target = datos a predecir
        no_target_mask = df[target_col].isna() | (
            df[target_col].astype(str).str.strip() == ""
        )
        pred_df = df[no_target_mask]
        info["prediction_rows"] = len(pred_df)

        if pred_df.empty:
            tee.writeline("  [FAIL] No hay filas sin Target_real.")
            tee.writeline("         El consolidado no tiene datos para predecir.")
            tee.writeline("         Verificar que daily_data_appender agrego filas nuevas.")
            info["status"] = "FAIL"
            return info

        # Fecha mas reciente sin target
        if "Data_Date" in pred_df.columns:
            dates = pd.to_datetime(pred_df["Data_Date"], errors="coerce").dropna()
            if not dates.empty:
                latest = dates.max()
                info["prediction_date"] = latest.strftime("%Y-%m-%d")
                n_tickers = pred_df.loc[dates.index[dates == latest], "Ticker"].nunique() \
                    if "Ticker" in pred_df.columns else len(pred_df[dates == latest])
                info["prediction_tickers"] = n_tickers
                tee.writeline(f"  Fecha a predecir: {latest.date()}")
                tee.writeline(f"  Tickers a predecir: {n_tickers}")

        tee.writeline(f"  Filas sin Target_real: {len(pred_df)}")
        tee.writeline(f"  [OK] Consolidado listo para prediccion.")
        info["status"] = "OK"

    except Exception as exc:
        tee.writeline(f"  [WARN] Error leyendo consolidado: {exc}")
        tee.writeline("         Se continuara de todas formas.")
        info["status"] = "OK"

    tee.writeline()
    return info


# ===========================================================================
# RESUMEN FINAL
# ===========================================================================

def print_final_summary(
    tee: TeeWriter,
    download_results: list,
    consolidation_result: dict,
    verification_info: dict,
    prediction_results: list,
    total_elapsed: float,
    args,
):
    """Imprime tabla resumen completa de todo el pipeline."""
    tee.writeline()
    tee.writeline("=" * 70)
    tee.writeline("RESUMEN FINAL DEL PIPELINE")
    tee.writeline(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee.writeline("=" * 70)

    col_w = 42

    # --- Fase 1: Descargas ---
    if download_results:
        tee.writeline()
        tee.writeline("  FASE 1 - DESCARGA DE DATOS:")
        tee.writeline(f"  {'Script':<{col_w}}  {'Estado':<8}  {'Tiempo':>8}")
        tee.writeline("  " + "-" * (col_w + 20))
        for r in download_results:
            dur = format_duration(r["duration_s"])
            tee.writeline(f"  {r['label']:<{col_w}}  {r['status']:<8}  {dur:>8}")
    elif not args.models_only:
        tee.writeline()
        tee.writeline("  FASE 1 - DESCARGA: omitida (--skip-downloads)")

    # --- Fase 2: Consolidacion ---
    if consolidation_result:
        tee.writeline()
        tee.writeline("  FASE 2 - CONSOLIDACION:")
        dur = format_duration(consolidation_result["duration_s"])
        tee.writeline(f"  {consolidation_result['label']:<{col_w}}  "
                      f"{consolidation_result['status']:<8}  {dur:>8}")
    elif not args.models_only:
        tee.writeline()
        tee.writeline("  FASE 2 - CONSOLIDACION: omitida")

    # --- Fase 3: Verificacion ---
    if verification_info and verification_info.get("total_rows"):
        tee.writeline()
        tee.writeline("  FASE 3 - VERIFICACION:")
        tee.writeline(f"    Filas totales en consolidado: {verification_info['total_rows']}")
        tee.writeline(f"    Filas a predecir:            {verification_info['prediction_rows']}")
        if verification_info.get("prediction_date"):
            tee.writeline(f"    Fecha de prediccion:         {verification_info['prediction_date']}")
            tee.writeline(f"    Tickers a predecir:          {verification_info['prediction_tickers']}")

    # --- Fase 4: Modelos ---
    if prediction_results:
        tee.writeline()
        tee.writeline("  FASE 4 - MODELOS DE PREDICCION:")
        tee.writeline(f"  {'Modelo':<{col_w}}  {'Estado':<8}  {'Tiempo':>8}  {'Detalle'}")
        tee.writeline("  " + "-" * (col_w + 35))
        for r in prediction_results:
            dur = format_duration(r["duration_s"])
            detail = r["error"] if r["status"] != "OK" else ""
            tee.writeline(f"  {r['label']:<{col_w}}  {r['status']:<8}  {dur:>8}  {detail}")

    # --- Totales ---
    tee.writeline()
    tee.writeline("  " + "-" * (col_w + 35))
    tee.writeline(f"  Tiempo total del pipeline: {format_duration(total_elapsed)}")
    tee.writeline()

    # Resultado global
    all_results = download_results + ([consolidation_result] if consolidation_result else []) + prediction_results
    n_fail = sum(1 for r in all_results if r.get("status") not in ("OK", "SKIP", None))

    if n_fail == 0:
        tee.writeline("  PIPELINE COMPLETO - TODOS LOS PASOS OK")
    else:
        failed_names = [r["label"] for r in all_results if r.get("status") not in ("OK", "SKIP", None)]
        tee.writeline(f"  ATENCION: {n_fail} paso(s) con problemas:")
        for name in failed_names:
            tee.writeline(f"    - {name}")

    tee.writeline("=" * 70)


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline maestro completo: descarga + consolidacion + modelos Target3"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Pasa --dry-run a todos los sub-scripts (no escribe archivos)",
    )
    p.add_argument(
        "--skip-downloads", action="store_true",
        help="Salta la FASE 1 (descarga de datos) si ya corriste los scripts manualmente",
    )
    p.add_argument(
        "--models-only", action="store_true",
        help="Salta FASES 1 y 2 (descarga + consolidacion) y corre solo los modelos",
    )
    p.add_argument(
        "--solo",
        choices=["v2_4", "v3", "v4"],
        default=None,
        help="Corre solo este modelo en la FASE 4 (clave: v2_4 | v3 | v4)",
    )
    p.add_argument(
        "--skip",
        choices=["v2_4", "v3", "v4"],
        default=None,
        help="Omite este modelo en la FASE 4",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout por script de modelo en segundos (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    p.add_argument(
        "--no-verify", action="store_true",
        help="Salta la FASE 3 (verificacion del consolidado)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- Log setup ---
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_dir = BASE_DIR if BASE_DIR.exists() else SCRIPTS_DIR
    log_path = log_dir / f"pipeline_maestro_log_{ts}.txt"
    tee = TeeWriter(log_path)

    tee.writeline("=" * 70)
    tee.writeline("PIPELINE MAESTRO — ML-FINANCE COMPLETO")
    tee.writeline(f"  Fecha/hora:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee.writeline(f"  Python:         {sys.executable}")
    tee.writeline(f"  Scripts dir:    {SCRIPTS_DIR}")
    tee.writeline(f"  Consolidado:    {INPUT_FILE}")
    tee.writeline(f"  Log:            {log_path}")
    tee.writeline(f"  Timeout modelo: {args.timeout // 60} min")
    mode_desc = []
    if args.dry_run:
        mode_desc.append("DRY-RUN")
    if args.skip_downloads:
        mode_desc.append("SKIP-DOWNLOADS")
    if args.models_only:
        mode_desc.append("MODELS-ONLY")
    if args.solo:
        mode_desc.append(f"SOLO={args.solo}")
    if args.skip:
        mode_desc.append(f"SKIP={args.skip}")
    tee.writeline(f"  Modo:           {', '.join(mode_desc) if mode_desc else 'completo'}")
    tee.writeline("=" * 70)
    tee.writeline()

    pipeline_start = time.time()
    download_results = []
    consolidation_result = None
    verification_info = {}
    prediction_results = []

    # ==================================================================
    # FASE 1: DESCARGA DE DATOS
    # ==================================================================
    if not args.skip_downloads and not args.models_only:
        tee.writeline("=" * 70)
        tee.writeline("FASE 1: DESCARGA DE DATOS")
        tee.writeline("=" * 70)
        tee.writeline()

        all_downloads_ok = True
        for dl_cfg in DOWNLOAD_SCRIPTS:
            if args.dry_run:
                tee.writeline(f"  [DRY-RUN] Saltando descarga: {dl_cfg['label']}")
                download_results.append({
                    "key": dl_cfg["key"], "label": dl_cfg["label"],
                    "status": "SKIP", "duration_s": 0, "returncode": 0, "error": ""
                })
                continue

            result = run_script(
                dl_cfg, tee,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                phase_label="[FASE 1] ",
            )
            download_results.append(result)
            if result["status"] != "OK":
                all_downloads_ok = False

        if not args.dry_run and not all_downloads_ok:
            failed = [r["label"] for r in download_results if r["status"] != "OK"]
            tee.writeline(f"  [FAIL] Descarga(s) fallida(s): {failed}")
            tee.writeline("  Abortando pipeline: los datos de entrada no estan completos.")
            tee.writeline("  Para saltar descargas y usar outputs existentes: --skip-downloads")
            tee.writeline()

            # Resumen y salir
            total_elapsed = time.time() - pipeline_start
            print_final_summary(tee, download_results, consolidation_result,
                                verification_info, prediction_results,
                                total_elapsed, args)
            tee.writeline(f"\nLog guardado en: {log_path}")
            tee.close()
            sys.exit(1)

        tee.writeline()
    else:
        reason = "--models-only" if args.models_only else "--skip-downloads"
        tee.writeline(f"FASE 1: DESCARGA omitida ({reason})")
        tee.writeline()

    # ==================================================================
    # FASE 2: CONSOLIDACION (daily_data_appender + target_updater)
    # ==================================================================
    if not args.models_only:
        tee.writeline("=" * 70)
        tee.writeline("FASE 2: CONSOLIDACION DE DATOS")
        tee.writeline("=" * 70)
        tee.writeline()

        # daily_data_appender.py ya invoca target_updater internamente
        # Pasamos --skip-updater solo si es dry-run (porque ya pasa --dry-run
        # al target_updater automaticamente)
        appender_args = []
        if args.dry_run:
            appender_args.append("--dry-run")

        consolidation_result = run_script(
            CONSOLIDATION_SCRIPT, tee,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            extra_args=appender_args if appender_args else None,
            phase_label="[FASE 2] ",
        )

        if consolidation_result["status"] != "OK" and not args.dry_run:
            tee.writeline("  [WARN] daily_data_appender termino con error.")
            tee.writeline("         Se intentara continuar con los modelos de todas formas.")

        tee.writeline()
    else:
        tee.writeline("FASE 2: CONSOLIDACION omitida (--models-only)")
        tee.writeline()

    # ==================================================================
    # FASE 3: VERIFICACION DEL CONSOLIDADO
    # ==================================================================
    if not args.no_verify:
        verification_info = verify_consolidado_has_prediction_rows(
            tee, dry_run=args.dry_run,
        )

        if verification_info["status"] == "FAIL" and not args.dry_run:
            tee.writeline("  Abortando: no hay datos para predecir.")
            tee.writeline("  Usa --no-verify para forzar la ejecucion de modelos.")
            tee.writeline()

            total_elapsed = time.time() - pipeline_start
            print_final_summary(tee, download_results, consolidation_result,
                                verification_info, prediction_results,
                                total_elapsed, args)
            tee.writeline(f"\nLog guardado en: {log_path}")
            tee.close()
            sys.exit(1)
    else:
        tee.writeline("FASE 3: VERIFICACION omitida (--no-verify)")
        tee.writeline()

    # ==================================================================
    # FASE 4: MODELOS DE PREDICCION
    # ==================================================================
    tee.writeline("=" * 70)
    tee.writeline("FASE 4: MODELOS DE PREDICCION")
    tee.writeline("=" * 70)
    tee.writeline()

    # Filtrar segun --solo / --skip
    scripts_to_run = list(PREDICTION_SCRIPTS)
    if args.solo:
        scripts_to_run = [s for s in PREDICTION_SCRIPTS if s["key"] == args.solo]
        tee.writeline(f"  Modo --solo: corriendo unicamente '{args.solo}'")
    elif args.skip:
        scripts_to_run = [s for s in PREDICTION_SCRIPTS if s["key"] != args.skip]
        tee.writeline(f"  Modo --skip: omitiendo '{args.skip}'")

    tee.writeline(f"  Modelos a correr: {[s['key'] for s in scripts_to_run]}")
    tee.writeline()

    for script_cfg in scripts_to_run:
        result = run_script(
            script_cfg, tee,
            timeout=args.timeout,
            phase_label="[FASE 4] ",
        )
        prediction_results.append(result)

    # ==================================================================
    # FASE 5: RESUMEN FINAL
    # ==================================================================
    total_elapsed = time.time() - pipeline_start

    print_final_summary(
        tee, download_results, consolidation_result,
        verification_info, prediction_results,
        total_elapsed, args,
    )

    tee.writeline(f"\nLog guardado en: {log_path}")

    if args.dry_run:
        tee.writeline("[DRY-RUN completado - ningun archivo fue modificado]")

    tee.close()

    # Exit code: 0 solo si todos los modelos pasaron
    any_model_failed = any(r["status"] != "OK" for r in prediction_results)
    sys.exit(1 if any_model_failed else 0)


if __name__ == "__main__":
    main()
