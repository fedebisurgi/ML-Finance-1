# -*- coding: utf-8 -*-
"""
pipeline_maestro.py
===================
Orquestador semanal de los 3 modelos de prediccion Target3.

Flujo:
  1. Verifica que el archivo de datos consolidado exista y sea reciente
  2. Corre en secuencia:
       - Predicciones_Target3_SUPERADOR_v2_4.py
       - Pred_T3_v3_MINPRICE.py
       - Pred_T3_v4.py
  3. Si un modelo falla, registra el error y continua con el siguiente
  4. Guarda un log completo en BASE_DIR con todo el output de cada script
  5. Imprime un resumen final: cuales terminaron OK, cuales fallaron

Uso:
  python pipeline_maestro.py
  python pipeline_maestro.py --solo v4          # corre solo un modelo
  python pipeline_maestro.py --skip v2_4        # omite uno
  python pipeline_maestro.py --timeout 14400    # timeout en segundos (default: 4h)

El script vive en el mismo directorio que los 3 modelos.
El log se guarda en BASE_DIR (mismo dir que los Excels de output).

Contexto del pipeline completo (para referencia):
  - Descarga semanal: Datos tickers para ML_1/2/3.py
  - Consolidado fuente: Consolidado_100_semanas_paste_todos.xlsx
  - Directorio de trabajo: C:\\Users\\GOFOYCOP_01\\00.Redes neuronales\\
                           04.Descarga anual\\03.consolidado
"""

import argparse
import sys
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

# ===========================================================================
# CONFIG
# ===========================================================================

# Directorio donde viven los Excels de input/output (igual que en los modelos)
BASE_DIR = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")

# Archivo de datos que deben leer los 3 modelos
INPUT_FILE = BASE_DIR / "Consolidado_100_semanas_paste_todos.xlsx"

# Minimo de dias desde la ultima modificacion del Excel para avisar que puede estar desactualizado
MAX_DAYS_STALE = 8  # si tiene mas de 8 dias sin tocar, se muestra advertencia

# Timeout por script en segundos (default 4 horas; eston modelos pueden tardar)
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60

# Directorio donde vive este script (y los 3 modelos)
SCRIPTS_DIR = Path(__file__).resolve().parent

# Los 3 scripts en orden de ejecucion
PIPELINE_SCRIPTS = [
    {
        "key":    "v2_4",
        "label":  "Predicciones Target3 SUPERADOR v2.4",
        "file":   "Predicciones_Target3_SUPERADOR_v2_4.py",
        "desc":   "Modelo base — FULL + MIN_PRICE (LGBMClassifier + LGBMRanker)",
    },
    {
        "key":    "v3",
        "label":  "Pred T3 v3 (MINPRICE)",
        "file":   "Pred_T3_v3_MINPRICE.py",
        "desc":   "v3 — sector neutralization, stacking, permutation test",
    },
    {
        "key":    "v4",
        "label":  "Pred T3 v4",
        "file":   "Pred_T3_v4.py",
        "desc":   "v4 — MAX_PRICE filter, Target 1 shorts, feature importance",
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
    """
    Lee stdout y stderr del proceso en tiempo real y los escribe via tee.
    Corre en dos threads separados para no bloquear ninguno de los dos streams.
    """
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


def check_input_file(tee: TeeWriter) -> bool:
    """Verifica que el archivo de datos exista y no este muy desactualizado."""
    tee.writeline("=" * 70)
    tee.writeline("VERIFICACION DEL ARCHIVO DE ENTRADA")
    tee.writeline("=" * 70)
    tee.writeline(f"  Path:   {INPUT_FILE}")

    if not INPUT_FILE.exists():
        tee.writeline(f"  FAIL  El archivo de datos NO existe.")
        tee.writeline(f"  Verificar que el consolidado semanal fue generado y pegado.")
        return False

    stat    = INPUT_FILE.stat()
    size_mb = stat.st_size / (1024 ** 2)
    mtime   = datetime.fromtimestamp(stat.st_mtime)
    age_days = (datetime.now() - mtime).days

    tee.writeline(f"  Tamaño: {size_mb:.1f} MB")
    tee.writeline(f"  Ultima modificacion: {mtime.strftime('%Y-%m-%d %H:%M')}  "
                  f"(hace {age_days} dias)")

    if age_days > MAX_DAYS_STALE:
        tee.writeline(f"  WARN  El archivo tiene {age_days} dias sin modificar. "
                      f"¿Olvidaste pegar los datos de esta semana?")
    else:
        tee.writeline(f"  PASS  Archivo presente y reciente.")

    tee.writeline()
    return True


def run_script(script_cfg: dict, tee: TeeWriter, timeout: int) -> dict:
    """
    Corre un script de Python como subproceso.
    Devuelve un dict con: key, label, status, duration_s, returncode, error.
    """
    script_path = SCRIPTS_DIR / script_cfg["file"]
    key         = script_cfg["key"]
    label       = script_cfg["label"]

    tee.writeline("=" * 70)
    tee.writeline(f"CORRIENDO: {label}")
    tee.writeline(f"  Script: {script_path}")
    tee.writeline(f"  Desc:   {script_cfg['desc']}")
    tee.writeline("=" * 70)

    if not script_path.exists():
        msg = f"Script no encontrado: {script_path}"
        tee.writeline(f"  FAIL  {msg}")
        tee.writeline()
        return {"key": key, "label": label, "status": "FAIL",
                "duration_s": 0, "returncode": -1, "error": msg}

    start = time.time()
    result = {"key": key, "label": label, "status": "?",
              "duration_s": 0, "returncode": -1, "error": ""}

    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(SCRIPTS_DIR),
        )

        # Streams en tiempo real
        stream_process(proc, tee)

        # Esperar con timeout
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            elapsed = time.time() - start
            msg = f"Timeout tras {elapsed/60:.1f} minutos ({timeout}s)"
            tee.writeline(f"\n  FAIL  {msg}")
            result.update({"status": "TIMEOUT", "duration_s": elapsed,
                           "returncode": -2, "error": msg})
            return result

        elapsed = time.time() - start
        rc = proc.returncode

        if rc == 0:
            tee.writeline(f"\n  PASS  Terminó OK en {elapsed/60:.1f} min "
                          f"(returncode={rc})")
            result.update({"status": "PASS", "duration_s": elapsed, "returncode": rc})
        else:
            msg = f"returncode={rc}"
            tee.writeline(f"\n  FAIL  Terminó con error en {elapsed/60:.1f} min "
                          f"({msg})")
            result.update({"status": "FAIL", "duration_s": elapsed,
                           "returncode": rc, "error": msg})

    except Exception as e:
        elapsed = time.time() - start
        tb = traceback.format_exc()
        msg = f"{type(e).__name__}: {e}"
        tee.writeline(f"\n  FAIL  Excepcion al lanzar el script:")
        tee.writeline(tb)
        result.update({"status": "FAIL", "duration_s": elapsed,
                       "returncode": -1, "error": msg})

    tee.writeline()
    return result


def print_summary(results: list[dict], tee: TeeWriter, total_elapsed: float):
    """Imprime la tabla de resumen final."""
    tee.writeline()
    tee.writeline("=" * 70)
    tee.writeline("RESUMEN FINAL DEL PIPELINE")
    tee.writeline("=" * 70)

    col_w = 38
    tee.writeline(
        f"  {'Modelo':<{col_w}}  {'Estado':<8}  {'Tiempo':>8}  {'Detalle'}"
    )
    tee.writeline("  " + "-" * (col_w + 30))

    all_ok = True
    for r in results:
        dur = f"{r['duration_s']/60:.1f} min" if r['duration_s'] >= 60 else \
              f"{r['duration_s']:.0f} s"
        detail = r["error"] if r["status"] != "PASS" else ""
        tee.writeline(
            f"  {r['label']:<{col_w}}  {r['status']:<8}  {dur:>8}  {detail}"
        )
        if r["status"] != "PASS":
            all_ok = False

    tee.writeline("  " + "-" * (col_w + 30))
    tee.writeline(f"  Tiempo total: {total_elapsed/60:.1f} minutos")
    tee.writeline()

    if all_ok:
        tee.writeline("  TODOS LOS MODELOS TERMINARON OK.")
    else:
        failed = [r["label"] for r in results if r["status"] != "PASS"]
        tee.writeline(f"  ATENCION: {len(failed)} modelo(s) con problemas:")
        for f in failed:
            tee.writeline(f"    - {f}")

    tee.writeline("=" * 70)


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline maestro de modelos Target3")
    p.add_argument(
        "--solo",
        choices=["v2_4", "v3", "v4"],
        default=None,
        help="Corre solo este modelo (clave: v2_4 | v3 | v4)",
    )
    p.add_argument(
        "--skip",
        choices=["v2_4", "v3", "v4"],
        default=None,
        help="Omite este modelo",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout por script en segundos (default: {DEFAULT_TIMEOUT_SECONDS})",
    )
    p.add_argument(
        "--no-check-input",
        action="store_true",
        help="Saltea la verificacion del archivo de datos",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Configurar log
    ts       = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = BASE_DIR / f"pipeline_maestro_log_{ts}.txt"

    # Si BASE_DIR no existe (ej. en entorno de dev), loguear en el dir del script
    if not BASE_DIR.exists():
        log_path = SCRIPTS_DIR / f"pipeline_maestro_log_{ts}.txt"

    tee = TeeWriter(log_path)

    tee.writeline("=" * 70)
    tee.writeline("PIPELINE MAESTRO — MODELOS TARGET3")
    tee.writeline(f"  Fecha/hora:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee.writeline(f"  Python:      {sys.executable}")
    tee.writeline(f"  Scripts dir: {SCRIPTS_DIR}")
    tee.writeline(f"  Log:         {log_path}")
    tee.writeline(f"  Timeout:     {args.timeout // 60} min por script")
    tee.writeline("=" * 70)
    tee.writeline()

    # Verificar archivo de entrada
    if not args.no_check_input:
        data_ok = check_input_file(tee)
        if not data_ok:
            tee.writeline("Abortando. Usá --no-check-input para forzar la ejecución.")
            tee.close()
            sys.exit(1)

    # Filtrar scripts según --solo / --skip
    scripts_to_run = list(PIPELINE_SCRIPTS)
    if args.solo:
        scripts_to_run = [s for s in PIPELINE_SCRIPTS if s["key"] == args.solo]
        tee.writeline(f"  Modo --solo: corriendo únicamente '{args.solo}'")
    elif args.skip:
        scripts_to_run = [s for s in PIPELINE_SCRIPTS if s["key"] != args.skip]
        tee.writeline(f"  Modo --skip: omitiendo '{args.skip}'")

    tee.writeline(f"  Scripts a correr: {[s['key'] for s in scripts_to_run]}")
    tee.writeline()

    # Correr cada script
    pipeline_start = time.time()
    results = []

    for script_cfg in scripts_to_run:
        result = run_script(script_cfg, tee, timeout=args.timeout)
        results.append(result)

    total_elapsed = time.time() - pipeline_start

    # Resumen
    print_summary(results, tee, total_elapsed)
    tee.writeline(f"\nLog guardado en: {log_path}")
    tee.close()

    # Exit code: 0 solo si todos pasaron
    any_failed = any(r["status"] != "PASS" for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
