# -*- coding: utf-8 -*-
"""
daily_data_appender.py
======================
Automatiza el proceso diario de consolidar datos nuevos de los 3 scripts
de descarga y preparar el archivo de input para los modelos de prediccion.

Flujo:
  1. (Opcional) Ejecuta los 3 scripts de descarga.
  2. Lee los 3 outputs y los concatena, deduplicando por (Ticker, Data_Date).
  3. Verifica alineacion de columnas con el consolidado existente.
  4. Appendea solo filas nuevas (no duplicadas por Ticker+Data_Date).
  5. Limpia filas de prediccion obsoletas (sin Target_real, fecha != ultima).
  6. Guarda con backup automatico.
  7. Invoca target_updater.py para completar targets donde ya hay precio.

Uso:
    python daily_data_appender.py                 # lee outputs existentes
    python daily_data_appender.py --run-downloads  # ejecuta los 3 scripts primero
    python daily_data_appender.py --dry-run        # simula sin escribir nada
    python daily_data_appender.py --skip-updater   # no corre target_updater al final
    python daily_data_appender.py --no-clean       # no elimina filas obsoletas

Idempotente: si se corre 2 veces el mismo dia, no duplica datos.
"""

import argparse
import shutil
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ===========================================================================
# CONFIG
# ===========================================================================

SCRIPTS_DIR = Path(__file__).resolve().parent

# Outputs de los 3 scripts de descarga (paths relativos a SCRIPTS_DIR)
ML_OUTPUT_FILES = [
    SCRIPTS_DIR / "Consolidado_1_semanas_todos los tickers_1.xlsx",
    SCRIPTS_DIR / "Consolidado_1_semanas_todos los tickers_2.xlsx",
    SCRIPTS_DIR / "Consolidado_1_semanas_todos los tickers_3.xlsx",
]

ML_SCRIPTS = [
    SCRIPTS_DIR / "Datos tickers para ML_1.py",
    SCRIPTS_DIR / "Datos tickers para ML_2.py",
    SCRIPTS_DIR / "Datos tickers para ML_3.py",
]

ML_SHEET = "Consolidado_5s"

# Archivo consolidado de destino
BASE_DIR    = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")
CONSO_FILE  = BASE_DIR / "Consolidado_100_semanas_paste_todos.xlsx"
CONSO_SHEET = "Hoja1"

# Columnas de identificacion unica para deduplicacion
DEDUP_KEYS = ["Ticker", "Data_Date"]

# Columnas 55-63 que gestiona target_updater (se dejan vacias en filas nuevas)
TARGET_COLS = [
    "Dia_una_semana", "Clave", "Precio", "Precio_una_semana",
    "Ret_Una_sem", "Min", "Max", "Target_real", "SMA_50_porc",
]

# Columna que indica si una fila ya tiene target calculado
TARGET_REAL_COL = "Target_real"

# Timeout por script de descarga (segundos)
DOWNLOAD_TIMEOUT = 3600  # 1 hora


# ===========================================================================
# UTILIDADES
# ===========================================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def normalize_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convierte columna de fecha a datetime64 sin timezone."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.tz_localize(None)
    return df


def backup_file(fpath: Path) -> Path:
    """Crea un backup del archivo antes de sobreescribirlo."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = fpath.with_name(fpath.stem + f"_backup_{timestamp}" + fpath.suffix)
    shutil.copy2(fpath, backup_path)
    return backup_path


# ===========================================================================
# PASO 1: DESCARGA
# ===========================================================================

def run_download_script(script_path: Path, dry_run: bool = False) -> bool:
    """Ejecuta un script de descarga. Retorna True si termino sin error."""
    label = script_path.name
    if dry_run:
        log(f"  [DRY-RUN] Saltando ejecucion de: {label}")
        return True
    if not script_path.exists():
        log(f"  [ERROR] Script no encontrado: {label}")
        return False

    log(f"  Ejecutando: {label} ...")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPTS_DIR),
            timeout=DOWNLOAD_TIMEOUT,
        )
        if result.returncode == 0:
            log(f"  [OK] {label} termino exitosamente.")
            return True
        else:
            log(f"  [WARN] {label} termino con codigo {result.returncode}.")
            return False
    except subprocess.TimeoutExpired:
        log(f"  [ERROR] {label} supero el timeout ({DOWNLOAD_TIMEOUT}s).")
        return False
    except Exception as exc:
        log(f"  [ERROR] {label}: {exc}")
        return False


# ===========================================================================
# PASO 2: LEER Y CONCATENAR OUTPUTS
# ===========================================================================

def load_ml_output(path: Path) -> pd.DataFrame:
    """Lee el Excel output de un script de descarga."""
    if not path.exists():
        log(f"  [WARN] No encontrado: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_excel(path, sheet_name=ML_SHEET, engine="openpyxl")
        log(f"  Leido: {path.name} -> {len(df)} filas, {df.shape[1]} cols")
        return df
    except Exception as exc:
        log(f"  [ERROR] Leyendo {path.name}: {exc}")
        return pd.DataFrame()


def load_and_combine_ml_outputs() -> pd.DataFrame:
    """Lee los 3 outputs, concatena y deduplica por (Ticker, Data_Date)."""
    dfs = []
    for path in ML_OUTPUT_FILES:
        df = load_ml_output(path)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        log("[ERROR] No se pudo leer ningun output de los scripts de descarga.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    log(f"  Total antes de deduplicar: {len(combined)} filas de {len(dfs)} archivos")

    combined = normalize_date_col(combined, "Data_Date")

    # Deduplicar — SPY viene repetido en los 3 outputs
    combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="first").reset_index(drop=True)
    log(
        f"  Tras deduplicar por {DEDUP_KEYS}: {len(combined)} filas "
        f"| {combined['Ticker'].nunique()} tickers unicos"
    )

    if "Data_Date" in combined.columns and combined["Data_Date"].notna().any():
        latest = combined["Data_Date"].max()
        log(f"  Fecha de datos nuevos: {latest.date()}")

    return combined


# ===========================================================================
# PASO 3: VERIFICAR COLUMNAS
# ===========================================================================

def check_column_alignment(new_df: pd.DataFrame, conso_df: pd.DataFrame) -> None:
    """Imprime advertencias sobre diferencias de columnas (no aborta)."""
    # Comparar solo columnas que no son de target (esas se dejan vacias a proposito)
    conso_base = [c for c in conso_df.columns if c not in TARGET_COLS]
    new_base   = [c for c in new_df.columns   if c not in TARGET_COLS]

    missing_in_new = set(conso_base) - set(new_base)
    extra_in_new   = set(new_base)   - set(conso_base)

    if missing_in_new:
        log(
            f"  [WARN] Columnas presentes en consolidado pero AUSENTES en nuevos datos "
            f"(se rellenaran con NaN): {sorted(missing_in_new)}"
        )
    if extra_in_new:
        log(
            f"  [INFO] Columnas extras en nuevos datos "
            f"(se incluiran si hay lugar en el schema): {sorted(extra_in_new)}"
        )
    if not missing_in_new and not extra_in_new:
        log("  [OK] Columnas alineadas correctamente.")


# ===========================================================================
# PASO 4 & 5: FILTRAR FILAS NUEVAS Y APPENDEAR
# ===========================================================================

def get_existing_keys(conso_df: pd.DataFrame) -> set:
    """Retorna set de tuplas (Ticker, Data_Date) ya existentes en el consolidado."""
    tmp = conso_df[["Ticker", "Data_Date"]].copy()
    tmp = normalize_date_col(tmp, "Data_Date")
    keys = set()
    for _, row in tmp.iterrows():
        ticker = str(row["Ticker"]).strip() if pd.notna(row["Ticker"]) else ""
        date   = row["Data_Date"]
        if ticker and pd.notna(date):
            keys.add((ticker, date))
    return keys


def filter_new_rows(new_df: pd.DataFrame, existing_keys: set) -> pd.DataFrame:
    """Retorna solo las filas de new_df que NO esten en existing_keys."""
    tmp = normalize_date_col(new_df.copy(), "Data_Date")
    mask = tmp.apply(
        lambda row: (str(row.get("Ticker", "")).strip(), row.get("Data_Date"))
        not in existing_keys,
        axis=1,
    )
    return new_df[mask].copy()


def align_to_consolidado(new_df: pd.DataFrame, conso_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reordena new_df para que coincida exactamente con las columnas del consolidado.
    Columnas faltantes -> NaN. Columnas de target -> siempre NaN (las llena target_updater).
    """
    aligned = pd.DataFrame(index=new_df.index, columns=conso_df.columns)
    for col in conso_df.columns:
        if col in TARGET_COLS:
            aligned[col] = np.nan
        elif col in new_df.columns:
            aligned[col] = new_df[col].values
        else:
            aligned[col] = np.nan
    return aligned


# ===========================================================================
# PASO 6: LIMPIAR FILAS DE PREDICCION OBSOLETAS
# ===========================================================================

def clean_stale_prediction_rows(conso_df: pd.DataFrame) -> tuple:
    """
    Elimina filas sin Target_real cuya Data_Date NO es la fecha mas reciente
    sin target. Esas son datos de semanas anteriores que nunca se predijeron
    correctamente (p.ej. una descarga parcial de otra semana).

    Retorna (df_limpio, n_eliminadas).
    """
    if TARGET_REAL_COL not in conso_df.columns:
        return conso_df, 0

    df = conso_df.copy()
    df = normalize_date_col(df, "Data_Date")

    # Filas sin Target_real (NaN o cadena vacia)
    no_target_mask = df[TARGET_REAL_COL].isna() | (
        df[TARGET_REAL_COL].astype(str).str.strip() == ""
    )

    no_target_df = df[no_target_mask]
    if no_target_df.empty:
        return conso_df, 0

    # Fecha mas reciente entre las filas sin target
    latest_pred_date = no_target_df["Data_Date"].max()

    # Obsoletas: sin target Y fecha distinta a la mas reciente
    stale_mask = no_target_mask & (df["Data_Date"] != latest_pred_date)
    n_stale = int(stale_mask.sum())

    if n_stale > 0:
        df = df[~stale_mask].reset_index(drop=True)
        log(f"  Fechas eliminadas: {sorted(df.loc[~stale_mask if False else pd.Series(stale_mask, index=df.index), 'Data_Date'].unique()) if False else 'ver log'}")

    return df, n_stale


# ===========================================================================
# PASO 8: TARGET_UPDATER
# ===========================================================================

def run_target_updater(dry_run: bool = False, skip: bool = False) -> bool:
    """Invoca target_updater.py como subprocess."""
    if skip:
        log("  [SKIP] target_updater.py omitido (--skip-updater).")
        return True

    updater_path = SCRIPTS_DIR / "target_updater.py"
    if not updater_path.exists():
        log(f"  [WARN] target_updater.py no encontrado en {SCRIPTS_DIR}")
        return False

    cmd = [sys.executable, str(updater_path)]
    if dry_run:
        cmd.append("--dry-run")

    log(f"  Ejecutando target_updater.py{' (--dry-run)' if dry_run else ''} ...")
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            log("  [OK] target_updater.py termino exitosamente.")
            return True
        else:
            log(f"  [WARN] target_updater.py termino con codigo {result.returncode}.")
            return False
    except Exception as exc:
        log(f"  [ERROR] target_updater.py: {exc}")
        return False


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolida outputs de los 3 scripts ML y appendea al consolidado semanal."
    )
    parser.add_argument(
        "--run-downloads", action="store_true",
        help="Ejecuta los 3 scripts de descarga antes de leer sus outputs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simula todo el proceso sin escribir ningun archivo.",
    )
    parser.add_argument(
        "--skip-updater", action="store_true",
        help="No ejecuta target_updater.py al final.",
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="No elimina filas de prediccion obsoletas.",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  daily_data_appender.py")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print("  MODO DRY-RUN: ningun archivo sera modificado")
    print("=" * 65)

    n_new    = 0
    n_stale  = 0

    # ------------------------------------------------------------------
    # PASO 1: Ejecutar scripts de descarga (opcional)
    # ------------------------------------------------------------------
    if args.run_downloads:
        log("PASO 1: Ejecutando scripts de descarga ...")
        download_results = {}
        for script, out_file in zip(ML_SCRIPTS, ML_OUTPUT_FILES):
            ok = run_download_script(script, dry_run=args.dry_run)
            download_results[script.name] = ok
        failed = [name for name, ok in download_results.items() if not ok]
        if failed:
            log(f"  [WARN] Scripts con error: {failed}. Se intentara leer sus outputs de todas formas.")
    else:
        log("PASO 1: --run-downloads no especificado. Leyendo outputs existentes.")

    # ------------------------------------------------------------------
    # PASO 2: Leer y concatenar outputs
    # ------------------------------------------------------------------
    log("PASO 2: Leyendo y concatenando outputs de los 3 scripts ...")
    combined = load_and_combine_ml_outputs()

    # ------------------------------------------------------------------
    # PASO 3: Cargar consolidado y verificar columnas
    # ------------------------------------------------------------------
    log("PASO 3: Cargando consolidado y verificando columnas ...")
    if not CONSO_FILE.exists():
        log(f"[ERROR] Consolidado no encontrado: {CONSO_FILE}")
        sys.exit(1)

    conso = pd.read_excel(CONSO_FILE, sheet_name=CONSO_SHEET, engine="openpyxl")

    # Sanear columnas duplicadas del consolidado (artefacto de pegado manual)
    if conso.columns.duplicated().any():
        dup_names = conso.columns[conso.columns.duplicated(keep=False)].unique().tolist()
        log(f"  [WARN] Columnas duplicadas en consolidado (se elimina la segunda ocurrencia): {dup_names}")
        conso = conso.loc[:, ~conso.columns.duplicated(keep="first")]

    conso = normalize_date_col(conso, "Data_Date")
    log(f"  Consolidado cargado: {len(conso)} filas, {conso.shape[1]} cols")

    check_column_alignment(combined, conso)

    # ------------------------------------------------------------------
    # PASOS 4 & 5: Filtrar y appendear solo filas nuevas
    # ------------------------------------------------------------------
    log("PASO 4: Identificando filas nuevas ...")
    existing_keys = get_existing_keys(conso)
    log(f"  Pares (Ticker, Data_Date) ya en consolidado: {len(existing_keys)}")

    new_rows = filter_new_rows(combined, existing_keys)
    n_new = len(new_rows)
    log(f"  Filas nuevas a appendear: {n_new}")

    if n_new > 0:
        log("PASO 5: Appendeando filas nuevas ...")
        new_rows_aligned = align_to_consolidado(new_rows, conso)
        conso = pd.concat([conso, new_rows_aligned], ignore_index=True)
        log(f"  Consolidado actualizado: {len(conso)} filas (+{n_new})")
    else:
        log("PASO 5: No hay filas nuevas. El consolidado ya esta actualizado para hoy.")

    # ------------------------------------------------------------------
    # PASO 6: Limpiar filas de prediccion obsoletas
    # ------------------------------------------------------------------
    if not args.no_clean:
        log("PASO 6: Limpiando filas de prediccion obsoletas ...")
        conso, n_stale = clean_stale_prediction_rows(conso)
        if n_stale > 0:
            log(f"  Eliminadas {n_stale} filas de semanas anteriores sin Target_real.")
        else:
            log("  No se encontraron filas obsoletas.")
    else:
        log("PASO 6: --no-clean especificado. Saltando limpieza.")

    # ------------------------------------------------------------------
    # PASO 7: Guardar con backup
    # ------------------------------------------------------------------
    if not args.dry_run:
        log("PASO 7: Guardando consolidado ...")
        if CONSO_FILE.exists():
            backup_path = backup_file(CONSO_FILE)
            log(f"  Backup creado: {backup_path.name}")
        try:
            with pd.ExcelWriter(CONSO_FILE, engine="openpyxl") as writer:
                conso.to_excel(writer, sheet_name=CONSO_SHEET, index=False)
            log(f"  [OK] Guardado: {CONSO_FILE.name} | {len(conso)} filas")
        except Exception as exc:
            log(f"  [ERROR] No se pudo guardar el consolidado: {exc}")
            sys.exit(1)
    else:
        log("PASO 7: [DRY-RUN] Consolidado NO guardado.")

    # ------------------------------------------------------------------
    # PASO 8: Invocar target_updater.py
    # ------------------------------------------------------------------
    log("PASO 8: Invocando target_updater.py ...")
    run_target_updater(dry_run=args.dry_run, skip=args.skip_updater)

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  RESUMEN FINAL")
    print("=" * 65)
    print(f"  Filas nuevas appendeadas       : {n_new}")
    if not args.no_clean:
        print(f"  Filas obsoletas eliminadas     : {n_stale}")
    print(f"  Total filas en consolidado     : {len(conso)}")

    # Filas listas para predecir (sin Target_real, solo la fecha mas reciente)
    if TARGET_REAL_COL in conso.columns:
        conso_tmp = normalize_date_col(conso.copy(), "Data_Date")
        no_target_mask = conso_tmp[TARGET_REAL_COL].isna() | (
            conso_tmp[TARGET_REAL_COL].astype(str).str.strip() == ""
        )
        no_target_df = conso_tmp[no_target_mask]
        if not no_target_df.empty:
            print(f"  Filas para prediccion (sin Target_real):")
            for date, group in no_target_df.groupby("Data_Date", sort=True):
                date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "N/A"
                print(f"    {date_str}: {len(group)} tickers")
        else:
            print("  Sin filas pendientes de prediccion.")

    print("=" * 65)
    if args.dry_run:
        print("  [DRY-RUN completado - ningun archivo fue modificado]")
    else:
        print("  Proceso completado exitosamente.")
    print("=" * 65)


if __name__ == "__main__":
    main()
