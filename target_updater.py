# -*- coding: utf-8 -*-
"""
target_updater.py
=================
Calcula automaticamente las columnas 55-63 del consolidado semanal
(Dia_una_semana, Clave, Precio, Precio_una_semana, Ret_Una_sem,
Min, Max, Target_real, SMA_50_porc) para las filas que ya tienen
precio futuro disponible pero todavia no tienen Target_real calculado.

Las filas cuyo "Dia una semana" aun no llego (futuro) se dejan intactas:
esas son las que los modelos van a predecir esta semana.

Uso:
    python target_updater.py
    python target_updater.py --dry-run      # muestra cuantas filas actualiza sin escribir
    python target_updater.py --backfill     # reprocesa TODAS las filas sin target (util la primera vez)

Logica de Min/Max (segun especificacion):
    SPY_ret_semanal = (SPY_close_en_dia_una_semana / SPY_close_en_Data_Date) - 1
    std_30          = desv.std movil de 30 ruedas de retornos DIARIOS del SPY,
                      calculada en Data_Date
    Min = SPY_ret_semanal - 1.5 * std_30
    Max = SPY_ret_semanal + 1.5 * std_30
    Target_real = 1 si Ret < Min | 3 si Ret > Max | 2 si esta en el medio
"""

import argparse
import sys
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

# ===========================================================================
# CONFIG
# ===========================================================================

BASE_DIR  = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")
FILE_NAME = "Consolidado_100_semanas_paste_todos.xlsx"
FILE_PATH = BASE_DIR / FILE_NAME
SHEET     = "Hoja1"

# Multiplier para el rango target (1.5 desvios)
STD_MULT = 1.5
# Ventana rolling para el desvio estandar del SPY (en ruedas / dias de trading)
STD_WINDOW = 30
# Dias hacia adelante para el precio futuro (7 dias calendario)
FWD_DAYS_CAL = 7
# Anos de SPY diario a descargar para tener suficiente historia para std_30
SPY_HISTORY_YEARS = 3
# Maximo de tickers por batch de descarga yfinance
BATCH_SIZE = 200

# Mapeo flexible de nombres de columna -> nombre canonico interno
# El script busca cada columna por los alias en orden; si no la encuentra la crea.
COL_ALIASES = {
    "Ticker":           ["Ticker", "ticker", "TICKER", "Symbol"],
    "Data_Date":        ["Data_Date", "data_date", "Date", "Fecha"],
    "Close":            ["Close", "close", "Precio_cierre", "Adj Close"],
    "SMA_50":           ["SMA_50", "sma_50", "SMA50"],
    # Columnas 55-63 a calcular
    "Dia_una_semana":   ["Dia una semana", "Dia_una_semana", "dia_una_semana"],
    "Clave":            ["Clave", "clave", "Key"],
    "Precio":           ["Precio", "precio"],                       # col 57 = copia de Close
    "Precio_una_semana":["Precio una semana.1", "Precio_una_semana",
                         "Precio una semana", "price_fwd"],
    "Ret_Una_sem":      ["Ret. Una sem.1", "Ret_Una_sem", "Ret. Una sem",
                         "ret_una_sem", "Retorno semanal"],
    "Min":              ["Min.1", "Min", "min"],
    "Max":              ["Max.1", "Max", "max"],
    "Target_real":      ["Target_real", "target_real", "Target Real"],
    "SMA_50_porc":      ["SMA_50_porc", "sma_50_porc", "(Close-SMA50)/SMA50"],
}

# ===========================================================================
# HELPERS
# ===========================================================================

def find_col(df: pd.DataFrame, key: str) -> str | None:
    """Devuelve el nombre real de la columna en df segun los aliases, o None."""
    for alias in COL_ALIASES.get(key, [key]):
        if alias in df.columns:
            return alias
    return None


def ensure_col(df: pd.DataFrame, key: str) -> str:
    """Devuelve el nombre de la columna; la crea vacia si no existe."""
    col = find_col(df, key)
    if col is None:
        col = COL_ALIASES[key][0]   # usa el nombre preferido
        df[col] = np.nan
        print(f"  [INFO] Columna '{col}' no encontrada, creada como nueva.")
    return col


def to_excel_serial(date) -> int:
    """Convierte una fecha a serial Excel (dias desde 1899-12-30)."""
    epoch = pd.Timestamp("1899-12-30")
    return int((pd.Timestamp(date) - epoch).days)


def next_trading_day(prices_series: pd.Series, target_date: pd.Timestamp) -> float:
    """
    Devuelve el primer precio disponible en prices_series cuya fecha
    sea >= target_date. Retorna NaN si no hay datos suficientes.
    """
    candidates = prices_series[prices_series.index >= target_date]
    if candidates.empty:
        return float("nan")
    return float(candidates.iloc[0])


def download_spy_daily(years: int = SPY_HISTORY_YEARS) -> pd.DataFrame:
    """Descarga precios diarios del SPY para calcular std movil."""
    start = (datetime.today() - timedelta(days=int(years * 365) + 30)).strftime("%Y-%m-%d")
    end   = (datetime.today() + timedelta(days=2)).strftime("%Y-%m-%d")
    print(f"  Descargando SPY diario {start} -> {end}...")
    raw = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Close"]].rename(columns={"Close": "SPY_Close"}).dropna()
    raw["SPY_Ret_Daily"] = raw["SPY_Close"].pct_change()
    raw["SPY_Std30"]     = raw["SPY_Ret_Daily"].rolling(STD_WINDOW, min_periods=10).std()
    print(f"  SPY: {len(raw)} dias, ultimo: {raw.index.max().date()}")
    return raw


def build_spy_lookup(spy_df: pd.DataFrame) -> dict:
    """
    Construye dos dicts indexados por fecha:
      spy_close  : {date -> precio cierre SPY}
      spy_std30  : {date -> std movil 30 ruedas en esa fecha}
    Incluye propagacion hacia adelante para llenar fines de semana/feriados.
    """
    spy_close = spy_df["SPY_Close"]
    spy_std   = spy_df["SPY_Std30"]
    # ffill hasta 7 dias para cubrir fines de semana
    idx = pd.date_range(spy_df.index.min(), spy_df.index.max() + timedelta(days=10))
    spy_close = spy_close.reindex(idx).ffill()
    spy_std   = spy_std.reindex(idx).ffill()
    return spy_close.to_dict(), spy_std.to_dict()


def batch_download_tickers(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = BATCH_SIZE,
) -> dict[str, pd.Series]:
    """
    Descarga precios de cierre de una lista de tickers en batches.
    Devuelve dict: ticker -> pd.Series de Close indexado por fecha.
    """
    result: dict[str, pd.Series] = {}
    n_batches = math.ceil(len(tickers) / batch_size)

    for i in range(n_batches):
        batch = tickers[i * batch_size : (i + 1) * batch_size]
        print(f"  Batch {i+1}/{n_batches}: {len(batch)} tickers "
              f"({batch[0]} ... {batch[-1]})...")
        try:
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            if raw.empty:
                continue

            # Normalizar MultiIndex vs columnas planas
            if isinstance(raw.columns, pd.MultiIndex):
                for tk in batch:
                    try:
                        s = raw[tk]["Close"].dropna()
                        if not s.empty:
                            result[tk] = s
                    except (KeyError, TypeError):
                        pass
            else:
                # Un solo ticker en el batch
                if "Close" in raw.columns:
                    tk = batch[0]
                    result[tk] = raw["Close"].dropna()

        except Exception as e:
            print(f"    [WARN] Error en batch {i+1}: {e}")
            # Reintentar de a uno como fallback
            for tk in batch:
                try:
                    r = yf.download(tk, start=start_date, end=end_date,
                                    progress=False, auto_adjust=True)
                    if not r.empty and "Close" in r.columns:
                        result[tk] = r["Close"].dropna()
                except Exception:
                    pass

    print(f"  Descarga completada: {len(result)}/{len(tickers)} tickers con datos")
    return result


# ===========================================================================
# CORE
# ===========================================================================

def compute_targets(df: pd.DataFrame, backfill: bool = False) -> tuple[pd.DataFrame, int]:
    """
    Rellena las columnas 55-63 para las filas elegibles.
    Devuelve (df_modificado, n_filas_actualizadas).
    """
    today = pd.Timestamp(datetime.today().date())

    # ---- Mapear columnas requeridas de entrada ----
    c_ticker  = find_col(df, "Ticker")
    c_date    = find_col(df, "Data_Date")
    c_close   = find_col(df, "Close")
    c_sma50   = find_col(df, "SMA_50")

    missing_input = [k for k, v in [("Ticker", c_ticker), ("Data_Date", c_date),
                                     ("Close", c_close)] if v is None]
    if missing_input:
        raise KeyError(f"Columnas de entrada no encontradas: {missing_input}. "
                       f"Columnas disponibles: {list(df.columns[:10])}...")

    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")

    # ---- Mapear / crear columnas de salida ----
    c_dia_sem   = ensure_col(df, "Dia_una_semana")
    c_clave     = ensure_col(df, "Clave")
    c_precio    = ensure_col(df, "Precio")
    c_precio_fw = ensure_col(df, "Precio_una_semana")
    c_ret       = ensure_col(df, "Ret_Una_sem")
    c_min       = ensure_col(df, "Min")
    c_max       = ensure_col(df, "Max")
    c_target    = ensure_col(df, "Target_real")
    c_sma50p    = ensure_col(df, "SMA_50_porc")

    # Asegurar que "Dia_una_semana" ya este calculado (no requiere descarga)
    mask_dia_empty = df[c_dia_sem].isna()
    if mask_dia_empty.any():
        df.loc[mask_dia_empty, c_dia_sem] = (
            df.loc[mask_dia_empty, c_date] + pd.Timedelta(days=FWD_DAYS_CAL)
        )

    df[c_dia_sem] = pd.to_datetime(df[c_dia_sem], errors="coerce")

    # ---- Seleccionar filas a actualizar ----
    # Criterio: Target_real vacio Y "Dia una semana" ya paso (precio futuro disponible)
    if backfill:
        mask = df[c_target].isna() & df[c_date].notna()
    else:
        mask = (
            df[c_target].isna()
            & df[c_dia_sem].notna()
            & (df[c_dia_sem] <= today)
        )

    n_to_update = int(mask.sum())
    if n_to_update == 0:
        print("  No hay filas elegibles para actualizar.")
        return df, 0

    print(f"  Filas a actualizar: {n_to_update:,}")
    sub = df[mask].copy()

    # ---- Calcular columnas que NO requieren descarga ----
    # Col 55: Dia_una_semana  (ya calculada arriba)
    # Col 56: Clave
    df.loc[mask, c_clave] = (
        sub[c_ticker].astype(str) + "-" +
        sub[c_date].apply(to_excel_serial).astype(str)
    )
    # Col 57: Precio = Close
    df.loc[mask, c_precio] = pd.to_numeric(sub[c_close], errors="coerce")

    # Col 63: SMA_50_porc
    if c_sma50:
        sma50_vals = pd.to_numeric(sub[c_sma50], errors="coerce")
        close_vals = pd.to_numeric(sub[c_close], errors="coerce")
        df.loc[mask, c_sma50p] = (close_vals - sma50_vals) / sma50_vals.replace(0, np.nan)
    else:
        print("  [WARN] Columna SMA_50 no encontrada; SMA_50_porc se dejara en NaN.")

    # ---- Descargar SPY diario ----
    spy_df = download_spy_daily(SPY_HISTORY_YEARS)
    spy_close_dict, spy_std_dict = build_spy_lookup(spy_df)

    # ---- Descargar precios futuros de tickers ----
    unique_tickers = [t for t in sub[c_ticker].dropna().unique().tolist()
                      if str(t).upper() != "SPY"]
    date_min = sub[c_date].min() - timedelta(days=5)
    date_max = sub[c_dia_sem].max() + timedelta(days=10)
    start_str = date_min.strftime("%Y-%m-%d")
    end_str   = date_max.strftime("%Y-%m-%d")

    print(f"  Rango de fechas: {start_str} -> {end_str}")
    print(f"  Tickers unicos a descargar: {len(unique_tickers)}")

    ticker_prices = batch_download_tickers(unique_tickers, start_str, end_str)
    # Agregar SPY al dict (ya lo tenemos)
    ticker_prices["SPY"] = spy_df["SPY_Close"]

    # ---- Calcular columnas por fila ----
    precio_fw_vals = []
    ret_vals       = []
    min_vals       = []
    max_vals       = []
    target_vals    = []

    rows_ok = rows_no_price = rows_no_spy = 0

    for idx, row in sub.iterrows():
        ticker   = str(row[c_ticker]).upper().strip()
        data_dt  = pd.Timestamp(row[c_date]).normalize()
        dia_fw   = pd.Timestamp(row[c_dia_sem]).normalize()
        close_p  = float(row[c_close]) if pd.notna(row[c_close]) else float("nan")

        # Precio futuro del ticker
        if ticker in ticker_prices and not math.isnan(close_p):
            precio_fw = next_trading_day(ticker_prices[ticker], dia_fw)
        else:
            precio_fw = float("nan")

        # Retorno del ticker
        if not math.isnan(precio_fw) and close_p > 0:
            ret = (precio_fw / close_p) - 1.0
        else:
            ret = float("nan")
            rows_no_price += 1

        # SPY ret semanal y std
        spy_price_data = spy_close_dict.get(data_dt, float("nan"))
        spy_price_fw   = spy_close_dict.get(dia_fw,  float("nan"))
        std_val        = spy_std_dict.get(data_dt,   float("nan"))

        if (not math.isnan(spy_price_data) and not math.isnan(spy_price_fw)
                and spy_price_data > 0):
            spy_ret_sem = (spy_price_fw / spy_price_data) - 1.0
        else:
            spy_ret_sem = float("nan")
            rows_no_spy += 1

        if not math.isnan(spy_ret_sem) and not math.isnan(std_val):
            mn = spy_ret_sem - STD_MULT * std_val
            mx = spy_ret_sem + STD_MULT * std_val
        else:
            mn = mx = float("nan")

        # Target
        if not math.isnan(ret) and not math.isnan(mn) and not math.isnan(mx):
            if ret < mn:
                tgt = 1
            elif ret > mx:
                tgt = 3
            else:
                tgt = 2
            rows_ok += 1
        else:
            tgt = float("nan")

        precio_fw_vals.append(precio_fw)
        ret_vals.append(ret)
        min_vals.append(mn)
        max_vals.append(mx)
        target_vals.append(tgt)

    df.loc[mask, c_precio_fw] = precio_fw_vals
    df.loc[mask, c_ret]       = ret_vals
    df.loc[mask, c_min]       = min_vals
    df.loc[mask, c_max]       = max_vals
    df.loc[mask, c_target]    = target_vals

    print(f"  Targets calculados OK : {rows_ok:,}")
    print(f"  Sin precio del ticker : {rows_no_price:,}")
    print(f"  Sin datos SPY         : {rows_no_spy:,}")

    n_filled = int(pd.Series(target_vals).notna().sum())
    return df, n_filled


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Calcula targets en el consolidado semanal")
    p.add_argument("--dry-run",  action="store_true",
                   help="Muestra cuantas filas se actualizarian sin escribir el archivo")
    p.add_argument("--backfill", action="store_true",
                   help="Reprocesa TODAS las filas sin Target_real (no solo las pasadas)")
    p.add_argument("--file",     default=str(FILE_PATH),
                   help=f"Ruta al Excel (default: {FILE_PATH})")
    return p.parse_args()


def main():
    args   = parse_args()
    fpath  = Path(args.file)
    today  = datetime.today()
    ts     = today.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 65)
    print("TARGET UPDATER")
    print(f"  Fecha/hora : {ts}")
    print(f"  Archivo    : {fpath}")
    print(f"  Dry-run    : {args.dry_run}")
    print(f"  Backfill   : {args.backfill}")
    print("=" * 65)

    # ---- Verificar que el archivo exista ----
    if not fpath.exists():
        print(f"  [FAIL] Archivo no encontrado: {fpath}")
        sys.exit(1)

    # ---- Cargar ----
    print(f"\n[1/4] Cargando {fpath.name}...")
    df = pd.read_excel(fpath, sheet_name=SHEET, header=0)
    print(f"  Filas: {len(df):,}  |  Columnas: {len(df.columns)}")
    print(f"  Primeras columnas: {list(df.columns[:8])}")

    if args.dry_run:
        # Solo contar cuantas filas se actualizarian
        today_ts = pd.Timestamp(today.date())
        c_target = None
        for alias in COL_ALIASES["Target_real"]:
            if alias in df.columns:
                c_target = alias
                break
        c_date = None
        for alias in COL_ALIASES["Data_Date"]:
            if alias in df.columns:
                c_date = alias
                break
        if c_target and c_date:
            df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
            mask = (df[c_target].isna()
                    & (df[c_date] + pd.Timedelta(days=FWD_DAYS_CAL) <= today_ts))
            print(f"\n  [DRY-RUN] Filas que se actualizarian: {mask.sum():,}")
            print(f"  [DRY-RUN] Filas con Target_real ya lleno: "
                  f"{df[c_target].notna().sum():,}")
        else:
            print("  [DRY-RUN] No se pudo encontrar la columna Target_real.")
        return

    # ---- Calcular targets ----
    print(f"\n[2/4] Calculando targets...")
    df, n_filled = compute_targets(df, backfill=args.backfill)

    if n_filled == 0:
        print("\n  Nada que actualizar. Archivo no modificado.")
        return

    # ---- Guardar ----
    print(f"\n[3/4] Guardando {fpath.name} ({n_filled:,} targets nuevos)...")

    # Backup del archivo original (por seguridad)
    backup_path = fpath.with_stem(fpath.stem + f"_backup_{today.strftime('%Y%m%d_%H%M%S')}")
    import shutil
    shutil.copy2(fpath, backup_path)
    print(f"  Backup guardado en: {backup_path.name}")

    with pd.ExcelWriter(fpath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=SHEET, index=False)
    print(f"  Guardado OK: {fpath}")

    # ---- Resumen ----
    c_target = find_col(df, "Target_real")
    if c_target:
        dist = df[c_target].value_counts().sort_index()
        total_filled = int(df[c_target].notna().sum())
        print(f"\n[4/4] Resumen")
        print("=" * 65)
        print(f"  Total filas con Target_real : {total_filled:,}")
        print(f"  Nuevos targets calculados   : {n_filled:,}")
        print(f"  Distribucion de targets:")
        for val, cnt in dist.items():
            pct = 100 * cnt / max(1, total_filled)
            label = {1.0: "Target 1 (underperform)", 2.0: "Target 2 (medio)",
                     3.0: "Target 3 (outperform)"}.get(float(val), str(val))
            print(f"    {label:30s}  {cnt:>7,}  ({pct:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
