# -*- coding: utf-8 -*-
"""
test_automatizacion.py
======================
Script de prueba para validar la automatizacion diaria ANTES de correr
el pipeline completo.

Que hace:
  1. Descarga datos historicos de SPY + 5 tickers de ejemplo via yfinance
  2. Calcula todos los indicadores tecnicos (Gemini + ChatGPT) del pipeline
  3. Agrega features de fuerza relativa vs SPY (RS1M_SPY, RS3M_SPY, RS6M_SPY)
  4. Guarda el resultado en un Excel pequeno con una hoja por ticker
     + una hoja "Resumen" con el estado de cada indicador

Uso:
  python test_automatizacion.py

Interpretacion del output:
  - PASS: descarga y calculo OK, Excel generado
  - WARN: el ticker tuvo algun problema pero el resto funcionó
  - FAIL: error critico que romperia el pipeline completo

Tickers de ejemplo configurables en TEST_TICKERS (no incluir SPY, se agrega
automaticamente).
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas_ta as ta

# ===========================================================================
# CONFIG
# ===========================================================================
TEST_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'TSLA']  # editables
PERIODO_ANOS = 3        # años de historia a descargar (igual que el pipeline)
OUTPUT_XLSX  = Path("test_automatizacion_output.xlsx")

# ===========================================================================
# HELPERS DE INDICADORES  (identicos a Datos tickers para ML_*.py)
# ===========================================================================

def _streak_up(s: pd.Series) -> pd.Series:
    inc = (s > s.shift(1)).astype(int)
    run, out = 0, []
    for v in inc.fillna(0):
        run = run + 1 if v == 1 else 0
        out.append(run)
    return pd.Series(out, index=s.index)


def _days_since_cross(short: pd.Series, long: pd.Series, cross='golden') -> pd.Series:
    short = short.copy()
    long  = long.copy()
    above_prev = (short.shift(1) > long.shift(1))
    above_now  = (short > long)
    if cross == 'golden':
        evt = (~above_prev.fillna(False)) & (above_now.fillna(False))
    else:
        evt = (above_prev.fillna(False)) & (~above_now.fillna(False))

    res = pd.Series(np.nan, index=short.index)
    last_idx = None
    for i, flag in enumerate(evt):
        if flag:
            last_idx = i
        if last_idx is not None:
            res.iat[i] = i - last_idx

    if res.notna().any():
        res = res.ffill()
    else:
        res[:] = len(res) + 1
    return res


def _aroon(df: pd.DataFrame, period=25):
    high = df['High']
    low  = df['Low']
    hh_idx = high.rolling(period).apply(np.argmax, raw=True)
    ll_idx = low.rolling(period).apply(np.argmin, raw=True)
    up   = 100 * (period - 1 - hh_idx) / (period - 1)
    down = 100 * (period - 1 - ll_idx) / (period - 1)
    return up, down, up - down


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame | None:
    """Calcula todos los indicadores Gemini + ChatGPT sobre un DataFrame OHLCV."""
    if df.empty or len(df) < 100:
        return None

    df = df.rename(columns={c: c.capitalize() for c in df.columns})

    # --- Gemini ---
    df.ta.mfi(length=14, append=True)
    df.ta.macd(append=True)

    bb = df.ta.bbands(length=20, std=2, append=False)
    if bb is not None and not bb.empty:
        bbp_cols = [c for c in bb.columns if "BBP" in c]
        df['PctB'] = bb[bbp_cols[0]] if bbp_cols else np.nan
    else:
        df['PctB'] = np.nan

    df.ta.cci(length=20, append=True)
    df.ta.obv(append=True)

    stoch = df.ta.stoch(k=14, d=3, smooth_k=3)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)

    df.ta.adx(length=14, append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    df.ta.sma(length=200, append=True)

    if 'SMA_50' not in df.columns:
        df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = df['Close'].rolling(200, min_periods=1).mean()

    # ATR%
    df.ta.atr(length=14, append=True)
    atr_col = next(
        (c for c in df.columns if c.upper().startswith('ATR') and c.endswith('_14')),
        None
    )
    if atr_col is None:
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift(1)).abs(),
            (df['Low']  - df['Close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=1).mean()
    else:
        atr14 = df[atr_col]
    df['ATRp_14'] = (atr14 / df['Close']).replace([np.inf, -np.inf], np.nan) * 100

    # VROC, OBV trend, Stoch cross
    df['VROC_14']   = df['Volume'].pct_change(14) * 100
    df['OBV_trend'] = df.get('OBV', pd.Series(index=df.index)).pct_change(20).replace(
        [np.inf, -np.inf], np.nan
    )

    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
        k, d = df['STOCHk_14_3_3'], df['STOCHd_14_3_3']
        df['STOCH_Cross'] = np.select(
            [(k.shift(1) <= d.shift(1)) & (k > d),
             (k.shift(1) >= d.shift(1)) & (k < d)],
            [1, -1], default=0
        )
    else:
        df['STOCH_Cross'] = 0

    if 'SMA_20' in df.columns:
        df['MA_SLOPE_20'] = df['SMA_20'].diff(5) / df['SMA_20'].shift(5).replace(0, np.nan)
    else:
        df['MA_SLOPE_20'] = np.nan

    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        s50, s200 = df['SMA_50'], df['SMA_200']
        df['Cross_Signal'] = np.select(
            [(s50.shift(1) < s200.shift(1)) & (s50 > s200),
             (s50.shift(1) > s200.shift(1)) & (s50 < s200)],
            [1, -1], default=0
        )
    else:
        df['Cross_Signal'] = 0

    if 'Volume' in df.columns:
        vol_grows = df['Volume'] > df['Volume'].shift(1)
        df['Consecutive_Volume_Growth'] = (
            vol_grows.cumsum()
            - vol_grows.cumsum().where(~vol_grows).ffill().fillna(0)
        )
    else:
        df['Consecutive_Volume_Growth'] = 0

    roll_90  = df['Close'].rolling(90)
    sw_high  = roll_90.max()
    sw_low   = roll_90.min()
    df['FIB_Range_90D'] = (df['Close'] - sw_low) / (sw_high - sw_low).replace(0, np.nan)

    # --- ChatGPT ---
    df.ta.rsi(length=14, append=True)

    if 'MACDh_12_26_9' in df.columns:
        df['MACD_Hist_Slope_5'] = df['MACDh_12_26_9'].diff(5)
    else:
        df['MACD_Hist_Slope_5'] = np.nan

    _, _, a_diff = _aroon(df, 25)
    df['Aroon_Diff_25'] = a_diff

    df['RET_1M'] = df['Close'].pct_change(20)
    df['RET_3M'] = df['Close'].pct_change(60)
    df['RET_6M'] = df['Close'].pct_change(126)

    if 'Volume' in df.columns:
        df['Vol_Rel_20']  = df['Volume'] / df['Volume'].rolling(20).mean().replace(0, np.nan)
        df['Vol_StreakUp'] = _streak_up(df['Volume'])
    else:
        df['Vol_Rel_20']  = np.nan
        df['Vol_StreakUp'] = np.nan

    sma50  = df.get('SMA_50',  pd.Series(index=df.index))
    sma200 = df.get('SMA_200', pd.Series(index=df.index))
    df['Days_since_Golden'] = _days_since_cross(sma50, sma200, 'golden')
    df['Days_since_Death']  = _days_since_cross(sma50, sma200, 'death')

    eps = 1e-9
    df['Ret13_Ratio'] = df['RET_1M'] / (df['RET_3M'].replace(0, eps) + eps)

    high_52w = df['Close'].rolling(252, min_periods=60).max()
    low_52w  = df['Close'].rolling(252, min_periods=60).min()
    rng_52w  = (high_52w - low_52w)
    df['Pct_in_52w_range'] = (df['Close'] - low_52w) / rng_52w.replace(0, np.nan)
    df['Drawdown_52w']     = (high_52w - df['Close']) / high_52w.replace(0, np.nan)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    return df


def add_spy_relative_strength(df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega RS1M_SPY, RS3M_SPY, RS6M_SPY alineando por fecha."""
    spy_close = spy_df['Close'].rename('SPY_Close')
    df = df.join(spy_close, how='left')
    df['SPY_Close'] = df['SPY_Close'].ffill()

    df['RS1M_SPY'] = df['RET_1M'] - df['SPY_Close'].pct_change(20)
    df['RS3M_SPY'] = df['RET_3M'] - df['SPY_Close'].pct_change(60)
    df['RS6M_SPY'] = df['RET_6M'] - df['SPY_Close'].pct_change(126)

    df.drop(columns=['SPY_Close'], inplace=True)
    return df


# ===========================================================================
# EXPECTED INDICATORS — lista completa para el chequeo de cobertura
# ===========================================================================
EXPECTED_INDICATORS = [
    'MFI_14', 'MACDh_12_26_9', 'PctB', 'OBV_trend', 'STOCH_Cross',
    'ADX_14', 'MA_SLOPE_20', 'Cross_Signal', 'ATRp_14', 'VROC_14',
    'Consecutive_Volume_Growth', 'FIB_Range_90D',
    'RSI_14', 'MACD_Hist_Slope_5', 'RET_1M', 'RET_3M', 'RET_6M',
    'Aroon_Diff_25', 'Vol_Rel_20', 'Vol_StreakUp',
    'Days_since_Golden', 'Days_since_Death',
    'Ret13_Ratio', 'Pct_in_52w_range', 'Drawdown_52w',
    'SMA_50', 'SMA_200',
    'RS1M_SPY', 'RS3M_SPY', 'RS6M_SPY',
]


# ===========================================================================
# DESCARGA
# ===========================================================================

def download_ticker(ticker: str, period_years: int) -> pd.DataFrame | None:
    """Descarga datos historicos de un ticker. Devuelve None si falla."""
    end   = datetime.today()
    start = end.replace(year=end.year - period_years)
    try:
        raw = yf.download(
            ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=True,
        )
        if raw.empty or len(raw) < 100:
            return None
        # yfinance puede devolver MultiIndex si se pide un solo ticker como lista
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw
    except Exception:
        return None


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    all_tickers = ['SPY'] + TEST_TICKERS
    date_stamp  = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    output_path = OUTPUT_XLSX.with_stem(f"{OUTPUT_XLSX.stem}_{date_stamp}")

    print("=" * 65)
    print("TEST DE AUTOMATIZACION — descarga + indicadores + Excel")
    print(f"Tickers: {all_tickers}")
    print(f"Periodo: {PERIODO_ANOS} años")
    print(f"Output:  {output_path}")
    print("=" * 65)

    # ── Paso 1: descargar todos los tickers ──────────────────────────
    print("\n[1/3] Descargando datos...")
    raw_data: dict[str, pd.DataFrame] = {}
    for tkr in all_tickers:
        df_raw = download_ticker(tkr, PERIODO_ANOS)
        if df_raw is None:
            print(f"  WARN  {tkr:8s}  descarga fallida o insuficiente")
        else:
            raw_data[tkr] = df_raw
            print(f"  PASS  {tkr:8s}  {len(df_raw):>4d} filas  "
                  f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    if 'SPY' not in raw_data:
        print("\n  FAIL  SPY no descargó. Sin SPY no hay fuerza relativa.")
        print("  Abortando — revisá la conexión a internet o el estado de yfinance.")
        sys.exit(1)

    spy_df = raw_data['SPY'].copy()

    # ── Paso 2: calcular indicadores ─────────────────────────────────
    print("\n[2/3] Calculando indicadores...")
    processed: dict[str, pd.DataFrame] = {}
    summary_rows = []

    for tkr in all_tickers:
        if tkr not in raw_data:
            summary_rows.append({
                'Ticker': tkr, 'Status': 'WARN',
                'Filas': 0, 'Ultima_Fecha': None,
                'Indicadores_OK': 0, 'Indicadores_Faltantes': len(EXPECTED_INDICATORS),
                'Faltantes_Detalle': 'descarga fallida',
            })
            continue

        try:
            df_ind = add_all_indicators(raw_data[tkr].copy())
            if df_ind is None:
                raise ValueError("add_all_indicators devolvio None (datos insuficientes)")

            if tkr != 'SPY':
                df_ind = add_spy_relative_strength(df_ind, spy_df)

            # Auditar cobertura de indicadores
            missing = [ind for ind in EXPECTED_INDICATORS if ind not in df_ind.columns]
            ok_count = len(EXPECTED_INDICATORS) - len(missing)
            status = 'PASS' if not missing else 'WARN'

            print(f"  {status}  {tkr:8s}  {len(df_ind):>4d} filas  "
                  f"{ok_count}/{len(EXPECTED_INDICATORS)} indicadores OK"
                  + (f"  faltantes: {missing}" if missing else ""))

            processed[tkr] = df_ind
            summary_rows.append({
                'Ticker': tkr,
                'Status': status,
                'Filas': len(df_ind),
                'Ultima_Fecha': df_ind.index[-1].date() if len(df_ind) > 0 else None,
                'Indicadores_OK': ok_count,
                'Indicadores_Faltantes': len(missing),
                'Faltantes_Detalle': ', '.join(missing) if missing else '',
            })

        except Exception as e:
            print(f"  FAIL  {tkr:8s}  error al calcular indicadores: {e}")
            traceback.print_exc()
            summary_rows.append({
                'Ticker': tkr, 'Status': 'FAIL',
                'Filas': 0, 'Ultima_Fecha': None,
                'Indicadores_OK': 0, 'Indicadores_Faltantes': len(EXPECTED_INDICATORS),
                'Faltantes_Detalle': str(e),
            })

    # ── Paso 3: guardar Excel ─────────────────────────────────────────
    print(f"\n[3/3] Guardando Excel en {output_path}...")
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Hoja de resumen primero
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name='Resumen', index=False)

            # Una hoja por ticker (solo ultimas 52 filas para mantener el archivo chico)
            cols_to_export = ['Open', 'High', 'Low', 'Close', 'Volume'] + EXPECTED_INDICATORS
            for tkr, df_out in processed.items():
                available_cols = [c for c in cols_to_export if c in df_out.columns]
                sheet_df = df_out[available_cols].tail(52).copy()
                sheet_df.index.name = 'Date'
                sheet_df.to_excel(writer, sheet_name=tkr[:31])  # Excel limita nombre a 31 chars

        print(f"  PASS  Excel guardado: {output_path}")
    except Exception as e:
        print(f"  FAIL  No se pudo guardar el Excel: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── Resumen final ─────────────────────────────────────────────────
    n_pass = sum(1 for r in summary_rows if r['Status'] == 'PASS')
    n_warn = sum(1 for r in summary_rows if r['Status'] == 'WARN')
    n_fail = sum(1 for r in summary_rows if r['Status'] == 'FAIL')

    print("\n" + "=" * 65)
    print(f"RESULTADO FINAL:  {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL")
    if n_fail > 0:
        print("  → Hay errores criticos. NO correr el pipeline completo.")
    elif n_warn > 0:
        print("  → Hay advertencias. Revisar tickers con WARN antes de produccion.")
    else:
        print("  → Todo OK. Pipeline completo puede ejecutarse.")
    print("=" * 65)

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == '__main__':
    main()
