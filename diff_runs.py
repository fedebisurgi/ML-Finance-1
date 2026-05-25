# -*- coding: utf-8 -*-
"""
diff_runs.py  —  Validador de paridad para el PR predict↔backtest.

USO
---
  python diff_runs.py <predict_excel> <backtest_excel>

  predict_excel : salida de target3_predict.py   (ej. T5_NEW_2026-05-25_...xlsx)
  backtest_excel: salida de target3_backtest.py  con UNA SOLA anchor_date
                  = la última fecha del dataset  (ej. T5_BACKTEST_NEW_...xlsx)

TESTS QUE CORRE
---------------
  A) predict Holdout_Scored == monolito viejo (si se pasa --monolito como 3er arg)
  B) predict Top_20_Last  vs  backtest sheet de la única fecha
     Criterio: ticker exacto y Prob_T3_FINAL / RankScore a 4 decimales.
"""

import sys
import numpy as np
import pandas as pd


TOL = 1e-4   # 4 decimales


def _load(path, sheet):
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        print(f"  [WARN] No se pudo leer '{sheet}' de {path}: {e}")
        return None


def _col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def check_top20_parity(pred_file, bt_file):
    print("\n" + "=" * 70)
    print("TEST B: predict Top_20_Last  ==  backtest anchor sheet")
    print("=" * 70)

    pred_top = _load(pred_file, "Top_20_Last")
    if pred_top is None:
        print("  FAIL — Top_20_Last no encontrado en predict.")
        return False

    # Backtest: leer Summary para saber cuál es la única fecha
    bt_summary = _load(bt_file, "Summary")
    bt_sheets = None
    try:
        xl = pd.ExcelFile(bt_file)
        date_sheets = [s for s in xl.sheet_names if s not in ("Summary", "Attribution", "Config")]
        if not date_sheets:
            print("  FAIL — Backtest no tiene sheets de fecha.")
            return False
        if len(date_sheets) > 1:
            print(f"  WARN — Backtest tiene {len(date_sheets)} fechas. "
                  "Para paridad se espera UNA SOLA. Usando la última: "
                  f"{sorted(date_sheets)[-1]}")
        sheet_name = sorted(date_sheets)[-1]
    except Exception as e:
        print(f"  FAIL — No se pudo leer backtest: {e}")
        return False

    bt_top = _load(bt_file, sheet_name)
    if bt_top is None:
        return False

    print(f"  Predict rows : {len(pred_top)}")
    print(f"  Backtest rows: {len(bt_top)}  (sheet: {sheet_name})")

    if len(pred_top) != len(bt_top):
        print(f"  FAIL — Distinto número de filas ({len(pred_top)} vs {len(bt_top)}).")
        return False

    tick_p = _col(pred_top, "Ticker", "TICKER", "ticker")
    tick_b = _col(bt_top,  "Ticker", "TICKER", "ticker")
    if tick_p is None or tick_b is None:
        print("  WARN — No encuentro columna Ticker; skipping ticker check.")
    else:
        mismatches = (pred_top[tick_p].values != bt_top[tick_b].values)
        if mismatches.any():
            diff_idx = np.where(mismatches)[0]
            for i in diff_idx[:5]:
                print(f"  MISMATCH Rank {i+1}: predict={pred_top[tick_p].iloc[i]}  "
                      f"backtest={bt_top[tick_b].iloc[i]}")
            print(f"  FAIL — {mismatches.sum()} tickers difieren.")
            return False
        print(f"  Tickers: {len(pred_top)} / {len(pred_top)} coinciden ✓")

    ok = True
    for col in ["RankScore", "Prob_T3_FINAL", "Prob_Clf"]:
        cp = _col(pred_top, col)
        cb = _col(bt_top,  col)
        if cp is None or cb is None:
            print(f"  WARN — '{col}' no encontrado en uno de los dos; skip.")
            continue
        diff = (pred_top[cp].astype(float) - bt_top[cb].astype(float)).abs()
        maxd = diff.max()
        if maxd > TOL:
            print(f"  FAIL — {col}: max diff = {maxd:.6f}  (tol={TOL})")
            ok = False
        else:
            print(f"  {col}: max diff = {maxd:.6f}  ✓")

    if ok:
        print("\n  *** PARIDAD EXACTA: predict Top_20_Last == backtest anchor. ***")
    return ok


def check_holdout_parity(pred_file, monolito_file):
    print("\n" + "=" * 70)
    print("TEST A: predict Holdout_Scored  ==  monolito Holdout_Scored")
    print("=" * 70)

    pred_h = _load(pred_file, "Holdout_Scored")
    mono_h = _load(monolito_file, "Holdout_Scored")
    if pred_h is None or mono_h is None:
        print("  No se pudo cargar una de las dos hojas.")
        return False

    print(f"  Predict rows : {len(pred_h)}")
    print(f"  Monolito rows: {len(mono_h)}")

    if len(pred_h) != len(mono_h):
        print(f"  FAIL — Distinto número de filas.")
        return False

    ok = True
    for col in ["RankScore", "Prob_T3_FINAL", "Prob_Clf"]:
        cp = _col(pred_h, col)
        cm = _col(mono_h, col)
        if cp is None or cm is None:
            print(f"  WARN — '{col}' no encontrado; skip.")
            continue
        diff = (pred_h[cp].astype(float) - mono_h[cm].astype(float)).abs()
        maxd = diff.max()
        if maxd > TOL:
            print(f"  FAIL — {col}: max diff = {maxd:.6f}  (tol={TOL})")
            ok = False
        else:
            print(f"  {col}: max diff = {maxd:.6f}  ✓")

    if ok:
        print("\n  *** PARIDAD EXACTA: predict Holdout_Scored == monolito. ***")
    return ok


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    pred_file = sys.argv[1]
    bt_file   = sys.argv[2]
    mono_file = sys.argv[3] if len(sys.argv) > 3 else None

    results = []
    results.append(check_top20_parity(pred_file, bt_file))
    if mono_file:
        results.append(check_holdout_parity(pred_file, mono_file))

    print("\n" + "=" * 70)
    if all(results):
        print("TODOS LOS TESTS PASARON — PR listo para mergear.")
    else:
        print("ALGÚN TEST FALLÓ — revisar diffs arriba antes de mergear.")
    print("=" * 70)


if __name__ == "__main__":
    main()
