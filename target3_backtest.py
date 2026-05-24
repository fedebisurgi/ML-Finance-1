# -*- coding: utf-8 -*-
"""
target3_backtest.py  —  entry point de validación walk-forward

CUÁNDO USARLO
-------------
Bajo demanda al iterar el modelo: cambios en features, hiperparámetros, filtros.
Entrena N veces (una por anchor_date en BACKTEST_DATES) y produce 3 Excels
de backtest con métricas de atribución (spearman, decile spread, top3 vs top20).

Duración aproximada: proporcional a len(BACKTEST_DATES). Para las 18 fechas
default puede tardar horas (el script imprime estimación tras el primer run).

NO produce los Top-20 productivos del último cierre. Para eso usar
target3_predict.py.
"""

import numpy as np

from target3_core import (
    load_and_prepare_data, run_backtest_loop,
    OUT_FILE_BT_BASELINE, OUT_FILE_BT_ABLATION, OUT_FILE_BT_NEW,
)

# ── Fechas a backtestear ─────────────────────────────────────────
# Ajustar antes de cada corrida. Default amplio: 17 fechas
# 2026-01-09 → 2026-05-15 (omitidas 27/3 y 3/4 a propósito).
BACKTEST_DATES = [
    "2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30",
    "2026-02-06", "2026-02-13", "2026-02-20", "2026-02-27",
    "2026-03-06", "2026-03-13", "2026-03-20",
    "2026-04-10", "2026-04-17", "2026-04-24",
    "2026-05-01", "2026-05-08", "2026-05-15",
]


def main():
    ctx = load_and_prepare_data()

    bt_base = run_backtest_loop(
        df_all=ctx.df,
        feature_cols=ctx.feature_cols_base,
        universe_mask_col="_univ_close5",
        run_label="BACKTEST_BASELINE",
        out_file=OUT_FILE_BT_BASELINE,
        backtest_dates=BACKTEST_DATES,
    )

    bt_abl = run_backtest_loop(
        df_all=ctx.df,
        feature_cols=ctx.feature_cols_base,
        universe_mask_col="_univ_pass",
        run_label="BACKTEST_ABLATION",
        out_file=OUT_FILE_BT_ABLATION,
        backtest_dates=BACKTEST_DATES,
    )

    bt_new = run_backtest_loop(
        df_all=ctx.df,
        feature_cols=ctx.feature_cols_new,
        universe_mask_col="_univ_pass",
        run_label="BACKTEST_NEW",
        out_file=OUT_FILE_BT_NEW,
        backtest_dates=BACKTEST_DATES,
    )

    # ── Tabla comparativa final ──────────────────────────────────
    def _bt_avg(bt_dict, key):
        vals = [v[key] for v in bt_dict.values()
                if isinstance(v.get(key), float) and not np.isnan(v[key])]
        return float(np.mean(vals)) if vals else np.nan

    def _bt_best_worst(bt_dict):
        rows = [(v["Anchor_Date"], v["Top20_AvgRet"]) for v in bt_dict.values()
                if isinstance(v.get("Top20_AvgRet"), float) and not np.isnan(v["Top20_AvgRet"])]
        if not rows:
            return "N/A", "N/A"
        best  = max(rows, key=lambda x: x[1])
        worst = min(rows, key=lambda x: x[1])
        return f"{best[0]}({best[1]:+.1%})", f"{worst[0]}({worst[1]:+.1%})"

    print(f"\n\n{'='*80}")
    print("=== COMPARATIVA FINAL BACKTEST ===")
    print(f"{'='*80}")

    _hdr = f"  {'Variante':<20} {'Avg P@20':>9} {'Avg Top20 Ret':>14} {'Avg HitRate vs SPY':>19} {'Mejor sem':>22} {'Peor sem':>22}"
    print(_hdr)
    print("  " + "-" * (len(_hdr) - 2))

    for lbl, bt_d in [("BASELINE", bt_base), ("ABLATION", bt_abl), ("NEW", bt_new)]:
        ap20  = _bt_avg(bt_d, "P@20_realized")
        aret  = _bt_avg(bt_d, "Top20_AvgRet")
        ahit  = _bt_avg(bt_d, "HitRate_vs_SPY")
        best_s, worst_s = _bt_best_worst(bt_d)
        ap20_s = f"{ap20:.3f}"  if not np.isnan(ap20) else " N/A "
        aret_s = f"{aret:+.2%}" if not np.isnan(aret) else "   N/A  "
        ahit_s = f"{ahit:.0%}"  if not np.isnan(ahit) else "  N/A  "
        print(f"  {lbl:<20} {ap20_s:>9} {aret_s:>14} {ahit_s:>19} {best_s:>22} {worst_s:>22}")

    print()


if __name__ == "__main__":
    main()
