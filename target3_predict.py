# -*- coding: utf-8 -*-
"""
target3_predict.py  —  entry point productivo (Top-20 semanal)

CUÁNDO USARLO
-------------
Todos los viernes después del cierre, una vez que el Excel consolidado tiene
la última semana. Entrena los 3 modelos (BASELINE / ABLATION / NEW), scorea
el último cierre y exporta los 3 Excels con el Top-20 a operar.

Duración aproximada: ~15 minutos para las 3 variantes.

NO corre backtest walk-forward. Para eso usar target3_backtest.py.
"""

import numpy as np
import pandas as pd

import target3_core as core
from target3_core import (
    load_and_prepare_data, run_pipeline,
    OUT_FILE_BASELINE, OUT_FILE_ABLATION, OUT_FILE_NEW,
    INPUT_HASH, TOP_K,
    UNIV_MIN_CLOSE, UNIV_MIN_DOLLAR_VOL_20D, UNIV_MAX_ZERO_RET_PCT_60D,
    CORR_ALERT_THRESHOLD,
)


def main():
    ctx = load_and_prepare_data()

    print("\n" + "=" * 80)
    print("PIPELINE v5.0 — MVP Universe Filters")
    print(f"  Run 1 BASELINE : features base, filtro Close>$5")
    print(f"  Run 2 ABLATION : features base, filtros universo completos")
    print(f"  Run 3 NEW      : features base + 4 nuevas, filtros universo")
    print("=" * 80)

    ho_base, fi_base, m_base, p20w_base = run_pipeline(
        df_all=ctx.df,
        feature_cols=ctx.feature_cols_base,
        universe_mask_col="_univ_close5",
        out_file=OUT_FILE_BASELINE,
        run_label="BASELINE",
    )

    ho_abl, fi_abl, m_abl, p20w_abl = run_pipeline(
        df_all=ctx.df,
        feature_cols=ctx.feature_cols_base,
        universe_mask_col="_univ_pass",
        out_file=OUT_FILE_ABLATION,
        run_label="ABLATION_FILTERS",
    )

    ho_new, fi_new, m_new, p20w_new = run_pipeline(
        df_all=ctx.df,
        feature_cols=ctx.feature_cols_new,
        universe_mask_col="_univ_pass",
        out_file=OUT_FILE_NEW,
        run_label="NEW_FULL",
    )

    df       = ctx.df
    TICK_COL = core.TICK_COL
    new_raw_features  = ctx.new_raw_features
    new_rank_features = ctx.new_rank_features
    feature_cols_base = ctx.feature_cols_base

    SEP = "=" * 70

    # ── FASE 0: UNIVERSO ─────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 0: UNIVERSO ===")
    print(SEP)

    total_tickers_orig = df[TICK_COL].nunique()
    n_after_close5     = df[df["_univ_close5"].fillna(False)][TICK_COL].nunique()
    n_after_dvol       = df[df["_univ_close5"].fillna(False) & df["_univ_dvol"].fillna(False)][TICK_COL].nunique()
    n_after_all        = df[df["_univ_pass"].fillna(False)][TICK_COL].nunique()

    print(f"  Total tickers dataset original           : {total_tickers_orig:,}")
    print(f"  Sobrevivientes Close > ${UNIV_MIN_CLOSE:.0f}              : {n_after_close5:,}")
    print(f"  + Dollar_Volume > ${UNIV_MIN_DOLLAR_VOL_20D/1e6:.0f}M          : {n_after_dvol:,}")
    print(f"  + Zero_Returns < {UNIV_MAX_ZERO_RET_PCT_60D:.0%}               : {n_after_all:,}")

    train_pool_df   = df[df["_DateKey"].isin(ctx.train_dates_pool)]
    holdout_pool_df = df[df["_DateKey"].isin(ctx.holdout_dates)]

    def median_tickers_per_date(d):
        return d.groupby("_DateKey")[TICK_COL].nunique().median()

    med_train_before = median_tickers_per_date(train_pool_df)
    med_train_after  = median_tickers_per_date(train_pool_df[train_pool_df["_univ_pass"].fillna(False)])
    med_hold_before  = median_tickers_per_date(holdout_pool_df)
    med_hold_after   = median_tickers_per_date(holdout_pool_df[holdout_pool_df["_univ_pass"].fillna(False)])

    print(f"\n  Mediana tickers/fecha — TRAIN   antes filtros: {med_train_before:.0f}  "
          f"| después: {med_train_after:.0f}")
    print(f"  Mediana tickers/fecha — HOLDOUT antes filtros: {med_hold_before:.0f}  "
          f"| después: {med_hold_after:.0f}")

    # ── FASE 1: FEATURES NUEVAS — DIAGNÓSTICO ────────────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 1: FEATURES NUEVAS — DIAGNÓSTICO ===")
    print(SEP)

    train_df_for_diag = df[df["_DateKey"].isin(ctx.train_dates_pool) & df["y_t3"].notna()].copy()

    for feat in new_raw_features:
        print(f"\n  ── {feat} ──")
        col = df[feat].replace([np.inf, -np.inf], np.nan)

        nan_pct = col.isna().mean()
        print(f"    % NaN       : {nan_pct:.2%}")

        if col.notna().sum() < 10:
            print("    [WARN] Menos de 10 valores no-NaN. Feature posiblemente no computable.")
            continue

        q  = col.quantile([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
        sk = float(col.skew())
        print(f"    Min={q[0.0]:.4f}  P1={q[0.01]:.4f}  P25={q[0.25]:.4f}  "
              f"Med={q[0.5]:.4f}  P75={q[0.75]:.4f}  P99={q[0.99]:.4f}  Max={q[1.0]:.4f}")
        print(f"    Skewness    : {sk:.3f}")

        tr_sub = train_df_for_diag[[feat] + feature_cols_base].copy()
        tr_sub = tr_sub.replace([np.inf, -np.inf], np.nan).dropna(subset=[feat])
        if len(tr_sub) > 50:
            corrs = tr_sub[feature_cols_base].corrwith(tr_sub[feat]).abs()
            max_corr_feat = corrs.idxmax()
            max_corr_val  = corrs.max()
            print(f"    Máx corr c/existentes: {max_corr_feat} = {max_corr_val:.4f}", end="")
            if max_corr_val > CORR_ALERT_THRESHOLD:
                print(f"  *** ALERTA: correlación > {CORR_ALERT_THRESHOLD} ***")
            else:
                print()

        tr_sub2 = train_df_for_diag[[feat, "y_t3"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr_sub2) > 50:
            corr_target = tr_sub2[feat].corr(tr_sub2["y_t3"])
            print(f"    Corr con target (train): {corr_target:.4f}")

    # ── FASE 2: TRAINING ─────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 2: TRAINING (modelo NEW) ===")
    print(SEP)
    print(f"  Tiempo de entrenamiento : {m_new['TrainingTime_s']:.1f}s")
    print(f"  N features usadas       : {m_new['N_Features']}")
    print(f"  N árboles CLF (final)   : {m_new['N_Trees_clf']}")
    print(f"  N árboles RNK (final)   : {m_new['N_Trees_rnk']}")

    # ── FASE 3: MÉTRICAS BASELINE vs ABLATION vs NUEVO ────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 3: MÉTRICAS COMPARATIVAS ===")
    print(SEP)

    def _fmt(v, fmt=".3f"):
        try:
            if v is None:
                return "  N/A  "
            fv = float(v)
            return "  N/A  " if np.isnan(fv) else format(fv, fmt)
        except (TypeError, ValueError):
            return "  N/A  "

    def _delta(new_val, base_val, fmt=".3f"):
        try:
            nv, bv = float(new_val), float(base_val)
            if np.isnan(nv) or np.isnan(bv):
                return "  N/A"
            d = nv - bv
            return f"{'+'if d>=0 else ''}{d:{fmt}}"
        except (TypeError, ValueError):
            return "  N/A"

    rows = [
        ("Métrica",                  "BASELINE",                          "ABLATION",                         "NEW",                          "Δ (NEW-BASE)"),
        ("-" * 28,                   "-" * 10,                            "-" * 10,                           "-" * 10,                       "-" * 12),
        (f"Holdout P@{TOP_K}",       _fmt(m_base["Holdout_P20"]),         _fmt(m_abl["Holdout_P20"]),         _fmt(m_new["Holdout_P20"]),     _delta(m_new["Holdout_P20"], m_base["Holdout_P20"])),
        ("Holdout P@10",             _fmt(m_base["Holdout_P10"]),         _fmt(m_abl["Holdout_P10"]),         _fmt(m_new["Holdout_P10"]),     _delta(m_new["Holdout_P10"], m_base["Holdout_P10"])),
        (f"Holdout NDCG@{TOP_K}",    _fmt(m_base["Holdout_NDCG20"]),      _fmt(m_abl["Holdout_NDCG20"]),      _fmt(m_new["Holdout_NDCG20"]),  _delta(m_new["Holdout_NDCG20"], m_base["Holdout_NDCG20"])),
        ("Lift Holdout",             _fmt(m_base["Holdout_Lift"], ".2f"), _fmt(m_abl["Holdout_Lift"], ".2f"), _fmt(m_new["Holdout_Lift"], ".2f"), _delta(m_new["Holdout_Lift"], m_base["Holdout_Lift"], ".2f")),
        ("Avg Ret Top-Bot (spread)", _fmt(m_base["AvgRet_Spread"]),       _fmt(m_abl["AvgRet_Spread"]),       _fmt(m_new["AvgRet_Spread"]),   _delta(m_new["AvgRet_Spread"], m_base["AvgRet_Spread"])),
        ("Hit rate Top20 vs SPY",    _fmt(m_base["HitRate_vs_SPY"]),      _fmt(m_abl["HitRate_vs_SPY"]),      _fmt(m_new["HitRate_vs_SPY"]),  _delta(m_new["HitRate_vs_SPY"], m_base["HitRate_vs_SPY"])),
        ("HHI sectorial Top20",      _fmt(m_base["HHI_Sectorial"]),       _fmt(m_abl["HHI_Sectorial"]),       _fmt(m_new["HHI_Sectorial"]),   _delta(m_new["HHI_Sectorial"], m_base["HHI_Sectorial"])),
        (f"OOF P@{TOP_K}",           _fmt(m_base["OOF_P20"]),             _fmt(m_abl["OOF_P20"]),             _fmt(m_new["OOF_P20"]),         _delta(m_new["OOF_P20"], m_base["OOF_P20"])),
        ("OOF-Holdout gap",          _fmt(m_base["OOF_Hold_Gap"]),        _fmt(m_abl["OOF_Hold_Gap"]),        _fmt(m_new["OOF_Hold_Gap"]),    _delta(m_new["OOF_Hold_Gap"], m_base["OOF_Hold_Gap"])),
    ]

    col_w = [30, 12, 12, 12, 14]
    for row in rows:
        print("  " + "".join(str(v).ljust(w) for v, w in zip(row, col_w)))

    print(f"\n  Ablation (filtros solos) Holdout P@{TOP_K}: {_fmt(m_abl['Holdout_P20'])}  "
          f"(Δ vs baseline: {_delta(m_abl['Holdout_P20'], m_base['Holdout_P20'])})")

    # ── FASE 4: FEATURE IMPORTANCE ────────────────────────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 4: FEATURE IMPORTANCE (modelo NEW — Top 20 by gain) ===")
    print(SEP)

    print(f"  {'Rank':<6} {'Feature':<40} {'Importance':>12}")
    print(f"  {'-'*6} {'-'*40} {'-'*12}")
    for _, row in fi_new.head(20).iterrows():
        print(f"  {int(row['Rank']):<6} {row['Feature']:<40} {row['Importance']:>12.1f}")

    print(f"\n  Posición de cada feature nueva en el ranking:")
    for feat in new_raw_features:
        sub = fi_new[fi_new["Feature"] == feat]
        if len(sub):
            pos = int(sub.iloc[0]["Rank"])
            imp = float(sub.iloc[0]["Importance"])
            alert = "  *** NO en top 30 ***" if pos > 30 else ""
            print(f"    {feat:<45} rank={pos:>3}  imp={imp:.1f}{alert}")
        else:
            print(f"    {feat:<45} *** NO encontrada en feature_importance (puede estar como __rank) ***")

    print(f"\n  Variantes __rank de features nuevas:")
    for feat in new_rank_features:
        sub = fi_new[fi_new["Feature"] == feat]
        if len(sub):
            pos = int(sub.iloc[0]["Rank"])
            imp = float(sub.iloc[0]["Importance"])
            alert = "  *** NO en top 30 ***" if pos > 30 else ""
            print(f"    {feat:<45} rank={pos:>3}  imp={imp:.1f}{alert}")

    # ── FASE 5: ANÁLISIS POR FECHA ────────────────────────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 5: ANÁLISIS POR FECHA (holdout) ===")
    print(SEP)

    sorted_dates = sorted(p20w_new.keys())
    print(f"\n  P@{TOP_K} por semana del holdout (BASELINE vs NEW):\n")
    print(f"  {'Fecha':<14} {'BASELINE':>10} {'NEW':>10} {'Δ':>8}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8}")

    better_weeks, worse_weeks = 0, 0
    best_week_new, best_p_new = None, -1.0
    worst_week_new, worst_p_new = None, 2.0

    for dt in sorted_dates:
        p_base_w = p20w_base.get(dt, np.nan)
        p_new_w  = p20w_new.get(dt, np.nan)
        d_str    = str(dt.date()) if hasattr(dt, "date") else str(dt)[:10]
        delta    = (p_new_w - p_base_w) if (pd.notna(p_new_w) and pd.notna(p_base_w)) else np.nan

        pb_s = f"{p_base_w:.3f}" if pd.notna(p_base_w) else " N/A "
        pn_s = f"{p_new_w:.3f}"  if pd.notna(p_new_w)  else " N/A "
        d_s  = f"{delta:+.3f}"   if pd.notna(delta)    else "  N/A"

        print(f"  {d_str:<14} {pb_s:>10} {pn_s:>10} {d_s:>8}")

        if pd.notna(delta):
            if delta > 0:
                better_weeks += 1
            elif delta < 0:
                worse_weeks += 1
        if pd.notna(p_new_w):
            if p_new_w > best_p_new:
                best_p_new, best_week_new = p_new_w, dt
            if p_new_w < worst_p_new:
                worst_p_new, worst_week_new = p_new_w, dt

    print(f"\n  Semanas que MEJORAN vs baseline : {better_weeks}")
    print(f"  Semanas que EMPEORAN vs baseline: {worse_weeks}")
    best_d  = str(best_week_new.date())  if best_week_new  and hasattr(best_week_new,  "date") else str(best_week_new)[:10]
    worst_d = str(worst_week_new.date()) if worst_week_new and hasattr(worst_week_new, "date") else str(worst_week_new)[:10]
    print(f"  Mejor semana  : {best_d}  P@{TOP_K}={best_p_new:.3f}")
    print(f"  Peor semana   : {worst_d}  P@{TOP_K}={worst_p_new:.3f}")
    print(f"  Rank stability (corr Spearman rank_pred vs rank_actual): {m_new['RankStability']:.4f}")

    # ── FASE 6: PRÓXIMOS PASOS ────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("=== FASE 6: PRÓXIMOS PASOS SUGERIDOS ===")
    print(SEP)

    top_new_feat = max(
        [(f, fi_new[fi_new["Feature"] == f]["Importance"].values[0]
          if len(fi_new[fi_new["Feature"] == f]) else 0.0)
         for f in new_raw_features],
        key=lambda x: x[1],
    )
    worst_new_feat = min(
        [(f, fi_new[fi_new["Feature"] == f]["Importance"].values[0]
          if len(fi_new[fi_new["Feature"] == f]) else 0.0)
         for f in new_raw_features],
        key=lambda x: x[1],
    )

    delta_hold     = m_new["Holdout_P20"] - m_base["Holdout_P20"]
    delta_filtr    = m_abl["Holdout_P20"] - m_base["Holdout_P20"]
    recommend_next = delta_hold > 0.01

    print(f"""
  1. Feature nueva más útil : {top_new_feat[0]}  (importance={top_new_feat[1]:.1f})
     Feature nueva menos útil: {worst_new_feat[0]}  (importance={worst_new_feat[1]:.1f})
     → Candidata a descartar/revisar si importance ≈ 0.

  2. Contribución de filtros SOLO (ablation):
     Holdout P@{TOP_K} baseline={m_base['Holdout_P20']:.3f}  ablation={m_abl['Holdout_P20']:.3f}
     Δ filtros={delta_filtr:+.3f}
     {'→ Los filtros SOLOS mueven la aguja positivamente.' if delta_filtr > 0 else
      '→ Los filtros solos NO mejoran el holdout. El gain viene de las features.'}

  3. Comparación full:
     Δ NEW vs BASELINE = {delta_hold:+.3f}  (sobre holdout de {core.HOLDOUT_N_WEEKS_ACTUAL} semanas)
     {'→ Resultado POSITIVO. Recomiendo avanzar a IVOL_FF3 + Residual_Reversal.' if recommend_next else
      '→ Resultado NEUTRAL o NEGATIVO. Revisar antes de agregar más features.'}

  4. ¿Avanzar con IVOL_FF3 + Residual_Reversal?
     {'SÍ — el experimento muestra ganancia real (' + f'{delta_hold:+.3f}' + '). '
      'Ambas features requieren retornos residuales (Fama-French 3F), lo que implica '
      'descargar datos de factores. Si el gain se mantiene en más semanas de holdout, '
      'el costo computacional se justifica.' if recommend_next else
      'NO aún — el gain del MVP no es suficientemente robusto (' + f'{delta_hold:+.3f}' + '). '
      'Primero investigar por qué las features nuevas no ayudan más: '
      '¿datos de volumen ausentes? ¿lookback insuficiente con datos semanales?'}
""")

    print(f"\n{SEP}")
    print("PIPELINE v5.0 — COMPLETADO")
    print(f"  BASELINE  : {OUT_FILE_BASELINE}")
    print(f"  ABLATION  : {OUT_FILE_ABLATION}")
    print(f"  NEW       : {OUT_FILE_NEW}")
    print(f"  Input hash: {INPUT_HASH}")
    print(SEP)


if __name__ == "__main__":
    main()
