# -*- coding: utf-8 -*-
"""
Target3 SUPERADOR v2.4 (DOBLE EXPORT): Full + MinPrice Filter
-------------------------------------------------------------
Exporta 2 Excels:
1) Universo completo (igual al actual)
2) Universo filtrado por precio minimo configurable para:
   - entrenamiento / regresion
   - scoring final del ultimo cierre

Filtro configurable:
    MIN_PRICE_FILTER = 2.0
"""

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRanker

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")
FILE_NAME = "Consolidado_100_semanas_paste_todos.xlsx"
FILE_PATH = BASE_DIR / FILE_NAME

TOP_K = 15

# Walk-forward
N_FOLDS = 6
EMBARGO_DATES = 1
MIN_TRAIN_DATES = 35
TEST_DATES_PER_FOLD = None

# Time-decay
USE_TIME_DECAY = True
HALF_LIFE_WEEKS = 26

# Ultimo cierre
MAX_NAN_FRAC_LAST = 0.55

# Blend weights iniciales
W_CLF_INIT = 0.65
W_RNK_INIT = 0.35

SEED = 42
np.random.seed(SEED)

# ===== NUEVO: filtro modificable =====
MIN_PRICE_FILTER = 5.0

# Si queres que el filtro se aplique SOLO al entrenamiento y NO al scoring final,
# pone esto en False. Asi como esta, se aplica a ambos en el excel filtrado.
APPLY_PRICE_FILTER_TO_LAST_CLOSE = True

DATE_STAMP = datetime.now().strftime('%Y-%m-%d')
OUT_FILE_FULL = BASE_DIR / f"Predicciones_Target3_SUPERADOR_v2_4_FULL_{DATE_STAMP}.xlsx"
OUT_FILE_FILTER = BASE_DIR / f"Predicciones_Target3_SUPERADOR_v2_4_MINPRICE_{str(MIN_PRICE_FILTER).replace('.', '_')}_{DATE_STAMP}.xlsx"

# =========================
# HELPERS
# =========================
def clean_colname(c: str) -> str:
    c = c.strip()
    c = re.sub(r"[^\w]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c

def ensure_numeric(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def time_decay_weights(date_series, half_life_weeks=26):
    d = pd.to_datetime(date_series)
    dmax = d.max()
    age_weeks = (dmax - d).dt.days / 7.0
    w = np.exp(-np.log(2.0) * (age_weeks / float(half_life_weeks)))
    return w.astype(float)

def make_walkforward_splits(unique_dates, n_folds=6, embargo=1, min_train_dates=35, test_dates_per_fold=None):
    dates = pd.Index(pd.to_datetime(sorted(pd.Series(unique_dates).dropna().unique())))
    n = len(dates)

    if n < (min_train_dates + 5):
        cut = int(n * 0.8)
        return [(dates[:cut], dates[cut:])]

    if test_dates_per_fold is None:
        test_dates_per_fold = max(3, n // (n_folds * 2))

    splits = []
    max_train_end = n - (embargo + test_dates_per_fold)
    if max_train_end <= min_train_dates:
        cut = int(n * 0.8)
        return [(dates[:cut], dates[cut:])]

    train_ends = np.linspace(min_train_dates, max_train_end, n_folds).astype(int)

    for te in train_ends:
        train_end = int(te)
        test_start = train_end + embargo
        test_end = min(n, test_start + test_dates_per_fold)

        tr_dates = dates[:train_end]
        va_dates = dates[test_start:test_end]

        if len(tr_dates) < min_train_dates or len(va_dates) < 3:
            continue
        splits.append((tr_dates, va_dates))

    if not splits:
        cut = int(n * 0.8)
        splits = [(dates[:cut], dates[cut:])]
    return splits

def precision_at_k_by_date(df_pred, date_col, y_true_col, score_col, k=15):
    precs = []
    for d, g in df_pred.groupby(date_col):
        g = g.sort_values(score_col, ascending=False)
        top = g.head(k)
        if len(top) == 0:
            continue
        precs.append(top[y_true_col].mean())
    return float(np.mean(precs)) if precs else np.nan

def safe_median_best(iters, default_val):
    vals = [i for i in iters if isinstance(i, (int, np.integer)) and i and i > 10]
    return int(np.median(vals)) if vals else default_val

def build_clf(spw, seed=42):
    return LGBMClassifier(
        objective="binary",
        n_estimators=6000,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=0.6,
        min_child_samples=70,
        scale_pos_weight=spw,
        n_jobs=-1,
        random_state=seed
    )

def build_ranker(seed=42):
    return LGBMRanker(
        objective="lambdarank",
        n_estimators=2500,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=60,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=0.8,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1
    )

def apply_price_filter(dataframe, price_col, min_price=None):
    if min_price is None:
        return dataframe.copy()
    return dataframe[pd.to_numeric(dataframe[price_col], errors="coerce") > float(min_price)].copy()

# =========================
# 1) LOAD + NORMALIZAR NOMBRES
# =========================
df = pd.read_excel(FILE_PATH)
print(f"Datos cargados: {df.shape}")

df.columns = [clean_colname(c) for c in df.columns]

TICK_COL = "Ticker"
DATE_COL = "Data_Date"
PRICE_COL = "Close" if "Close" in df.columns else ("Precio" if "Precio" in df.columns else None)
if PRICE_COL is None:
    raise KeyError("No encuentro columna Close ni Precio.")

YCOL = "Target3" if "Target3" in df.columns else None
if YCOL is None:
    raise KeyError("No encuentro columna Target3 en el Excel.")

BASELINE_COL = "Predicted_Outperform_Prob" if "Predicted_Outperform_Prob" in df.columns else None

for c in [TICK_COL, DATE_COL, PRICE_COL, YCOL]:
    if c not in df.columns:
        raise KeyError(f"Falta columna '{c}'.")

df[TICK_COL] = df[TICK_COL].astype(str).str.upper().str.strip()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df["_DateKey"] = df[DATE_COL].dt.normalize()

df = ensure_numeric(df, [PRICE_COL, YCOL])
df[YCOL] = df[YCOL].apply(lambda x: np.nan if pd.isna(x) else (1 if float(x) >= 0.5 else 0))
df["y_t3"] = df[YCOL].astype(float)

df = df.sort_values([TICK_COL, DATE_COL]).reset_index(drop=True)

# =========================
# 2) FEATURES
# =========================
leak_cols = set(["Precio_una_semana", "Ret_Una_sem", "Target", "Target_real", "Predicted_Outperform_Prob"])

candidates = [
    "ADX_14","ATRp_14","Aroon_Diff_25","CCI_20_0_015",
    "Consecutive_Volume_Growth","Cross_Signal",
    "Days_since_Death","Days_since_Golden","Drawdown_52w",
    "FIB_Range_90D","MACD_Hist_Slope_5","MACDh_12_26_9",
    "MA_SLOPE_20","MFI_14","OBV_trend","PctB",
    "Pct_in_52w_range","RET_1M","RET_3M","RET_6M",
    "RS1M_SPY","RS3M_SPY","RS6M_SPY",
    "RSI_14","Ret13_Ratio","SMA_200","SMA_50",
    "STOCH_Cross","VROC_14","Vol_Rel_20","Vol_StreakUp",
    "MA50_over_MA200","Close_over_MA200","Price_gt_MA200",
    "Trend_Score","Momentum_Score","RelStr_Score","RiskPos_Score",
    "Setup_Tag","NearHigh_NoVol","ScoreSimple","Ret_Ultimos_dias",
    "SPY_RET_1M","SPY_RET_3M","SPY_RET_6M","SPY_Ret",
    "Min","Max"
]
base_features = [c for c in candidates if (c in df.columns and c not in leak_cols)]
df = ensure_numeric(df, base_features)

# FE segura desde Close
df["Close_prev"] = df.groupby(TICK_COL)[PRICE_COL].shift(1)
df["Ret_1w_calc"] = (df[PRICE_COL] / df["Close_prev"]) - 1.0
df["LogRet_1w"] = np.log(df[PRICE_COL] / df["Close_prev"])
df["GapPct"] = (df[PRICE_COL] - df["Close_prev"]) / df["Close_prev"]
df["GapPct"] = df["GapPct"].replace([np.inf, -np.inf], np.nan)

df["Vol_12w"] = (
    df.groupby(TICK_COL)["LogRet_1w"]
      .rolling(12, min_periods=6)
      .std()
      .reset_index(0, drop=True)
)

eps = 1e-6
for c in ["RET_3M", "RET_6M"]:
    if c not in df.columns:
        df[c] = np.nan
df = ensure_numeric(df, ["RET_3M", "RET_6M"])
df["RiskAdjMom_3M"] = df["RET_3M"] / (df["Vol_12w"] + eps)
df["RiskAdjMom_6M"] = df["RET_6M"] / (df["Vol_12w"] + eps)

fe_new = ["GapPct", "Ret_1w_calc", "Vol_12w", "RiskAdjMom_3M", "RiskAdjMom_6M"]
df = ensure_numeric(df, fe_new)

# Cross-sectional ranks
xs_pick = [c for c in [
    "RET_1M","RET_3M","RET_6M",
    "RS1M_SPY","RS3M_SPY","RS6M_SPY",
    "Momentum_Score","RelStr_Score",
    "RiskAdjMom_3M","RiskAdjMom_6M",
    "Vol_Rel_20","Drawdown_52w","GapPct","Vol_12w"
] if c in df.columns]

xs_rank_cols = []
for c in xs_pick[:25]:
    rcol = f"{c}__rank"
    df[rcol] = df.groupby("_DateKey")[c].rank(pct=True)
    xs_rank_cols.append(rcol)

feature_cols = [c for c in (base_features + fe_new + xs_rank_cols) if c in df.columns and c not in leak_cols]
feature_cols = list(dict.fromkeys(feature_cols))
print(f"Total features finales: {len(feature_cols)}")

# =========================
# ANCHOR DATE GLOBAL
# =========================
if "SPY" in set(df[TICK_COL].unique()):
    GLOBAL_ANCHOR_DATE = df.loc[df[TICK_COL] == "SPY", "_DateKey"].dropna().max()
else:
    GLOBAL_ANCHOR_DATE = df["_DateKey"].dropna().max()

print(f"Anchor global detectado: {GLOBAL_ANCHOR_DATE.date()}")

# =========================
# CORE PIPELINE
# =========================
def run_pipeline(df_all, min_price_filter=None, out_file=None, run_label="FULL"):
    print("\n" + "=" * 80)
    print(f"RUN: {run_label}")
    print("=" * 80)

    # -------- entrenamiento --------
    df_train_base = apply_price_filter(df_all, PRICE_COL, min_price_filter)

    train_df = df_train_base[df_train_base["y_t3"].notna()].copy()
    train_df = train_df.dropna(subset=["_DateKey"]).copy()
    train_df["y_t3_int"] = train_df["y_t3"].astype(int)
    train_df = train_df.reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"[{run_label}] No hay filas de entrenamiento luego del filtro de precio.")

    print(f"[{run_label}] Filas entrenamiento: {len(train_df):,}")
    print(f"[{run_label}] Tickers entrenamiento: {train_df[TICK_COL].nunique():,}")

    unique_dates = sorted(train_df["_DateKey"].unique())
    splits = make_walkforward_splits(
        unique_dates,
        N_FOLDS,
        EMBARGO_DATES,
        MIN_TRAIN_DATES,
        TEST_DATES_PER_FOLD
    )
    print(f"[{run_label}] Splits walk-forward efectivos: {len(splits)}")

    oof_clf = np.full(len(train_df), np.nan, dtype=float)
    oof_rnk = np.full(len(train_df), np.nan, dtype=float)

    best_iters_clf, best_iters_rnk = [], []
    fold_rows = []

    W_CLF = W_CLF_INIT
    W_RNK = W_RNK_INIT

    for k, (tr_dates, te_dates) in enumerate(splits, start=1):
        tr = train_df[train_df["_DateKey"].isin(tr_dates)].copy()
        te = train_df[train_df["_DateKey"].isin(te_dates)].copy()

        if len(tr) < 3000 or len(te) < 500:
            print(f"[{run_label}] Fold {k}: skip (tr={len(tr)}, te={len(te)})")
            continue

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(tr[feature_cols].replace([np.inf, -np.inf], np.nan).values)
        X_te = imp.transform(te[feature_cols].replace([np.inf, -np.inf], np.nan).values)

        y_tr = tr["y_t3_int"].values
        y_te = te["y_t3_int"].values

        w_tr = None
        if USE_TIME_DECAY:
            w_tr = time_decay_weights(tr["_DateKey"], HALF_LIFE_WEEKS).values

        pos = max(1, int((y_tr == 1).sum()))
        neg = max(1, int((y_tr == 0).sum()))
        spw = neg / pos

        # --- CLF ---
        clf = build_clf(spw, seed=SEED)
        clf.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_te, y_te)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        p_te = clf.predict_proba(X_te)[:, 1]
        oof_clf[te.index.values] = p_te
        best_iters_clf.append(getattr(clf, "best_iteration_", None))

        # --- RANKER ---
        tr_sorted = tr.sort_values(["_DateKey", TICK_COL]).copy()
        te_sorted = te.sort_values(["_DateKey", TICK_COL]).copy()

        g_tr = tr_sorted.groupby("_DateKey").size().values
        g_te = te_sorted.groupby("_DateKey").size().values

        X_tr_r = imp.transform(tr_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values)
        X_te_r = imp.transform(te_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values)

        y_tr_rel = (tr_sorted["y_t3_int"] * 2).values
        y_te_rel = (te_sorted["y_t3_int"] * 2).values

        rnk = build_ranker(seed=SEED)
        rnk.fit(
            X_tr_r, y_tr_rel,
            group=g_tr,
            eval_set=[(X_te_r, y_te_rel)],
            eval_group=[g_te],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        s_te = rnk.predict(X_te_r)
        s_te_series = pd.Series(s_te, index=te_sorted.index)
        oof_rnk[te.index.values] = s_te_series.loc[te.index.values].values
        best_iters_rnk.append(getattr(rnk, "best_iteration_", None))

        te_eval = te[["_DateKey", "y_t3_int"]].copy()
        te_eval["Prob_Clf"] = p_te
        te_eval["Score_Ranker"] = oof_rnk[te.index.values]
        te_eval["RankerPct"] = te_eval.groupby("_DateKey")["Score_Ranker"].rank(pct=True)
        te_eval["RankScore"] = W_CLF * te_eval["Prob_Clf"] + W_RNK * te_eval["RankerPct"]

        auc = roc_auc_score(y_te, p_te)
        ap = average_precision_score(y_te, p_te)
        p15 = precision_at_k_by_date(te_eval, "_DateKey", "y_t3_int", "RankScore", k=TOP_K)

        fold_rows.append([k, len(tr), len(te), auc, ap, p15])
        print(f"[{run_label}] Fold {k}: AUC={auc:.4f} PR-AUC={ap:.4f} | P@{TOP_K}={p15:.3f}")

    mask = np.isfinite(oof_clf) & np.isfinite(oof_rnk)
    if mask.sum() == 0:
        raise ValueError(f"[{run_label}] No hubo predicciones OOF validas. Revisa el filtro o el tamano de la muestra.")

    y_oof = train_df.loc[mask, "y_t3_int"].values

    df_oof = train_df.loc[mask, ["_DateKey"]].copy()
    df_oof["y"] = y_oof
    df_oof["Prob_Clf"] = oof_clf[mask]
    df_oof["Score_Ranker"] = oof_rnk[mask]
    df_oof["RankerPct"] = df_oof.groupby("_DateKey")["Score_Ranker"].rank(pct=True)
    df_oof["RankScore"] = W_CLF * df_oof["Prob_Clf"] + W_RNK * df_oof["RankerPct"]

    # busqueda de mejores pesos
    best_w = None
    best_p15 = -1

    for w in np.arange(0.35, 0.86, 0.05):
        score = w * df_oof["Prob_Clf"] + (1 - w) * df_oof["RankerPct"]
        tmp = df_oof.copy()
        tmp["ScoreTmp"] = score
        p15 = precision_at_k_by_date(tmp, "_DateKey", "y", "ScoreTmp", k=TOP_K)
        if p15 > best_p15:
            best_p15 = p15
            best_w = float(w)

    W_CLF = best_w
    W_RNK = 1.0 - best_w
    df_oof["RankScore"] = W_CLF * df_oof["Prob_Clf"] + W_RNK * df_oof["RankerPct"]

    print(f"[{run_label}] [PASS] Best weights OOF: W_CLF={W_CLF:.2f} | W_RNK={W_RNK:.2f} | P@{TOP_K}={best_p15:.3f}")

    # calibracion Platt
    platt = LogisticRegression(max_iter=2000, random_state=SEED)
    platt.fit(df_oof[["RankScore"]].values, y_oof)
    df_oof["Prob_T3_FINAL"] = platt.predict_proba(df_oof[["RankScore"]].values)[:, 1]

    auc_oof = roc_auc_score(y_oof, df_oof["Prob_Clf"].values)
    ap_oof = average_precision_score(y_oof, df_oof["Prob_Clf"].values)
    base_rate = y_oof.mean()
    p_at_k = precision_at_k_by_date(df_oof, "_DateKey", "y", "RankScore", k=TOP_K)

    print(f"[{run_label}] OOF_AUC={auc_oof:.4f}")
    print(f"[{run_label}] OOF_PR_AUC={ap_oof:.4f}")
    print(f"[{run_label}] BaseRate={base_rate:.2%}")
    print(f"[{run_label}] P@{TOP_K}={p_at_k:.3f}")

    # -------- fit final --------
    n_estim_clf = safe_median_best(best_iters_clf, 2500)
    n_estim_rnk = safe_median_best(best_iters_rnk, 1200)

    imp_final = SimpleImputer(strategy="median")
    X_full = imp_final.fit_transform(train_df[feature_cols].replace([np.inf, -np.inf], np.nan).values)
    y_full = train_df["y_t3_int"].values

    w_full = None
    if USE_TIME_DECAY:
        w_full = time_decay_weights(train_df["_DateKey"], HALF_LIFE_WEEKS).values

    pos = max(1, int((y_full == 1).sum()))
    neg = max(1, int((y_full == 0).sum()))
    spw_full = neg / pos

    clf_final = LGBMClassifier(
        objective="binary",
        n_estimators=n_estim_clf,
        learning_rate=0.02,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.3,
        reg_lambda=0.6,
        min_child_samples=70,
        scale_pos_weight=spw_full,
        n_jobs=-1,
        random_state=SEED
    )
    clf_final.fit(X_full, y_full, sample_weight=w_full)

    train_sorted = train_df.sort_values(["_DateKey", TICK_COL]).copy()
    g_full = train_sorted.groupby("_DateKey").size().values
    X_full_r = imp_final.transform(train_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values)
    y_full_rel = (train_sorted["y_t3_int"] * 2).values

    rnk_final = LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estim_rnk,
        learning_rate=0.03,
        num_leaves=63,
        min_data_in_leaf=60,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1
    )
    rnk_final.fit(X_full_r, y_full_rel, group=g_full)

    # -------- ultimo cierre --------
    if min_price_filter is not None and APPLY_PRICE_FILTER_TO_LAST_CLOSE:
        df_score_base = apply_price_filter(df_all, PRICE_COL, min_price_filter)
    else:
        df_score_base = df_all.copy()

    df_last = df_score_base[df_score_base["_DateKey"] == GLOBAL_ANCHOR_DATE].copy()
    df_last = df_last.sort_values([TICK_COL, DATE_COL]).drop_duplicates(subset=[TICK_COL], keep="last")

    total_tickers = df_score_base[TICK_COL].nunique()

    if df_last[TICK_COL].nunique() < 0.5 * max(1, total_tickers):
        df_last = (
            df_score_base
            .sort_values([TICK_COL, DATE_COL])
            .groupby(TICK_COL)
            .tail(1)
            .copy()
        )
        df_last = df_last.sort_values([TICK_COL, DATE_COL]).drop_duplicates(subset=[TICK_COL], keep="last")

    last_by_ticker = df_score_base.dropna(subset=["_DateKey"]).groupby(TICK_COL)["_DateKey"].max()
    missing = last_by_ticker[last_by_ticker < GLOBAL_ANCHOR_DATE].sort_values()
    missing_df = missing.reset_index()
    missing_df.columns = [TICK_COL, "Last_Available_Date"]
    missing_df["Expected_Last_Close"] = GLOBAL_ANCHOR_DATE

    nan_frac_last = df_last[feature_cols].isna().mean(axis=1)
    df_last = df_last[nan_frac_last <= MAX_NAN_FRAC_LAST].copy()

    if len(df_last) == 0:
        raise ValueError(f"[{run_label}] No quedaron tickers en el ultimo cierre luego de filtros.")

    X_last = imp_final.transform(df_last[feature_cols].replace([np.inf, -np.inf], np.nan).values)
    p_last = clf_final.predict_proba(X_last)[:, 1]

    df_last_sorted = df_last.sort_values(["_DateKey", TICK_COL]).copy()
    X_last_r = imp_final.transform(df_last_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values)
    s_last = rnk_final.predict(X_last_r)
    s_last_series = pd.Series(s_last, index=df_last_sorted.index)
    s_last_aligned = s_last_series.loc[df_last.index].values

    out = df_last[[TICK_COL, DATE_COL, PRICE_COL]].copy()

    if BASELINE_COL and BASELINE_COL in df_last.columns:
        out["Baseline_Predicted_Outperform_Prob"] = pd.to_numeric(df_last[BASELINE_COL], errors="coerce")

    out["Prob_Clf"] = p_last
    out["Score_Ranker"] = s_last_aligned
    out["RankerPct"] = out["Score_Ranker"].rank(pct=True)
    out["RankScore"] = W_CLF * out["Prob_Clf"] + W_RNK * out["RankerPct"]
    out["Prob_T3_FINAL"] = platt.predict_proba(out[["RankScore"]].values)[:, 1]

    sort_cols = ["RankScore", "Prob_Clf", "RankerPct"]
    asc = [False, False, False]
    if "Baseline_Predicted_Outperform_Prob" in out.columns:
        sort_cols.append("Baseline_Predicted_Outperform_Prob")
        asc.append(False)
    sort_cols.append(TICK_COL)
    asc.append(True)

    out = out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    topk = out.head(TOP_K).copy()

    fold_df = pd.DataFrame(
        fold_rows,
        columns=["Fold", "TrainRows", "TestRows", "AUC", "PR_AUC", f"P@{TOP_K}_RankScore"]
    )

    summary = pd.DataFrame({
        "Metric": [
            "RunLabel",
            "MinPriceFilter",
            "OOF_AUC(Prob_Clf)",
            "OOF_PR_AUC(Prob_Clf)",
            f"OOF_P@{TOP_K}_RankScore_by_date",
            "BaseRate_Target3",
            "AnchorDate",
            "LastClose_Tickers_Used",
            "Total_Tickers_Scored",
            "TrainRows",
            "TrainTickers",
            "Weights_W_CLF",
            "Weights_W_RNK"
        ],
        "Value": [
            run_label,
            "None" if min_price_filter is None else float(min_price_filter),
            float(auc_oof),
            float(ap_oof),
            float(p_at_k),
            float(base_rate),
            str(GLOBAL_ANCHOR_DATE.date()),
            int(len(out)),
            int(total_tickers),
            int(len(train_df)),
            int(train_df[TICK_COL].nunique()),
            float(W_CLF),
            float(W_RNK)
        ]
    })

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        topk.to_excel(writer, sheet_name=f"Top_{TOP_K}_Last", index=False)
        out.to_excel(writer, sheet_name="All_Last", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        fold_df.to_excel(writer, sheet_name="Fold_Metrics", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_LastClose", index=False)

    print(f"\n[{run_label}] [PASS] Export OK: {out_file}")
    print(f"[{run_label}] Anchor date: {GLOBAL_ANCHOR_DATE.date()}")
    print(f"[{run_label}] Top {TOP_K}:")
    print(topk[["Rank", TICK_COL, PRICE_COL, "RankScore", "Prob_T3_FINAL", "Prob_Clf", "RankerPct"]].to_string(index=False))

# =========================
# 6) EJECUCION DOBLE
# =========================
run_pipeline(
    df_all=df,
    min_price_filter=None,
    out_file=OUT_FILE_FULL,
    run_label="FULL_UNIVERSE"
)

run_pipeline(
    df_all=df,
    min_price_filter=MIN_PRICE_FILTER,
    out_file=OUT_FILE_FILTER,
    run_label=f"MIN_PRICE_GT_{MIN_PRICE_FILTER}"
)

print("\n" + "=" * 80)
print("PROCESO TERMINADO")
print(f"Archivo 1: {OUT_FILE_FULL}")
print(f"Archivo 2: {OUT_FILE_FILTER}")
print("=" * 80)
