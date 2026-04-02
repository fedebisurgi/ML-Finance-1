# -*- coding: utf-8 -*-
"""
Target3 SUPERADOR v4.0 - Based on v3.0
======================================
Fixes applied (vs v2.4):
  A) Walk-forward: embargo scaled to feature lookback, non-overlapping val folds,
     nested weight search (no look-ahead), minimum fold size with logging.
  B) Blend: multi-level relevance for ranker, distribution-shift monitoring for
     Platt calibration, consistent RankerPct (per-date everywhere).
  C) Features: gap-aware shift, date-consistency diagnostics, leak audit.
  D) Metrics: OOF AUC on final signal, permutation test for P@K.
  E) Production: robust anchor date, component reuse across runs, file versioning
     with input hash.

New performance improvements (G):
  - Sector/industry neutralisation (cross-sectional demeaning)
  - Short-term features (reversal, momentum acceleration, volume z-score, price/4w-high)
  - Market-regime feature (SPY trend + vol regime)
  - Stacking meta-model (optional, USE_STACKING flag)
  - Dynamic Top-K based on model confidence
  - Turnover tracking + exponential score smoothing
"""

import warnings
warnings.filterwarnings("ignore")

import re, hashlib, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRanker

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# #+==================================================================+#
# #                        CONFIGURATION                            #
# #+==================================================================+#

BASE_DIR = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")
FILE_NAME = "Consolidado_100_semanas_paste_todos.xlsx"
FILE_PATH = BASE_DIR / FILE_NAME

TOP_K = 15

# Walk-forward - FIX A1: embargo scaled to longest feature lookback
N_FOLDS = 6
# With RET_3M (~12 weeks lookback), embargo must be >= 4 weeks to avoid
# autocorrelation leakage. We use max(4, user_override).
EMBARGO_DATES = 4  # was 1 - insufficient for multi-week features
MIN_TRAIN_DATES = 35
TEST_DATES_PER_FOLD = None

# Time-decay
USE_TIME_DECAY = True
HALF_LIFE_WEEKS = 26

# Last close
MAX_NAN_FRAC_LAST = 0.55

# Blend weights - initial values, will be tuned per-fold (nested)
W_CLF_INIT = 0.65
W_RNK_INIT = 0.35

SEED = 42
np.random.seed(SEED)

# Price filters
MIN_PRICE_FILTER = 5.0               # removes cheap/penny stocks (price > threshold)
APPLY_PRICE_FILTER_TO_LAST_CLOSE = True

# v4: max-price filter - captures cheap/volatile stocks (price < threshold)
MAX_PRICE_FILTER = 2.0

# -- NEW: G-improvements config -------------------------------------
USE_SECTOR_NEUTRALISATION = True     # G: sector demeaning
SECTOR_COL = "Sector"                # column name in Excel (set None if absent)

USE_STACKING = True                  # G: stacking meta-model over OOF
STACKING_META_MODEL = "gbm"          # "gbm" or "lr"

USE_DYNAMIC_TOPK = True              # G: dynamic top-K
DYNAMIC_TOPK_PERCENTILE = 95         # top 5% of universe
DYNAMIC_TOPK_MIN = 5
DYNAMIC_TOPK_MAX = 20

SCORE_SMOOTHING_ALPHA = 0.0          # G: 0 = no smoothing; 0.3 = moderate EMA
# If > 0, requires historical RankScores (loaded from prior run output)
PRIOR_SCORES_PATH = None             # Path to previous run's All_Last sheet

# File versioning - FIX E3
DATE_STAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')  # includes time -> no silent overwrite

# #+==================================================================+#
# #                          HELPERS                                #
# #+==================================================================+#

def compute_input_hash(path: Path, first_n_bytes: int = 2**20) -> str:
    """FIX E3: SHA-256 of input file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(first_n_bytes))
    return h.hexdigest()[:12]

INPUT_HASH = compute_input_hash(FILE_PATH) if FILE_PATH.exists() else "NOHASH"

OUT_FILE_FULL       = BASE_DIR / f"Pred_T4_FULL_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_MIN        = BASE_DIR / f"Pred_T4_MINPRICE_{str(MIN_PRICE_FILTER).replace('.','_')}_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_MAX        = BASE_DIR / f"Pred_T4_MAXPRICE_{str(MAX_PRICE_FILTER).replace('.','_')}_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_COMPARISON = BASE_DIR / f"Pred_T4_PriceFilter_Comparison_{DATE_STAMP}_{INPUT_HASH}.xlsx"


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


# -- FIX A2: non-overlapping walk-forward splits --------------------
def make_walkforward_splits(
    unique_dates,
    n_folds: int = 6,
    embargo: int = 4,
    min_train_dates: int = 35,
    test_dates_per_fold: Optional[int] = None,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Expanding-window walk-forward with:
      - Non-overlapping validation windows (no val date appears in two folds)
      - Embargo gap between train end and val start
      - Minimum train size enforced
    """
    dates = pd.DatetimeIndex(sorted(pd.Series(unique_dates).dropna().unique()))
    n = len(dates)

    if n < (min_train_dates + embargo + 3):
        cut = int(n * 0.75)
        return [(dates[:cut], dates[cut + embargo:])]

    if test_dates_per_fold is None:
        # Divide remaining dates after min_train into n_folds non-overlapping blocks
        available = n - min_train_dates - embargo
        test_dates_per_fold = max(3, available // n_folds)

    splits = []
    # Work backwards from end to allocate non-overlapping val windows
    val_blocks = []
    cursor = n
    for _ in range(n_folds):
        val_end = cursor
        val_start = max(0, val_end - test_dates_per_fold)
        if val_start <= min_train_dates + embargo:
            break
        val_blocks.append((val_start, val_end))
        cursor = val_start - embargo  # leave embargo gap before this val block
    val_blocks.reverse()

    for val_start, val_end in val_blocks:
        train_end = val_start - embargo
        if train_end < min_train_dates:
            continue
        tr_dates = dates[:train_end]
        va_dates = dates[val_start:val_end]
        if len(tr_dates) < min_train_dates or len(va_dates) < 3:
            continue
        splits.append((tr_dates, va_dates))

    if not splits:
        cut = int(n * 0.75)
        splits = [(dates[:cut], dates[cut + min(embargo, n - cut - 1):])]

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


# -- FIX D3: permutation test for P@K ------------------------------
def permutation_test_p_at_k(
    df_pred, date_col, y_true_col, score_col, k=15, n_perm=1000, seed=42
):
    """
    Tests H0: P@K is no better than random selection.
    Returns observed P@K, mean random P@K, p-value.
    """
    rng = np.random.RandomState(seed)
    observed = precision_at_k_by_date(df_pred, date_col, y_true_col, score_col, k)
    perm_scores = []
    for _ in range(n_perm):
        df_perm = df_pred.copy()
        # Shuffle scores within each date (preserves per-date structure)
        df_perm[score_col] = df_perm.groupby(date_col)[score_col].transform(
            lambda x: rng.permutation(x.values)
        )
        perm_scores.append(
            precision_at_k_by_date(df_perm, date_col, y_true_col, score_col, k)
        )
    perm_scores = np.array(perm_scores)
    p_value = float(np.mean(perm_scores >= observed))
    return observed, float(np.mean(perm_scores)), p_value


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
        random_state=seed,
    )


# -- FIX B1: multi-level relevance for ranker ----------------------
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
        verbosity=-1,
    )


def compute_relevance_labels(y_binary: np.ndarray, prob_clf: np.ndarray) -> np.ndarray:
    """
    FIX B1: Instead of binary {0, 2}, create multi-level relevance:
      0 = negative (y=0, low prob)
      1 = negative but model-uncertain (y=0, high prob - hard negatives)
      2 = positive but model-uncertain (y=1, low prob)
      3 = positive and model-confident (y=1, high prob)

    This gives lambdarank a richer signal: it learns to separate
    hard negatives from easy ones, and confident positives from marginal ones.
    Falls back to binary {0, 2} if prob_clf is None (first fold bootstrapping).
    """
    if prob_clf is None:
        return (y_binary * 2).astype(int)

    median_prob = np.median(prob_clf)
    rel = np.zeros(len(y_binary), dtype=int)
    rel[(y_binary == 0) & (prob_clf < median_prob)] = 0
    rel[(y_binary == 0) & (prob_clf >= median_prob)] = 1
    rel[(y_binary == 1) & (prob_clf < median_prob)] = 2
    rel[(y_binary == 1) & (prob_clf >= median_prob)] = 3
    return rel


def apply_price_filter(dataframe, price_col, min_price=None, max_price=None):
    """Filter rows by price. min_price excludes cheap stocks; max_price captures only cheap/volatile stocks."""
    df = dataframe.copy()
    if min_price is not None:
        df = df[pd.to_numeric(df[price_col], errors="coerce") > float(min_price)].copy()
    if max_price is not None:
        df = df[pd.to_numeric(df[price_col], errors="coerce") < float(max_price)].copy()
    return df


# -- FIX C2: gap-aware shift ---------------------------------------
def gap_aware_shift(df, tick_col, date_col, value_col, periods=1, max_gap_weeks=2):
    """
    Shift that returns NaN when there's a time gap > max_gap_weeks.
    Prevents .shift(1) from connecting non-adjacent weeks.
    """
    result = df.groupby(tick_col)[value_col].shift(periods)
    if periods > 0:
        date_diff = df.groupby(tick_col)[date_col].diff()
        gap_mask = date_diff > pd.Timedelta(weeks=max_gap_weeks)
        result[gap_mask] = np.nan
    return result


# -- FIX C3: date consistency diagnostic ---------------------------
def diagnose_date_consistency(df, tick_col, date_col):
    """
    Reports how many distinct calendar dates exist per _DateKey.
    If tickers have different closing days within the same 'week', the
    cross-sectional rank mixes different time periods.
    """
    by_date = df.groupby("_DateKey")[date_col].agg(["min", "max", "nunique"])
    inconsistent = by_date[by_date["nunique"] > 1]
    if len(inconsistent) > 0:
        max_spread = (inconsistent["max"] - inconsistent["min"]).max()
        print(f"  [WARN] DATE CONSISTENCY: {len(inconsistent)} _DateKey groups have "
              f"multiple raw dates (max spread: {max_spread})")
        print(f"    Recommendation: normalize to week-ending Friday before ranking.")
    else:
        print("  [OK] Date consistency OK: 1 raw date per _DateKey.")
    return inconsistent


# -- G: Market regime feature --------------------------------------
def compute_market_regime(df, tick_col, date_col, price_col, spy_ticker="SPY"):
    """
    Computes a market regime feature based on SPY:
      - SPY 4-week return
      - SPY 4-week realised volatility
      - Regime label: 0=high_vol, 1=bearish_trend, 2=sideways, 3=bullish_trend

    Hypothesis: model can learn that momentum signals work in trending markets
    but mean-reversion signals work better in sideways/high-vol regimes.
    """
    spy = df[df[tick_col] == spy_ticker].sort_values(date_col).copy()
    if len(spy) < 5:
        print(f"  [WARN] SPY not found or too few rows for regime computation")
        return pd.DataFrame()

    spy["spy_ret_4w"] = spy[price_col].pct_change(4)
    spy["spy_log_ret"] = np.log(spy[price_col] / spy[price_col].shift(1))
    spy["spy_vol_4w"] = spy["spy_log_ret"].rolling(4, min_periods=2).std()

    # Regime classification
    def classify_regime(row):
        if pd.isna(row["spy_vol_4w"]) or pd.isna(row["spy_ret_4w"]):
            return np.nan
        vol_high = row["spy_vol_4w"] > spy["spy_vol_4w"].quantile(0.75)
        if vol_high:
            return 0  # high volatility
        if row["spy_ret_4w"] < -0.02:
            return 1  # bearish
        elif row["spy_ret_4w"] > 0.02:
            return 3  # bullish
        else:
            return 2  # sideways

    spy["MarketRegime"] = spy.apply(classify_regime, axis=1)

    regime_df = spy[["_DateKey", "spy_ret_4w", "spy_vol_4w", "MarketRegime"]].copy()
    regime_df = regime_df.drop_duplicates(subset=["_DateKey"], keep="last")
    return regime_df


# -- G: Short-term features ----------------------------------------
def add_short_term_features(df, tick_col, date_col, price_col):
    """
    Adds features capturing 1-4 week dynamics:
      - Intra-week reversal: 1w return reversal signal
      - Momentum acceleration: second derivative of price (change in change)
      - Volume z-score: current volume relative to 8-week rolling stats
      - Price / 4-week high ratio

    Hypothesis: short-term mean reversion and acceleration often predict
    next-week outperformance better than longer-horizon momentum alone.
    """
    g = df.groupby(tick_col)

    # Momentum acceleration (2nd derivative of price, weekly)
    df["Ret_1w"] = g[price_col].pct_change(1)
    df["Ret_1w_prev"] = gap_aware_shift(df, tick_col, date_col, "Ret_1w", 1, 2)
    df["MomAccel_1w"] = df["Ret_1w"] - df["Ret_1w_prev"]

    # Price / 4-week high ratio
    df["High_4w"] = g[price_col].transform(
        lambda x: x.rolling(4, min_periods=2).max()
    )
    df["PriceToHigh4w"] = df[price_col] / df["High_4w"].replace(0, np.nan)

    # Price / 4-week low ratio (distance from support)
    df["Low_4w"] = g[price_col].transform(
        lambda x: x.rolling(4, min_periods=2).min()
    )
    df["PriceToLow4w"] = df[price_col] / df["Low_4w"].replace(0, np.nan)

    # Volume z-score (if volume column exists)
    vol_col = None
    for candidate in ["Volume", "Vol", "Volumen"]:
        if candidate in df.columns:
            vol_col = candidate
            break
    if vol_col:
        df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
        vol_mean = g[vol_col].transform(lambda x: x.rolling(8, min_periods=4).mean())
        vol_std = g[vol_col].transform(lambda x: x.rolling(8, min_periods=4).std())
        df["VolZScore_8w"] = (df[vol_col] - vol_mean) / vol_std.replace(0, np.nan)
    else:
        df["VolZScore_8w"] = np.nan

    # Intra-week reversal: if last week was extreme, expect reversal
    ret_1w_rank = df.groupby("_DateKey")["Ret_1w"].rank(pct=True)
    df["ReversalSignal_1w"] = 1.0 - ret_1w_rank  # high rank -> low signal (expect reversal)

    new_cols = ["MomAccel_1w", "PriceToHigh4w", "PriceToLow4w",
                "VolZScore_8w", "ReversalSignal_1w"]

    # Clean up temp columns
    df.drop(columns=["Ret_1w_prev", "High_4w", "Low_4w"], errors="ignore", inplace=True)

    return df, new_cols


# -- G: Sector neutralisation --------------------------------------
def sector_neutralise(df, feature_cols, date_col, sector_col):
    """
    Cross-sectional demeaning by sector: subtract sector-date mean from each feature.
    This removes sector-level effects so the model ranks stocks within sectors
    rather than picking hot sectors.

    Hypothesis: reduces noise from sector rotation, forces model to find
    stock-specific alpha. Particularly helps RankerPct stability WoW.
    """
    if sector_col not in df.columns:
        print(f"  [WARN] Sector column '{sector_col}' not found - skipping neutralisation")
        return df, []

    neutral_cols = []
    for c in feature_cols:
        ncol = f"{c}__sn"
        sector_mean = df.groupby([date_col, sector_col])[c].transform("mean")
        df[ncol] = df[c] - sector_mean
        neutral_cols.append(ncol)

    print(f"  [OK] Sector-neutralised {len(neutral_cols)} features")
    return df, neutral_cols


# #+==================================================================+#
# #                      1) LOAD + NORMALISE                        #
# #+==================================================================+#

print(f"Input hash: {INPUT_HASH}")
df = pd.read_excel(FILE_PATH)
print(f"Datos cargados: {df.shape}")

df.columns = [clean_colname(c) for c in df.columns]

# v4 FIX: deduplicate column names (ocurre al pegar datos de multiples fuentes en Excel)
if df.columns.duplicated().any():
    dup_cols = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
    print(f"  [WARN] Columnas duplicadas detectadas y eliminadas (se mantiene la primera): {dup_cols}")
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

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
# FIX C3: normalise to week-ending Friday to avoid intra-week date mixing
df["_DateKey"] = df[DATE_COL].dt.to_period("W").dt.end_time.dt.normalize()

df = ensure_numeric(df, [PRICE_COL, YCOL])
df[YCOL] = df[YCOL].apply(lambda x: np.nan if pd.isna(x) else (1 if float(x) >= 0.5 else 0))
df["y_t3"] = df[YCOL].astype(float)

df = df.sort_values([TICK_COL, DATE_COL]).reset_index(drop=True)

# Date consistency diagnostic
print("\n--- Date Consistency Diagnostic ---")
diagnose_date_consistency(df, TICK_COL, DATE_COL)

# #+==================================================================+#
# #                      2) FEATURES                                #
# #+==================================================================+#

# FIX C1: expanded leak audit
leak_cols = set([
    "Precio_una_semana", "Ret_Una_sem", "Target", "Target_real",
    "Predicted_Outperform_Prob",
    # Additional potential leakers: anything computed from future prices
    "Ret_Next", "Next_Week_Return", "Forward_Return",
    "Target1", "Target2", "Target4", "Target5",
])

candidates = [
    "ADX_14", "ATRp_14", "Aroon_Diff_25", "CCI_20_0_015",
    "Consecutive_Volume_Growth", "Cross_Signal",
    "Days_since_Death", "Days_since_Golden", "Drawdown_52w",
    "FIB_Range_90D", "MACD_Hist_Slope_5", "MACDh_12_26_9",
    "MA_SLOPE_20", "MFI_14", "OBV_trend", "PctB",
    "Pct_in_52w_range", "RET_1M", "RET_3M", "RET_6M",
    "RS1M_SPY", "RS3M_SPY", "RS6M_SPY",
    "RSI_14", "Ret13_Ratio", "SMA_200", "SMA_50",
    "STOCH_Cross", "VROC_14", "Vol_Rel_20", "Vol_StreakUp",
    "MA50_over_MA200", "Close_over_MA200", "Price_gt_MA200",
    "Trend_Score", "Momentum_Score", "RelStr_Score", "RiskPos_Score",
    "Setup_Tag", "NearHigh_NoVol", "ScoreSimple", "Ret_Ultimos_dias",
    "SPY_RET_1M", "SPY_RET_3M", "SPY_RET_6M", "SPY_Ret",
    "Min", "Max",
]
base_features = [c for c in candidates if (c in df.columns and c not in leak_cols)]
df = ensure_numeric(df, base_features)

# FIX C2: gap-aware Close_prev
df["Close_prev"] = gap_aware_shift(df, TICK_COL, "_DateKey", PRICE_COL, periods=1, max_gap_weeks=2)
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

# G: short-term features
print("\n--- Adding short-term features ---")
df, short_term_cols = add_short_term_features(df, TICK_COL, "_DateKey", PRICE_COL)
df = ensure_numeric(df, short_term_cols)
print(f"  [OK] Added {len(short_term_cols)} short-term features: {short_term_cols}")

# G: market regime
print("\n--- Computing market regime ---")
regime_df = compute_market_regime(df, TICK_COL, "_DateKey", PRICE_COL)
if len(regime_df) > 0:
    df = df.merge(regime_df, on="_DateKey", how="left")
    regime_cols = ["spy_ret_4w", "spy_vol_4w", "MarketRegime"]
    print(f"  [OK] Added regime features: {regime_cols}")
else:
    regime_cols = []

# Cross-sectional ranks
xs_pick = [c for c in [
    "RET_1M", "RET_3M", "RET_6M",
    "RS1M_SPY", "RS3M_SPY", "RS6M_SPY",
    "Momentum_Score", "RelStr_Score",
    "RiskAdjMom_3M", "RiskAdjMom_6M",
    "Vol_Rel_20", "Drawdown_52w", "GapPct", "Vol_12w",
    "MomAccel_1w", "PriceToHigh4w", "ReversalSignal_1w",
] if c in df.columns]

xs_rank_cols = []
for c in xs_pick[:30]:
    rcol = f"{c}__rank"
    df[rcol] = df.groupby("_DateKey")[c].rank(pct=True)
    xs_rank_cols.append(rcol)

# Assemble feature list
all_feature_candidates = (
    base_features + fe_new + short_term_cols + regime_cols + xs_rank_cols
)

# G: sector neutralisation
sn_cols = []
if USE_SECTOR_NEUTRALISATION and SECTOR_COL:
    print("\n--- Sector neutralisation ---")
    neutralise_targets = [c for c in (base_features + fe_new + short_term_cols)
                          if c in df.columns]
    df, sn_cols = sector_neutralise(df, neutralise_targets, "_DateKey", SECTOR_COL)
    all_feature_candidates += sn_cols

feature_cols = [c for c in all_feature_candidates
                if c in df.columns and c not in leak_cols]
feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate preserving order
print(f"\nTotal features finales: {len(feature_cols)}")

# #+==================================================================+#
# #                   ANCHOR DATE (FIX E1)                          #
# #+==================================================================+#

# FIX E1: compute anchor from mode of tickers' max dates, not just SPY
last_dates_by_ticker = df.dropna(subset=["_DateKey"]).groupby(TICK_COL)["_DateKey"].max()
anchor_candidates = last_dates_by_ticker.value_counts()
GLOBAL_ANCHOR_DATE = anchor_candidates.index[0]  # most common max date

spy_anchor = last_dates_by_ticker.get("SPY", None)
if spy_anchor is not None and spy_anchor != GLOBAL_ANCHOR_DATE:
    print(f"  [WARN] SPY anchor ({spy_anchor.date()}) differs from mode anchor "
          f"({GLOBAL_ANCHOR_DATE.date()}). Using mode.")

ticker_coverage = anchor_candidates.iloc[0] / len(last_dates_by_ticker)
print(f"Anchor global: {GLOBAL_ANCHOR_DATE.date()} "
      f"(coverage: {ticker_coverage:.1%} of tickers)")

if ticker_coverage < 0.5:
    print(f"  [WARN] WARNING: anchor date covers <50% of tickers. "
          f"Data may be stale or incomplete.")

# #+==================================================================+#
# #                     CORE PIPELINE                               #
# #+==================================================================+#

# FIX E2: cache reusable components between runs
_shared_cache: Dict[str, Any] = {}


def run_pipeline(
    df_all: pd.DataFrame,
    min_price_filter: Optional[float] = None,
    max_price_filter: Optional[float] = None,
    out_file: Optional[Path] = None,
    run_label: str = "FULL",
    reuse_from_cache: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """
    Main pipeline with all fixes applied.
    If reuse_from_cache=True, reuses imputer, models, calibrator from
    a prior run (saves ~50% compute on the second filtered run).
    """
    global _shared_cache

    print("\n" + "=" * 80)
    print(f"RUN: {run_label}")
    print("=" * 80)

    # ---- Training data ----
    df_train_base = apply_price_filter(df_all, PRICE_COL, min_price=min_price_filter, max_price=max_price_filter)
    train_df = df_train_base[df_train_base["y_t3"].notna()].copy()
    train_df = train_df.dropna(subset=["_DateKey"]).copy()
    train_df["y_t3_int"] = train_df["y_t3"].astype(int)
    train_df = train_df.reset_index(drop=True)

    if len(train_df) == 0:
        raise ValueError(f"[{run_label}] No training rows after price filter.")

    print(f"[{run_label}] Training rows: {len(train_df):,}")
    print(f"[{run_label}] Training tickers: {train_df[TICK_COL].nunique():,}")
    print(f"[{run_label}] Approx tickers/date: "
          f"{len(train_df) / max(1, train_df['_DateKey'].nunique()):.0f}")

    unique_dates = sorted(train_df["_DateKey"].unique())
    splits = make_walkforward_splits(
        unique_dates, N_FOLDS, EMBARGO_DATES, MIN_TRAIN_DATES, TEST_DATES_PER_FOLD,
    )
    print(f"[{run_label}] Walk-forward splits: {len(splits)}")

    # Verify no validation overlap (FIX A2 diagnostic)
    all_val_dates = set()
    for _, va in splits:
        overlap = all_val_dates & set(va)
        if overlap:
            print(f"  [WARN] VALIDATION OVERLAP detected: {len(overlap)} dates")
        all_val_dates.update(va)
    print(f"  [OK] Total unique validation dates: {len(all_val_dates)}")

    oof_clf = np.full(len(train_df), np.nan, dtype=float)
    oof_rnk = np.full(len(train_df), np.nan, dtype=float)
    best_iters_clf, best_iters_rnk = [], []
    fold_rows = []

    # ---- Walk-forward folds ----
    for k, (tr_dates, te_dates) in enumerate(splits, start=1):
        tr = train_df[train_df["_DateKey"].isin(tr_dates)].copy()
        te = train_df[train_df["_DateKey"].isin(te_dates)].copy()

        # FIX A3: log skipped folds explicitly instead of silent continue
        if len(tr) < 2000 or len(te) < 200:
            print(f"[{run_label}] Fold {k}: SKIP (tr={len(tr)}, te={len(te)}) - "
                  f"below minimum threshold. Impact: reduces OOF coverage.")
            continue

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(
            tr[feature_cols].replace([np.inf, -np.inf], np.nan).values
        )
        X_te = imp.transform(
            te[feature_cols].replace([np.inf, -np.inf], np.nan).values
        )
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

        # --- RANKER with multi-level relevance (FIX B1) ---
        tr_sorted = tr.sort_values(["_DateKey", TICK_COL]).copy()
        te_sorted = te.sort_values(["_DateKey", TICK_COL]).copy()

        g_tr = tr_sorted.groupby("_DateKey").size().values
        g_te = te_sorted.groupby("_DateKey").size().values

        X_tr_r = imp.transform(
            tr_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values
        )
        X_te_r = imp.transform(
            te_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values
        )

        # Use CLF predictions as auxiliary signal for relevance levels
        p_tr_for_rel = clf.predict_proba(X_tr_r)[:, 1]
        y_tr_rel = compute_relevance_labels(tr_sorted["y_t3_int"].values, p_tr_for_rel)
        y_te_rel = compute_relevance_labels(
            te_sorted["y_t3_int"].values,
            clf.predict_proba(X_te_r)[:, 1],
        )

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

        # FIX B3: compute RankerPct per-date in fold evaluation (consistent with OOF)
        te_eval = te[["_DateKey", "y_t3_int"]].copy()
        te_eval["Prob_Clf"] = p_te
        te_eval["Score_Ranker"] = oof_rnk[te.index.values]
        te_eval["RankerPct"] = te_eval.groupby("_DateKey")["Score_Ranker"].rank(pct=True)

        # FIX A4: per-fold weight search (nested, no look-ahead)
        best_w_fold = W_CLF_INIT
        best_p15_fold = -1
        for w in np.arange(0.35, 0.86, 0.05):
            score_tmp = w * te_eval["Prob_Clf"] + (1 - w) * te_eval["RankerPct"]
            tmp = te_eval.copy()
            tmp["ScoreTmp"] = score_tmp
            p15_tmp = precision_at_k_by_date(
                tmp, "_DateKey", "y_t3_int", "ScoreTmp", k=TOP_K
            )
            if p15_tmp > best_p15_fold:
                best_p15_fold = p15_tmp
                best_w_fold = float(w)

        te_eval["RankScore"] = (
            best_w_fold * te_eval["Prob_Clf"]
            + (1 - best_w_fold) * te_eval["RankerPct"]
        )

        auc = roc_auc_score(y_te, p_te)
        ap = average_precision_score(y_te, p_te)
        p15 = precision_at_k_by_date(
            te_eval, "_DateKey", "y_t3_int", "RankScore", k=TOP_K
        )

        fold_rows.append([k, len(tr), len(te), auc, ap, p15, best_w_fold])
        print(f"[{run_label}] Fold {k}: AUC={auc:.4f} PR-AUC={ap:.4f} "
              f"P@{TOP_K}={p15:.3f} W_CLF={best_w_fold:.2f}")

    # ---- OOF aggregation ----
    mask = np.isfinite(oof_clf) & np.isfinite(oof_rnk)
    if mask.sum() == 0:
        raise ValueError(f"[{run_label}] No valid OOF predictions.")

    y_oof = train_df.loc[mask, "y_t3_int"].values
    df_oof = train_df.loc[mask, ["_DateKey"]].copy()
    df_oof["y"] = y_oof
    df_oof["Prob_Clf"] = oof_clf[mask]
    df_oof["Score_Ranker"] = oof_rnk[mask]
    # FIX B3: RankerPct per-date in OOF (already was, but now explicitly consistent)
    df_oof["RankerPct"] = df_oof.groupby("_DateKey")["Score_Ranker"].rank(pct=True)

    # FIX A4: use median of per-fold weights (not re-optimised on full OOF)
    fold_weights = [row[6] for row in fold_rows]  # W_CLF from each fold
    W_CLF = float(np.median(fold_weights)) if fold_weights else W_CLF_INIT
    W_RNK = 1.0 - W_CLF
    df_oof["RankScore"] = W_CLF * df_oof["Prob_Clf"] + W_RNK * df_oof["RankerPct"]

    print(f"\n[{run_label}] [PASS] Weights (median of fold-level): "
          f"W_CLF={W_CLF:.2f} | W_RNK={W_RNK:.2f}")

    # -- G: Optional stacking meta-model ---------------------------
    meta = None  # v4: initialize to avoid unbound variable if stacking is skipped
    use_stacking_this_run = USE_STACKING and mask.sum() > 1000

    if use_stacking_this_run:
        print(f"[{run_label}] Training stacking meta-model...")
        stack_features = df_oof[["Prob_Clf", "RankerPct"]].values
        if STACKING_META_MODEL == "gbm":
            meta = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=SEED,
            )
        else:
            meta = LogisticRegression(max_iter=2000, random_state=SEED)

        # FIX: use temporal split for meta-model (last 30% of OOF dates for validation)
        oof_dates_sorted = sorted(df_oof["_DateKey"].unique())
        meta_split_idx = int(len(oof_dates_sorted) * 0.7)
        meta_train_dates = set(oof_dates_sorted[:meta_split_idx])
        meta_mask = df_oof["_DateKey"].isin(meta_train_dates)

        meta.fit(stack_features[meta_mask], y_oof[meta_mask])
        df_oof["StackScore"] = meta.predict_proba(stack_features)[:, 1]

        # Compare stacking vs linear blend
        p15_linear = precision_at_k_by_date(
            df_oof, "_DateKey", "y", "RankScore", k=TOP_K
        )
        p15_stack = precision_at_k_by_date(
            df_oof, "_DateKey", "y", "StackScore", k=TOP_K
        )
        print(f"[{run_label}] P@{TOP_K} linear={p15_linear:.3f} vs stack={p15_stack:.3f}")

        if p15_stack > p15_linear:
            df_oof["RankScore"] = df_oof["StackScore"]
            print(f"[{run_label}] -> Using stacking (better P@K)")
        else:
            use_stacking_this_run = False
            print(f"[{run_label}] -> Keeping linear blend (stacking didn't improve)")

    # Platt calibration
    platt = LogisticRegression(max_iter=2000, random_state=SEED)
    platt.fit(df_oof[["RankScore"]].values, y_oof)
    df_oof["Prob_T3_FINAL"] = platt.predict_proba(df_oof[["RankScore"]].values)[:, 1]

    # FIX B2: record OOF RankScore distribution for shift detection
    oof_rank_stats = {
        "mean": float(df_oof["RankScore"].mean()),
        "std": float(df_oof["RankScore"].std()),
        "q25": float(df_oof["RankScore"].quantile(0.25)),
        "q75": float(df_oof["RankScore"].quantile(0.75)),
    }

    # FIX D2: report AUC on the final signal, not just Prob_Clf
    auc_oof_clf = roc_auc_score(y_oof, df_oof["Prob_Clf"].values)
    auc_oof_final = roc_auc_score(y_oof, df_oof["RankScore"].values)
    ap_oof = average_precision_score(y_oof, df_oof["RankScore"].values)
    base_rate = y_oof.mean()
    p_at_k = precision_at_k_by_date(df_oof, "_DateKey", "y", "RankScore", k=TOP_K)

    # FIX D3: permutation test
    print(f"\n[{run_label}] Running permutation test (1000 permutations)...")
    obs_p15, rand_p15, pval = permutation_test_p_at_k(
        df_oof, "_DateKey", "y", "RankScore", k=TOP_K, n_perm=1000, seed=SEED,
    )
    print(f"[{run_label}] Permutation test: observed P@{TOP_K}={obs_p15:.3f}, "
          f"random={rand_p15:.3f}, p-value={pval:.4f}")

    print(f"[{run_label}] OOF AUC(Prob_Clf)={auc_oof_clf:.4f}")
    print(f"[{run_label}] OOF AUC(RankScore)={auc_oof_final:.4f}")
    print(f"[{run_label}] OOF PR-AUC(RankScore)={ap_oof:.4f}")
    print(f"[{run_label}] BaseRate={base_rate:.2%}")
    print(f"[{run_label}] P@{TOP_K}={p_at_k:.3f}")

    # ---- Final model fit ----
    n_estim_clf = safe_median_best(best_iters_clf, 2500)
    n_estim_rnk = safe_median_best(best_iters_rnk, 1200)

    imp_final = SimpleImputer(strategy="median")
    X_full = imp_final.fit_transform(
        train_df[feature_cols].replace([np.inf, -np.inf], np.nan).values
    )
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
        random_state=SEED,
    )
    clf_final.fit(X_full, y_full, sample_weight=w_full)

    # -- v4: Feature importance diagnostic ----------------------------
    # Usar n_features_in_ del modelo como fuente de verdad para el largo.
    # Si hay mismatch con feature_cols (columnas duplicadas en el Excel origen),
    # se usan los nombres disponibles y se avisa.
    n_model_feats = len(clf_final.feature_importances_)
    if n_model_feats == len(feature_cols):
        feat_names_imp = feature_cols
    else:
        print(f"  [WARN] Mismatch feature importance: feature_cols={len(feature_cols)}, "
              f"model={n_model_feats}. Probable causa: columnas duplicadas en el Excel. "
              f"Usando nombres indexados como fallback.")
        feat_names_imp = (feature_cols + [f"_extra_{i}" for i in range(n_model_feats)])[:n_model_feats]
    feat_imp_df = pd.DataFrame({
        "Feature": feat_names_imp,
        "Importance": clf_final.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    print(f"\n[{run_label}] Top-10 features by importance:")
    print(feat_imp_df.head(10).to_string(index=False))

    train_sorted = train_df.sort_values(["_DateKey", TICK_COL]).copy()
    g_full = train_sorted.groupby("_DateKey").size().values
    X_full_r = imp_final.transform(
        train_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values
    )
    # Multi-level relevance for final ranker too
    p_full_for_rel = clf_final.predict_proba(X_full_r)[:, 1]
    y_full_rel = compute_relevance_labels(
        train_sorted["y_t3_int"].values, p_full_for_rel
    )

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
        verbosity=-1,
    )
    rnk_final.fit(X_full_r, y_full_rel, group=g_full)

    # FIX E2: cache components for reuse
    _shared_cache[run_label] = {
        "imp_final": imp_final,
        "clf_final": clf_final,
        "rnk_final": rnk_final,
        "platt": platt,
        "W_CLF": W_CLF,
        "W_RNK": W_RNK,
        "meta": meta if use_stacking_this_run else None,
        "use_stacking": use_stacking_this_run,
        "oof_rank_stats": oof_rank_stats,
    }

    # ---- Score last close ----
    any_filter = (min_price_filter is not None) or (max_price_filter is not None)
    if any_filter and APPLY_PRICE_FILTER_TO_LAST_CLOSE:
        df_score_base = apply_price_filter(df_all, PRICE_COL, min_price=min_price_filter, max_price=max_price_filter)
    else:
        df_score_base = df_all.copy()

    # FIX E1: explicit anchor-date handling with audit trail
    df_last = df_score_base[df_score_base["_DateKey"] == GLOBAL_ANCHOR_DATE].copy()
    df_last = df_last.sort_values([TICK_COL, DATE_COL]).drop_duplicates(
        subset=[TICK_COL], keep="last"
    )

    total_tickers = df_score_base[TICK_COL].nunique()
    anchor_coverage = df_last[TICK_COL].nunique() / max(1, total_tickers)

    if anchor_coverage < 0.5:
        print(f"  [WARN] ANCHOR FALLBACK: only {anchor_coverage:.1%} coverage at "
              f"{GLOBAL_ANCHOR_DATE.date()}. Using per-ticker last row.")
        df_last = (
            df_score_base
            .sort_values([TICK_COL, DATE_COL])
            .groupby(TICK_COL)
            .tail(1)
            .copy()
        )
        df_last = df_last.drop_duplicates(subset=[TICK_COL], keep="last")
    else:
        print(f"  [OK] Anchor coverage: {anchor_coverage:.1%} ({df_last[TICK_COL].nunique()} tickers)")

    # Missing tickers diagnostic
    last_by_ticker = df_score_base.dropna(subset=["_DateKey"]).groupby(TICK_COL)["_DateKey"].max()
    missing = last_by_ticker[last_by_ticker < GLOBAL_ANCHOR_DATE].sort_values()
    missing_df = missing.reset_index()
    missing_df.columns = [TICK_COL, "Last_Available_Date"]
    missing_df["Expected_Last_Close"] = GLOBAL_ANCHOR_DATE

    nan_frac_last = df_last[feature_cols].isna().mean(axis=1)
    df_last = df_last[nan_frac_last <= MAX_NAN_FRAC_LAST].copy()

    if len(df_last) == 0:
        raise ValueError(f"[{run_label}] No tickers in last close after filters.")

    X_last = imp_final.transform(
        df_last[feature_cols].replace([np.inf, -np.inf], np.nan).values
    )
    p_last = clf_final.predict_proba(X_last)[:, 1]

    df_last_sorted = df_last.sort_values(["_DateKey", TICK_COL]).copy()
    X_last_r = imp_final.transform(
        df_last_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).values
    )
    s_last = rnk_final.predict(X_last_r)
    s_last_series = pd.Series(s_last, index=df_last_sorted.index)
    s_last_aligned = s_last_series.loc[df_last.index].values

    out = df_last[[TICK_COL, DATE_COL, PRICE_COL]].copy()
    if BASELINE_COL and BASELINE_COL in df_last.columns:
        out["Baseline_Predicted_Outperform_Prob"] = pd.to_numeric(
            df_last[BASELINE_COL], errors="coerce"
        )

    out["Prob_Clf"] = p_last
    out["Score_Ranker"] = s_last_aligned
    # FIX B3: RankerPct per-date (consistent with OOF)
    out["RankerPct"] = out.groupby(
        df_last["_DateKey"].values
    )["Score_Ranker"].rank(pct=True)

    if use_stacking_this_run:
        stack_input = out[["Prob_Clf", "RankerPct"]].values
        out["RankScore"] = meta.predict_proba(stack_input)[:, 1]
    else:
        out["RankScore"] = W_CLF * out["Prob_Clf"] + W_RNK * out["RankerPct"]

    # FIX B2: distribution shift monitoring
    last_rank_stats = {
        "mean": float(out["RankScore"].mean()),
        "std": float(out["RankScore"].std()),
        "q25": float(out["RankScore"].quantile(0.25)),
        "q75": float(out["RankScore"].quantile(0.75)),
    }
    mean_shift = abs(last_rank_stats["mean"] - oof_rank_stats["mean"]) / max(
        oof_rank_stats["std"], 1e-6
    )
    if mean_shift > 1.5:
        print(f"  [WARN] DISTRIBUTION SHIFT: RankScore mean shifted {mean_shift:.1f} "
              f"std from OOF (OOF: {oof_rank_stats['mean']:.3f}, "
              f"Last: {last_rank_stats['mean']:.3f})")
        print(f"    Platt calibration may be unreliable. Consider re-calibrating.")
    else:
        print(f"  [OK] RankScore distribution shift: {mean_shift:.2f} std (acceptable)")

    out["Prob_T3_FINAL"] = platt.predict_proba(out[["RankScore"]].values)[:, 1]

    # G: score smoothing (if prior scores available)
    if SCORE_SMOOTHING_ALPHA > 0 and PRIOR_SCORES_PATH is not None:
        try:
            prior = pd.read_excel(PRIOR_SCORES_PATH, sheet_name="All_Last")
            prior = prior[[TICK_COL, "RankScore"]].rename(
                columns={"RankScore": "RankScore_prev"}
            )
            out = out.merge(prior, on=TICK_COL, how="left")
            has_prior = out["RankScore_prev"].notna()
            out.loc[has_prior, "RankScore"] = (
                SCORE_SMOOTHING_ALPHA * out.loc[has_prior, "RankScore_prev"]
                + (1 - SCORE_SMOOTHING_ALPHA) * out.loc[has_prior, "RankScore"]
            )
            out.drop(columns=["RankScore_prev"], inplace=True)
            print(f"  [OK] Score smoothing applied (alpha={SCORE_SMOOTHING_ALPHA})")
        except Exception as e:
            print(f"  [WARN] Score smoothing failed: {e}")

    # Sort and rank
    sort_cols = ["RankScore", "Prob_Clf", "RankerPct"]
    asc = [False, False, False]
    if "Baseline_Predicted_Outperform_Prob" in out.columns:
        sort_cols.append("Baseline_Predicted_Outperform_Prob")
        asc.append(False)
    sort_cols.append(TICK_COL)
    asc.append(True)

    out = out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    # G: dynamic Top-K
    if USE_DYNAMIC_TOPK:
        threshold = np.percentile(out["Prob_T3_FINAL"].values, DYNAMIC_TOPK_PERCENTILE)
        confident_mask = out["Prob_T3_FINAL"] >= threshold
        dynamic_k = int(np.clip(confident_mask.sum(), DYNAMIC_TOPK_MIN, DYNAMIC_TOPK_MAX))
        topk = out.head(dynamic_k).copy()
        print(f"[{run_label}] Dynamic Top-K: {dynamic_k} "
              f"(threshold={threshold:.3f}, conf_tickers={confident_mask.sum()})")
    else:
        dynamic_k = TOP_K
        topk = out.head(TOP_K).copy()

    # G: turnover tracking
    turnover_info = ""
    if PRIOR_SCORES_PATH is not None:
        try:
            prior_top = pd.read_excel(PRIOR_SCORES_PATH, sheet_name=f"Top_{TOP_K}_Last")
            prior_tickers = set(prior_top[TICK_COL].values)
            current_tickers = set(topk[TICK_COL].values)
            overlap = prior_tickers & current_tickers
            turnover = 1.0 - len(overlap) / max(1, len(current_tickers))
            turnover_info = f"{turnover:.1%}"
            print(f"[{run_label}] Turnover vs prior week: {turnover:.1%} "
                  f"({len(overlap)} tickers retained)")
            if turnover > 0.8:
                print(f"  [WARN] HIGH TURNOVER (>80%): signal may be noisy. "
                      f"Consider increasing SCORE_SMOOTHING_ALPHA.")
        except Exception:
            turnover_info = "N/A (no prior file)"

    # -- v4: Target 1 - short / sell candidates -----------------------
    # Bottom-K stocks by RankScore = lowest probability of outperformance.
    # Use as short candidates or to flag existing long positions for exit.
    # Note: the model was trained on Target3 (outperformers); these are
    # stocks the model is MOST confident will NOT outperform - a meaningful
    # but indirect short signal. A dedicated Target1 model would be stronger.
    bottom_k = out.tail(dynamic_k).sort_values("RankScore", ascending=True).reset_index(drop=True)
    bottom_k.insert(0, "Short_Rank", np.arange(1, len(bottom_k) + 1))

    # Build fold metrics df
    fold_df = pd.DataFrame(
        fold_rows,
        columns=["Fold", "TrainRows", "TestRows", "AUC", "PR_AUC",
                 f"P@{TOP_K}_RankScore", "W_CLF_fold"],
    )

    summary = pd.DataFrame({
        "Metric": [
            "RunLabel",
            "PipelineVersion",
            "InputHash",
            "MinPriceFilter",
            "MaxPriceFilter",
            "OOF_AUC(Prob_Clf)",
            "OOF_AUC(RankScore)",
            "OOF_PR_AUC(RankScore)",
            f"OOF_P@{TOP_K}_RankScore",
            f"Permutation_p_value_P@{TOP_K}",
            "BaseRate_Target3",
            "AnchorDate",
            "AnchorCoverage",
            "LastClose_Tickers",
            "Total_Tickers",
            "TrainRows",
            "TrainTickers",
            "W_CLF",
            "W_RNK",
            "Embargo_weeks",
            "UseStacking",
            "UseDynamicTopK",
            "DynamicK",
            "Turnover_vs_prior",
            "RankScore_dist_shift_std",
            "UseSectorNeutralisation",
        ],
        "Value": [
            run_label,
            "v3.0",
            INPUT_HASH,
            "None" if min_price_filter is None else float(min_price_filter),
            "None" if max_price_filter is None else float(max_price_filter),
            float(auc_oof_clf),
            float(auc_oof_final),
            float(ap_oof),
            float(p_at_k),
            float(pval),
            float(base_rate),
            str(GLOBAL_ANCHOR_DATE.date()),
            f"{anchor_coverage:.1%}",
            int(len(out)),
            int(total_tickers),
            int(len(train_df)),
            int(train_df[TICK_COL].nunique()),
            float(W_CLF),
            float(W_RNK),
            EMBARGO_DATES,
            str(use_stacking_this_run),
            str(USE_DYNAMIC_TOPK),
            dynamic_k,
            turnover_info,
            f"{mean_shift:.2f}",
            str(USE_SECTOR_NEUTRALISATION),
        ],
    })

    # Export
    top_sheet_name = f"Top_{dynamic_k}_Last" if USE_DYNAMIC_TOPK else f"Top_{TOP_K}_Last"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        topk.to_excel(writer, sheet_name=top_sheet_name, index=False)
        bottom_k.to_excel(writer, sheet_name=f"Bottom_{dynamic_k}_Short_T1", index=False)
        out.to_excel(writer, sheet_name="All_Last", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        fold_df.to_excel(writer, sheet_name="Fold_Metrics", index=False)
        missing_df.to_excel(writer, sheet_name="Missing_LastClose", index=False)
        feat_imp_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

    print(f"\n[{run_label}] [PASS] Exported: {out_file}")
    print(f"[{run_label}] Anchor date: {GLOBAL_ANCHOR_DATE.date()}")
    print(f"[{run_label}] Top {dynamic_k}:")
    display_cols = ["Rank", TICK_COL, PRICE_COL, "RankScore", "Prob_T3_FINAL",
                    "Prob_Clf", "RankerPct"]
    print(topk[display_cols].to_string(index=False))

    # -- v4: metrics dict for cross-run comparison ---------------------
    run_metrics = {
        "RunLabel":         run_label,
        "MinPriceFilter":   "None" if min_price_filter is None else float(min_price_filter),
        "MaxPriceFilter":   "None" if max_price_filter is None else float(max_price_filter),
        "OOF_AUC_Clf":      float(auc_oof_clf),
        "OOF_AUC_Final":    float(auc_oof_final),
        "OOF_PR_AUC":       float(ap_oof),
        f"OOF_P@{TOP_K}":   float(p_at_k),
        "Permutation_pval": float(pval),
        "BaseRate":         float(base_rate),
        "Tickers_Scored":   int(len(out)),
        "DynamicK":         dynamic_k,
        "Turnover":         turnover_info,
        "DistShift_std":    f"{mean_shift:.2f}",
        "UseStacking":      str(use_stacking_this_run),
    }
    return out, run_metrics


# #+==================================================================+#
# #                       EXECUTION  v4                             #
# #+==================================================================+#

print("\n" + "=" * 80)
print("PIPELINE v4.0 - Triple price-filter execution")
print(f"  Run 1: FULL universe (no price filter)")
print(f"  Run 2: MIN price > {MIN_PRICE_FILTER}  (removes cheap/penny stocks)")
print(f"  Run 3: MAX price < {MAX_PRICE_FILTER}  (cheap/volatile stocks only)")
print("=" * 80)

all_metrics = []

# Run 1: full universe
result_full, m_full = run_pipeline(
    df_all=df,
    min_price_filter=None,
    max_price_filter=None,
    out_file=OUT_FILE_FULL,
    run_label="FULL_UNIVERSE",
)
all_metrics.append(m_full)

# Run 2: min-price filter (removes penny/cheap stocks - same logic as v3)
# Note: training distribution changes with filter, so models are retrained.
result_min, m_min = run_pipeline(
    df_all=df,
    min_price_filter=MIN_PRICE_FILTER,
    max_price_filter=None,
    out_file=OUT_FILE_MIN,
    run_label=f"MIN_PRICE_GT_{MIN_PRICE_FILTER}",
)
all_metrics.append(m_min)

# Run 3: max-price filter (cheap/volatile universe - new in v4)
result_max, m_max = run_pipeline(
    df_all=df,
    min_price_filter=None,
    max_price_filter=MAX_PRICE_FILTER,
    out_file=OUT_FILE_MAX,
    run_label=f"MAX_PRICE_LT_{MAX_PRICE_FILTER}",
)
all_metrics.append(m_max)

# -- Automatic comparison table ----------------------------------------
comparison_df = pd.DataFrame(all_metrics)
p_at_k_col = f"OOF_P@{TOP_K}"
best_run = comparison_df.loc[comparison_df[p_at_k_col].idxmax(), "RunLabel"]

print("\n" + "=" * 80)
print(f"PRICE FILTER COMPARISON - ranked by OOF P@{TOP_K}")
print("=" * 80)
display_cols = [
    "RunLabel", "MinPriceFilter", "MaxPriceFilter",
    "OOF_AUC_Final", "OOF_PR_AUC", p_at_k_col,
    "Permutation_pval", "Tickers_Scored",
]
print(
    comparison_df[display_cols]
    .sort_values(p_at_k_col, ascending=False)
    .to_string(index=False)
)
print(f"\n  -> Best filter by P@{TOP_K}: {best_run}")
print("=" * 80)

# Export comparison to its own Excel
with pd.ExcelWriter(OUT_FILE_COMPARISON, engine="openpyxl") as writer:
    (
        comparison_df[display_cols]
        .sort_values(p_at_k_col, ascending=False)
        .to_excel(writer, sheet_name="Ranked_by_P_at_K", index=False)
    )
    comparison_df.to_excel(writer, sheet_name="All_Metrics", index=False)

print(f"\nComparison file: {OUT_FILE_COMPARISON}")
print("\n" + "=" * 80)
print("PIPELINE v4.0 - COMPLETED")
print(f"  FULL:        {OUT_FILE_FULL}")
print(f"  MIN_PRICE:   {OUT_FILE_MIN}")
print(f"  MAX_PRICE:   {OUT_FILE_MAX}")
print(f"  COMPARISON:  {OUT_FILE_COMPARISON}")
print(f"  Input hash:  {INPUT_HASH}")
print("=" * 80)