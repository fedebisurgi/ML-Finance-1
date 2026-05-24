# -*- coding: utf-8 -*-
"""
target3_core.py  —  módulo compartido para predict y backtest

CONTENIDO
---------
  - Configuración (paths, hiperparámetros, fechas backtest default).
  - Helpers (walk-forward, métricas, builders LGBM, etc.).
  - compute_universe_masks, compute_illiquidity_momentum_features.
  - load_and_prepare_data() → PipelineContext.
  - run_pipeline (entrenamiento + holdout scoring + score último cierre).
  - run_single_backtest, run_backtest_loop, _compute_attribution_metrics.

USO
---
  from target3_core import load_and_prepare_data, run_pipeline, ...
  ctx = load_and_prepare_data()
  run_pipeline(ctx.df, ctx.feature_cols_base, ...)

NO ejecutar este módulo directamente. Usar:
  - target3_predict.py   para Top-20 semanal productivo
  - target3_backtest.py  para validación walk-forward histórica
"""

import warnings
warnings.filterwarnings("ignore")

import time
import re
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from typing import List, Tuple, Optional, Dict, Any, NamedTuple

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRanker

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from sklearn.linear_model import LogisticRegression

# ╔══════════════════════════════════════════════════════════════════╗
# ║                        CONFIGURATION                            ║
# ╚══════════════════════════════════════════════════════════════════╝

BASE_DIR = Path(r"C:\Users\GOFOYCOP_01\00.Redes neuronales\04.Descarga anual\03.consolidado")
FILE_NAME = "Consolidado_100_semanas_paste_todos.xlsx"
FILE_PATH = BASE_DIR / FILE_NAME

TOP_K = 20

# Walk-forward
N_FOLDS = 6
EMBARGO_DATES = 4
MIN_TRAIN_DATES = 35
TEST_DATES_PER_FOLD = None

# Holdout: últimas N fechas únicas, fijas para todos los runs
HOLDOUT_N_WEEKS = 20

# Time-decay
USE_TIME_DECAY = True
HALF_LIFE_WEEKS = 26

# Último cierre
MAX_NAN_FRAC_LAST = 0.55

# Blend weights iniciales (ajustados por fold dentro del pipeline)
W_CLF_INIT = 0.65

SEED = 42
np.random.seed(SEED)

# Filtro de precio para baseline (equivale al "filtro viejo")
BASELINE_MIN_PRICE = 5.0
APPLY_PRICE_FILTER_TO_LAST_CLOSE = True

# ── Filtros de universo (nuevos) ─────────────────────────────────
UNIV_MIN_CLOSE              = 5.0
UNIV_MIN_DOLLAR_VOL_20D     = 5_000_000.0
UNIV_MAX_ZERO_RET_PCT_60D   = 0.20

# Columnas buscadas en orden de preferencia
VOL_COL_CANDIDATES  = ["Volume", "Vol", "Volumen"]
HIGH_COL_CANDIDATES = ["High", "Alto"]

# ── Correlación alerta ────────────────────────────────────────────
CORR_ALERT_THRESHOLD = 0.85

DATE_STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# ── Backtest walk-forward ────────────────────────────────────────
# 18 fechas (2026-01-02 a 2026-05-15). 27/3 y 3/4 omitidas a propósito.
BACKTEST_DATES = [
    "2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23", "2026-01-30",
    "2026-02-06", "2026-02-13", "2026-02-20", "2026-02-27",
    "2026-03-06", "2026-03-13", "2026-03-20",
    "2026-04-10", "2026-04-17", "2026-04-24",
    "2026-05-01", "2026-05-08", "2026-05-15",
]
# Tiempo máximo estimado antes de pedir confirmación (horas)
BACKTEST_MAX_HOURS_AUTO = 1.5


# ╔══════════════════════════════════════════════════════════════════╗
# ║                          HELPERS                                ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_input_hash(path: Path, first_n_bytes: int = 2 ** 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(first_n_bytes))
    return h.hexdigest()[:12]


INPUT_HASH = compute_input_hash(FILE_PATH) if FILE_PATH.exists() else "NOHASH"

OUT_FILE_BASELINE  = BASE_DIR / f"T5_BASELINE_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_ABLATION  = BASE_DIR / f"T5_ABLATION_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_NEW       = BASE_DIR / f"T5_NEW_{DATE_STAMP}_{INPUT_HASH}.xlsx"

OUT_FILE_BT_BASELINE = BASE_DIR / f"T5_BACKTEST_BASELINE_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_BT_ABLATION = BASE_DIR / f"T5_BACKTEST_ABLATION_{DATE_STAMP}_{INPUT_HASH}.xlsx"
OUT_FILE_BT_NEW      = BASE_DIR / f"T5_BACKTEST_NEW_{DATE_STAMP}_{INPUT_HASH}.xlsx"


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


def make_walkforward_splits(
    unique_dates,
    n_folds: int = 6,
    embargo: int = 4,
    min_train_dates: int = 35,
    test_dates_per_fold: Optional[int] = None,
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Non-overlapping expanding-window walk-forward (FIX A2 from v3.0)."""
    dates = pd.DatetimeIndex(sorted(pd.Series(unique_dates).dropna().unique()))
    n = len(dates)

    if n < (min_train_dates + embargo + 3):
        cut = int(n * 0.75)
        return [(dates[:cut], dates[cut + embargo:])]

    if test_dates_per_fold is None:
        available = n - min_train_dates - embargo
        test_dates_per_fold = max(3, available // n_folds)

    val_blocks, cursor = [], n
    for _ in range(n_folds):
        val_end = cursor
        val_start = max(0, val_end - test_dates_per_fold)
        if val_start <= min_train_dates + embargo:
            break
        val_blocks.append((val_start, val_end))
        cursor = val_start - embargo
    val_blocks.reverse()

    splits = []
    for val_start, val_end in val_blocks:
        train_end = val_start - embargo
        if train_end < min_train_dates:
            continue
        tr_dates = dates[:train_end]
        va_dates = dates[val_start:val_end]
        if len(tr_dates) >= min_train_dates and len(va_dates) >= 3:
            splits.append((tr_dates, va_dates))

    if not splits:
        cut = int(n * 0.75)
        splits = [(dates[:cut], dates[cut + min(embargo, n - cut - 1):])]
    return splits


def precision_at_k_by_date(df_pred, date_col, y_true_col, score_col, k=20):
    precs = []
    for _, g in df_pred.groupby(date_col):
        top = g.nlargest(k, score_col)
        if len(top):
            precs.append(top[y_true_col].mean())
    return float(np.mean(precs)) if precs else np.nan


def ndcg_at_k_by_date(df_pred, date_col, y_true_col, score_col, k=20):
    scores = []
    for _, g in df_pred.groupby(date_col):
        if len(g) < 2:
            continue
        try:
            scores.append(
                ndcg_score(
                    [g[y_true_col].values],
                    [g[score_col].values],
                    k=k,
                )
            )
        except Exception:
            pass
    return float(np.mean(scores)) if scores else np.nan


def permutation_test_p_at_k(df_pred, date_col, y_true_col, score_col, k=20, n_perm=500, seed=42):
    rng = np.random.RandomState(seed)
    observed = precision_at_k_by_date(df_pred, date_col, y_true_col, score_col, k)
    perm_scores = []
    for _ in range(n_perm):
        df_perm = df_pred.copy()
        df_perm[score_col] = df_perm.groupby(date_col)[score_col].transform(
            lambda x: rng.permutation(x.values)
        )
        perm_scores.append(
            precision_at_k_by_date(df_perm, date_col, y_true_col, score_col, k)
        )
    perm_scores = np.array(perm_scores)
    return observed, float(np.mean(perm_scores)), float(np.mean(perm_scores >= observed))


def _isnan_safe(v) -> bool:
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return True


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
    """Multi-level relevance 0-3 (FIX B1 from v3.0)."""
    if prob_clf is None:
        return (y_binary * 2).astype(int)
    median_prob = np.median(prob_clf)
    rel = np.zeros(len(y_binary), dtype=int)
    rel[(y_binary == 0) & (prob_clf >= median_prob)] = 1
    rel[(y_binary == 1) & (prob_clf < median_prob)]  = 2
    rel[(y_binary == 1) & (prob_clf >= median_prob)] = 3
    return rel


def gap_aware_shift(df, tick_col, date_col, value_col, periods=1, max_gap_weeks=2):
    """Shift que devuelve NaN si hay gap > max_gap_weeks (FIX C2 from v3.0)."""
    result = df.groupby(tick_col)[value_col].shift(periods)
    if periods > 0:
        date_diff = df.groupby(tick_col)[date_col].diff()
        result[date_diff > pd.Timedelta(weeks=max_gap_weeks)] = np.nan
    return result


def diagnose_date_consistency(df, tick_col, date_col):
    by_date = df.groupby("_DateKey")[date_col].agg(["min", "max", "nunique"])
    inconsistent = by_date[by_date["nunique"] > 1]
    if len(inconsistent):
        spread = (inconsistent["max"] - inconsistent["min"]).max()
        print(f"  [WARN] DATE CONSISTENCY: {len(inconsistent)} _DateKey con múltiples fechas "
              f"(spread máx: {spread})")
    else:
        print("  [OK] Date consistency OK: 1 fecha raw por _DateKey.")
    return inconsistent


def hhi_sector(df_top, sector_col):
    if sector_col not in df_top.columns or df_top[sector_col].isna().all():
        return np.nan
    w = df_top[sector_col].value_counts(normalize=True)
    return float((w ** 2).sum())


# ╔══════════════════════════════════════════════════════════════════╗
# ║               FILTROS DE UNIVERSO (NUEVO v5.0)                  ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_universe_masks(df, tick_col, price_col, ret_col="_ret_w_tmp"):
    """
    Calcula 3 columnas booleanas por fila (sin leakage: solo datos hasta t):
      _univ_close5  : Close > $5
      _univ_dvol    : Dollar_Volume_20d > $5M  (NaN -> True si no hay vol)
      _univ_zret    : Zero_Return_Days_Pct_60d < 20%
      _univ_pass    : las 3 juntas

    Notas de implementación:
      - Dollar_Volume_20d: rolling 4 semanas ≈ 20 días hábiles.
      - Zero_Return_Days_Pct_60d: rolling 12 semanas ≈ 60 días hábiles;
        usa el retorno semanal ya calculado en ret_col (gap-aware).
    """
    df = df.copy()

    price = pd.to_numeric(df[price_col], errors="coerce")

    # Filter 1: Close > $5
    df["_univ_close5"] = price > UNIV_MIN_CLOSE

    # Buscar volumen raw; fallback a Vol_Rel_20 (Volume/20d-mean) si no está
    vol_col     = next((c for c in VOL_COL_CANDIDATES if c in df.columns), None)
    vol_rel_col = "Vol_Rel_20" if "Vol_Rel_20" in df.columns else None

    # Filter 2: Dollar_Volume_20d > $5M
    if vol_col:
        vol = pd.to_numeric(df[vol_col], errors="coerce").replace(0, np.nan)
        dv_raw = price * vol
        df["Dollar_Volume_20d"] = (
            dv_raw.groupby(df[tick_col])
            .transform(lambda x: x.rolling(4, min_periods=2).mean())
        )
        df["_univ_dvol"] = df["Dollar_Volume_20d"] > UNIV_MIN_DOLLAR_VOL_20D
    elif vol_rel_col:
        # Vol_Rel_20 = Volume / Volume.rolling(20).mean()  (ya computado en datos)
        # Proxy: Close * Vol_Rel_20 es proporcional al dollar volume relativo.
        # Threshold ajustado: > 0.5 ≈ volumen al menos 50% del promedio histórico.
        print("  [WARN] Sin Volume raw: usando Vol_Rel_20 > 0.5 como proxy de liquidez.")
        vr = pd.to_numeric(df[vol_rel_col], errors="coerce").replace(0, np.nan)
        dv_proxy = price * vr
        df["Dollar_Volume_20d"] = (
            dv_proxy.groupby(df[tick_col])
            .transform(lambda x: x.rolling(4, min_periods=2).mean())
        )
        df["_univ_dvol"] = df["Dollar_Volume_20d"] > 0.5
    else:
        print("  [WARN] Sin Volume ni Vol_Rel_20: filtro Dollar_Volume_20d desactivado.")
        df["Dollar_Volume_20d"] = np.nan
        df["_univ_dvol"] = True

    # Filter 3: Zero_Return_Days_Pct_60d < 20%
    # Usa ret_col si existe, sino recalcula pct_change
    if ret_col in df.columns:
        ret = df[ret_col].copy()
    else:
        ret = df.groupby(tick_col)[price_col].pct_change(1)

    is_zero = (ret.abs() < 1e-4).astype(float)
    df["Zero_Return_Days_Pct_60d"] = (
        is_zero.groupby(df[tick_col])
        .transform(lambda x: x.rolling(12, min_periods=6).mean())
    )
    df["_univ_zret"] = df["Zero_Return_Days_Pct_60d"] < UNIV_MAX_ZERO_RET_PCT_60D

    df["_univ_pass"] = df["_univ_close5"] & df["_univ_dvol"] & df["_univ_zret"]

    return df


# ╔══════════════════════════════════════════════════════════════════╗
# ║          4 FEATURES NUEVAS (OHLCV SEMANAL, NUEVO v5.0)         ║
# ╚══════════════════════════════════════════════════════════════════╝

def compute_illiquidity_momentum_features(df, tick_col, date_col, price_col, ret_1w_col="Ret_1w_calc"):
    """
    Computa 4 features + su variante __rank cross-sectional por fecha.

    Aproximación a datos diarios con frecuencia semanal:
      - Amihud_ILLIQ_20d  : rolling 4 semanas ≈ 20d hábiles
      - MAX5_21d_neg       : rolling 12 semanas, top-5 de las disponibles
                            (12w ≈ 60d; spirit: capturar lottery stocks)
      - Ratio_52w_High     : rolling 52 semanas
      - InformationDisc_12M: rolling 52 semanas ≈ 252d
    """

    price = pd.to_numeric(df[price_col], errors="coerce")

    # Usar retorno semanal ya calculado (gap-aware)
    if ret_1w_col in df.columns:
        ret_w = df[ret_1w_col].copy()
    else:
        ret_w = df.groupby(tick_col)[price_col].pct_change(1)
        print(f"  [WARN] '{ret_1w_col}' no encontrado; usando pct_change(1) sin gap-awareness.")

    # ── 1) Amihud_ILLIQ_20d ─────────────────────────────────────
    print("  [1/4] Amihud_ILLIQ_20d ...")
    vol_col = next((c for c in VOL_COL_CANDIDATES if c in df.columns), None)
    if vol_col:
        vol = pd.to_numeric(df[vol_col], errors="coerce").replace(0, np.nan)
        illiq_raw = ret_w.abs() / (price * vol)
        df["Amihud_ILLIQ_20d"] = (
            illiq_raw.groupby(df[tick_col])
            .transform(lambda x: x.rolling(4, min_periods=2).mean())
        )
        df["Amihud_ILLIQ_20d"] = np.log1p(df["Amihud_ILLIQ_20d"])
        # Winsorize per date 1%/99%
        df["Amihud_ILLIQ_20d"] = df.groupby(date_col)["Amihud_ILLIQ_20d"].transform(
            lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
        )
        nan_pct = df["Amihud_ILLIQ_20d"].isna().mean()
        print(f"       NaN: {nan_pct:.1%}")
    else:
        vol_rel_col = "Vol_Rel_20" if "Vol_Rel_20" in df.columns else None
        if vol_rel_col:
            vr = pd.to_numeric(df[vol_rel_col], errors="coerce").replace(0, np.nan)
            # Vol_Rel_20 = Volume / Volume.rolling(20).mean()  → proxy relativo de liquidez
            illiq_raw = ret_w.abs() / (price * vr)
            df["Amihud_ILLIQ_20d"] = (
                illiq_raw.groupby(df[tick_col])
                .transform(lambda x: x.rolling(4, min_periods=2).mean())
            )
            df["Amihud_ILLIQ_20d"] = np.log1p(df["Amihud_ILLIQ_20d"])
            df["Amihud_ILLIQ_20d"] = df.groupby(date_col)["Amihud_ILLIQ_20d"].transform(
                lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99))
            )
            nan_pct = df["Amihud_ILLIQ_20d"].isna().mean()
            print(f"       [WARN] Sin Volume raw: usando Vol_Rel_20 como proxy. NaN: {nan_pct:.1%}")
        else:
            df["Amihud_ILLIQ_20d"] = np.nan
            print("       [WARN] Sin Volume ni Vol_Rel_20: Amihud_ILLIQ_20d = NaN 100%")

    # ── 2) MAX5_21d_neg ──────────────────────────────────────────
    # Cita: Bali, Cakici & Whitelaw (2011, JFE)
    # Signo invertido: lottery stocks underperforman -> valor alto = señal long
    print("  [2/4] MAX5_21d_neg ...")

    def _max5(w):
        valid = w[~np.isnan(w)]
        if len(valid) < 3:
            return np.nan
        n_top = min(5, len(valid))
        return -1.0 * np.sort(valid)[-n_top:].mean()

    # Guardamos ret_w en columna temporal para el rolling apply groupby
    # Clip retornos extremos: <-99% son errores de datos (penny stocks), >1000% idem
    _tmp_col = "__ret_w_tmp_max5"
    df[_tmp_col] = ret_w.clip(lower=-0.99, upper=10.0).values

    df["MAX5_21d_neg"] = (
        df.groupby(tick_col)[_tmp_col]
        .transform(lambda x: x.rolling(12, min_periods=5).apply(_max5, raw=True))
    )
    df.drop(columns=[_tmp_col], inplace=True)
    print(f"       NaN: {df['MAX5_21d_neg'].isna().mean():.1%}")

    # ── 3) Ratio_52w_High ────────────────────────────────────────
    # Cita: George & Hwang (2004, JF). Rango [0,1].
    print("  [3/4] Ratio_52w_High ...")
    high_col = next((c for c in HIGH_COL_CANDIDATES if c in df.columns), None)
    if high_col:
        high_series = pd.to_numeric(df[high_col], errors="coerce")
    else:
        high_series = price.copy()

    high52 = high_series.groupby(df[tick_col]).transform(
        lambda x: x.rolling(52, min_periods=26).max()
    )
    df["Ratio_52w_High"] = price / high52.replace(0, np.nan)
    df["Ratio_52w_High"] = df["Ratio_52w_High"].clip(0.0, 1.0)
    print(f"       NaN: {df['Ratio_52w_High'].isna().mean():.1%}")

    # ── 4) Information_Discreteness_12M ─────────────────────────
    # Cita: Da, Gurun & Warachka (2014, RFS)
    # ID = sign(PRET) * (pct_neg - pct_pos)
    print("  [4/4] Information_Discreteness_12M ...")

    def _info_disc(w):
        valid = w[~np.isnan(w)]
        if len(valid) < 20:
            return np.nan
        pret = float(np.prod(1.0 + valid) - 1.0)
        pct_pos = float(np.mean(valid > 0))
        pct_neg = float(np.mean(valid < 0))
        return float(np.sign(pret)) * (pct_neg - pct_pos)

    _tmp_col2 = "__ret_w_tmp_id"
    df[_tmp_col2] = ret_w.values

    df["Information_Discreteness_12M"] = (
        df.groupby(tick_col)[_tmp_col2]
        .transform(lambda x: x.rolling(52, min_periods=20).apply(_info_disc, raw=True))
    )
    df.drop(columns=[_tmp_col2], inplace=True)
    print(f"       NaN: {df['Information_Discreteness_12M'].isna().mean():.1%}")

    new_features = [
        "Amihud_ILLIQ_20d",
        "MAX5_21d_neg",
        # Ratio_52w_High removida: corr=0.98 con Drawdown_52w existente (redundante)
        "Information_Discreteness_12M",
    ]

    # Cross-sectional ranks por fecha
    new_rank_features = []
    for feat in new_features:
        rcol = f"{feat}__rank"
        df[rcol] = df.groupby(date_col)[feat].rank(pct=True)
        new_rank_features.append(rcol)

    print(f"  [OK] 4 features nuevas + 4 __rank computadas.")
    return df, new_features, new_rank_features


# ╔══════════════════════════════════════════════════════════════════╗
# ║         LOAD + FEATURE ENGINEERING + HOLDOUT SPLIT              ║
# ╚══════════════════════════════════════════════════════════════════╝

# Constantes runtime — pobladas por load_and_prepare_data().
# Las funciones run_pipeline / run_single_backtest las leen como globales del módulo
# (alternativa a pasarlas por parámetro a cada función).
TICK_COL: Optional[str]      = None
DATE_COL: Optional[str]      = None
PRICE_COL: Optional[str]     = None
YCOL: Optional[str]          = None
SECTOR_COL: Optional[str]    = None
BASELINE_COL: Optional[str]  = None
RET_REAL_COL: Optional[str]  = None
GLOBAL_ANCHOR_DATE: Optional[pd.Timestamp] = None
HOLDOUT_DATES: set           = set()
TRAIN_DATES_POOL: set        = set()
HOLDOUT_N_WEEKS_ACTUAL: int  = HOLDOUT_N_WEEKS


class PipelineContext(NamedTuple):
    df: pd.DataFrame
    feature_cols_base: List[str]
    feature_cols_new: List[str]
    train_dates_pool: set
    holdout_dates: set
    anchor_date: pd.Timestamp
    new_raw_features: List[str]
    new_rank_features: List[str]


def load_and_prepare_data(file_path: Optional[Path] = None) -> PipelineContext:
    """
    Carga el dataset, normaliza columnas, computa features y separa holdout.

    Idempotente: dos llamadas con el mismo input producen el mismo PipelineContext
    (y dejan las constantes runtime — TICK_COL, RET_REAL_COL, TRAIN_DATES_POOL,
    etc. — pobladas en globals() del módulo). Las funciones run_pipeline /
    run_single_backtest las leen desde ahí.
    """
    global TICK_COL, DATE_COL, PRICE_COL, YCOL, SECTOR_COL, BASELINE_COL, RET_REAL_COL
    global GLOBAL_ANCHOR_DATE, HOLDOUT_DATES, TRAIN_DATES_POOL, HOLDOUT_N_WEEKS_ACTUAL

    fp = file_path or FILE_PATH

    print(f"Input hash: {INPUT_HASH}")
    df = pd.read_excel(fp)
    print(f"Datos cargados: {df.shape}")

    df.columns = [clean_colname(c) for c in df.columns]

    # Dedup columnas (v4 fix)
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated(keep=False)].unique().tolist()
        print(f"  [WARN] Columnas duplicadas eliminadas (keep=first): {dup}")
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    TICK_COL  = "Ticker"
    DATE_COL  = "Data_Date"
    PRICE_COL = "Close" if "Close" in df.columns else ("Precio" if "Precio" in df.columns else None)
    if PRICE_COL is None:
        raise KeyError("No encuentro columna Close ni Precio.")

    YCOL = "Target3" if "Target3" in df.columns else None
    if YCOL is None:
        raise KeyError("No encuentro columna Target3.")

    SECTOR_COL = "Sector" if "Sector" in df.columns else None

    BASELINE_COL = (
        "Predicted_Outperform_Prob"
        if "Predicted_Outperform_Prob" in df.columns
        else None
    )

    for c in [TICK_COL, DATE_COL, PRICE_COL, YCOL]:
        if c not in df.columns:
            raise KeyError(f"Falta columna '{c}'.")

    df[TICK_COL] = df[TICK_COL].astype(str).str.upper().str.strip()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["_DateKey"] = df[DATE_COL].dt.to_period("W").dt.end_time.dt.normalize()

    df = ensure_numeric(df, [PRICE_COL, YCOL])
    df[YCOL] = df[YCOL].apply(lambda x: np.nan if pd.isna(x) else (1 if float(x) >= 0.5 else 0))
    df["y_t3"] = df[YCOL].astype(float)

    RET_REAL_COL = next(
        (c for c in ["Ret_Una_sem", "Ret_1w_real", "Ret_Una_Semana"] if c in df.columns),
        None,
    )
    if RET_REAL_COL:
        df[RET_REAL_COL] = pd.to_numeric(df[RET_REAL_COL], errors="coerce")
        print(f"  [OK] Columna de retorno real para métricas: '{RET_REAL_COL}'")

    df = df.sort_values([TICK_COL, DATE_COL]).reset_index(drop=True)

    print("\n--- Date Consistency Diagnostic ---")
    diagnose_date_consistency(df, TICK_COL, DATE_COL)

    # ── FEATURE ENGINEERING ──────────────────────────────────────────
    leak_cols = set([
        "Precio_una_semana", "Ret_Una_sem", "Target", "Target_real",
        "Predicted_Outperform_Prob", "Ret_Next", "Next_Week_Return",
        "Forward_Return", "Target1", "Target2", "Target4", "Target5",
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
    base_features = [c for c in candidates if c in df.columns and c not in leak_cols]
    df = ensure_numeric(df, base_features)

    df["Close_prev"]   = gap_aware_shift(df, TICK_COL, "_DateKey", PRICE_COL, 1, 2)
    df["Ret_1w_calc"]  = (df[PRICE_COL] / df["Close_prev"]) - 1.0
    df["LogRet_1w"]    = np.log(df[PRICE_COL] / df["Close_prev"])
    df["GapPct"]       = df["Ret_1w_calc"].copy()
    df["GapPct"]       = df["GapPct"].replace([np.inf, -np.inf], np.nan)

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

    df["Ret_1w_prev"]   = gap_aware_shift(df, TICK_COL, "_DateKey", "Ret_1w_calc", 1, 2)
    df["MomAccel_1w"]   = df["Ret_1w_calc"] - df["Ret_1w_prev"]
    df["High_4w"]       = df.groupby(TICK_COL)[PRICE_COL].transform(
        lambda x: x.rolling(4, min_periods=2).max()
    )
    df["PriceToHigh4w"] = df[PRICE_COL] / df["High_4w"].replace(0, np.nan)
    df.drop(columns=["Ret_1w_prev", "High_4w"], inplace=True)

    vol_col_found = next((c for c in VOL_COL_CANDIDATES if c in df.columns), None)
    if vol_col_found:
        df[vol_col_found] = pd.to_numeric(df[vol_col_found], errors="coerce")
        vm = df.groupby(TICK_COL)[vol_col_found].transform(lambda x: x.rolling(8, min_periods=4).mean())
        vs = df.groupby(TICK_COL)[vol_col_found].transform(lambda x: x.rolling(8, min_periods=4).std())
        df["VolZScore_8w"] = (df[vol_col_found] - vm) / vs.replace(0, np.nan)
    else:
        df["VolZScore_8w"] = np.nan

    short_term_cols = ["MomAccel_1w", "PriceToHigh4w", "VolZScore_8w"]
    fe_new = ["GapPct", "Ret_1w_calc", "Vol_12w", "RiskAdjMom_3M", "RiskAdjMom_6M"]
    df = ensure_numeric(df, fe_new + short_term_cols)

    xs_pick_base = [c for c in [
        "RET_1M", "RET_3M", "RET_6M",
        "RS1M_SPY", "RS3M_SPY", "RS6M_SPY",
        "Momentum_Score", "RelStr_Score",
        "RiskAdjMom_3M", "RiskAdjMom_6M",
        "Vol_Rel_20", "Drawdown_52w", "GapPct", "Vol_12w",
        "MomAccel_1w", "PriceToHigh4w",
    ] if c in df.columns]

    xs_rank_cols_base = []
    for c in xs_pick_base[:25]:
        rcol = f"{c}__rank"
        df[rcol] = df.groupby("_DateKey")[c].rank(pct=True)
        xs_rank_cols_base.append(rcol)

    _all_base = base_features + fe_new + short_term_cols + xs_rank_cols_base
    feature_cols_base = [c for c in _all_base if c in df.columns and c not in leak_cols]
    feature_cols_base = list(dict.fromkeys(feature_cols_base))

    print(f"\nFeatures BASE: {len(feature_cols_base)}")

    print("\n--- Computando filtros de universo ---")
    df = compute_universe_masks(df, TICK_COL, PRICE_COL, ret_col="Ret_1w_calc")

    print("\n--- Computando 4 features nuevas ---")
    t0_fe = time.time()
    df, new_raw_features, new_rank_features = compute_illiquidity_momentum_features(
        df, TICK_COL, "_DateKey", PRICE_COL, ret_1w_col="Ret_1w_calc"
    )
    print(f"  Tiempo: {time.time() - t0_fe:.1f}s")

    feature_cols_new = list(dict.fromkeys(
        feature_cols_base + new_raw_features + new_rank_features
    ))
    feature_cols_new = [c for c in feature_cols_new if c in df.columns and c not in leak_cols]

    print(f"Features BASE: {len(feature_cols_base)}")
    print(f"Features NEW : {len(feature_cols_new)}")

    # ── ANCHOR DATE + HOLDOUT SPLIT ──────────────────────────────────
    last_dates = df.dropna(subset=["_DateKey"]).groupby(TICK_COL)["_DateKey"].max()
    anchor_cands = last_dates.value_counts()
    GLOBAL_ANCHOR_DATE = anchor_cands.index[0]
    print(f"\nAnchor global: {GLOBAL_ANCHOR_DATE.date()} "
          f"(coverage: {anchor_cands.iloc[0]/len(last_dates):.1%} de tickers)")

    all_unique_dates = sorted(df["_DateKey"].dropna().unique())
    if len(all_unique_dates) <= HOLDOUT_N_WEEKS + N_FOLDS * 3:
        HOLDOUT_N_WEEKS_ACTUAL = max(5, len(all_unique_dates) // 5)
        print(f"  [WARN] Pocos fechas; ajustando holdout a {HOLDOUT_N_WEEKS_ACTUAL} semanas.")
    else:
        HOLDOUT_N_WEEKS_ACTUAL = HOLDOUT_N_WEEKS

    HOLDOUT_DATES    = set(all_unique_dates[-HOLDOUT_N_WEEKS_ACTUAL:])
    TRAIN_DATES_POOL = set(all_unique_dates[:-HOLDOUT_N_WEEKS_ACTUAL])
    print(f"Holdout: {HOLDOUT_N_WEEKS_ACTUAL} fechas ({min(HOLDOUT_DATES).date()} "
          f"→ {max(HOLDOUT_DATES).date()})")
    print(f"Train pool: {len(TRAIN_DATES_POOL)} fechas")

    return PipelineContext(
        df=df,
        feature_cols_base=feature_cols_base,
        feature_cols_new=feature_cols_new,
        train_dates_pool=TRAIN_DATES_POOL,
        holdout_dates=HOLDOUT_DATES,
        anchor_date=GLOBAL_ANCHOR_DATE,
        new_raw_features=new_raw_features,
        new_rank_features=new_rank_features,
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        CORE PIPELINE                            ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_pipeline(
    df_all: pd.DataFrame,
    feature_cols: list,
    universe_mask_col: str = "_univ_close5",
    out_file: Optional[Path] = None,
    run_label: str = "RUN",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parámetros
    ----------
    universe_mask_col : columna booleana para filtrar universo.
        '_univ_close5' = baseline (solo Close>$5)
        '_univ_pass'   = filtros completos (Close>$5 + DolVol + ZeroRet)
    Retorna
    -------
    holdout_scored : df del holdout con scores del modelo final
    feat_imp_df    : feature importance del clf_final
    metrics        : dict con todas las métricas
    """
    print("\n" + "=" * 80)
    print(f"RUN: {run_label}  |  universe_mask: {universe_mask_col}  "
          f"|  features: {len(feature_cols)}")
    print("=" * 80)
    t_run_start = time.time()

    # ── Aplicar máscara de universo ──────────────────────────────
    mask = df_all[universe_mask_col].fillna(False)
    df_filtered = df_all[mask].copy()

    # ── Separar train / holdout ──────────────────────────────────
    train_df = df_filtered[
        df_filtered["_DateKey"].isin(TRAIN_DATES_POOL) & df_filtered["y_t3"].notna()
    ].copy()
    train_df = train_df.dropna(subset=["_DateKey"])
    train_df["y_t3_int"] = train_df["y_t3"].astype(int)
    train_df = train_df.reset_index(drop=True)

    holdout_df_raw = df_filtered[df_filtered["_DateKey"].isin(HOLDOUT_DATES)].copy()

    print(f"[{run_label}] Train rows  : {len(train_df):,}")
    print(f"[{run_label}] Train tickers: {train_df[TICK_COL].nunique():,}")
    print(f"[{run_label}] Holdout rows : {len(holdout_df_raw):,}")

    if len(train_df) == 0:
        raise ValueError(f"[{run_label}] Sin filas de entrenamiento tras filtro.")
    if len(holdout_df_raw) == 0:
        raise ValueError(f"[{run_label}] Sin filas en holdout tras filtro.")

    unique_dates = sorted(train_df["_DateKey"].unique())
    splits = make_walkforward_splits(
        unique_dates, N_FOLDS, EMBARGO_DATES, MIN_TRAIN_DATES, TEST_DATES_PER_FOLD
    )
    print(f"[{run_label}] Walk-forward splits: {len(splits)}")

    # Diagnóstico solapamiento de fechas de validación
    all_val_dates = set()
    for _, va in splits:
        overlap = all_val_dates & set(va)
        if overlap:
            print(f"  [WARN] VALIDACIÓN SOLAPADA: {len(overlap)} fechas")
        all_val_dates.update(va)
    print(f"  [OK] Fechas de validación únicas: {len(all_val_dates)}")

    oof_clf = np.full(len(train_df), np.nan)
    oof_rnk = np.full(len(train_df), np.nan)
    best_iters_clf, best_iters_rnk, fold_rows = [], [], []

    # ── Walk-forward folds ───────────────────────────────────────
    for k, (tr_dates, te_dates) in enumerate(splits, 1):
        tr = train_df[train_df["_DateKey"].isin(tr_dates)].copy()
        te = train_df[train_df["_DateKey"].isin(te_dates)].copy()

        if len(tr) < 2000 or len(te) < 200:
            print(f"[{run_label}] Fold {k}: SKIP (tr={len(tr)}, te={len(te)})")
            continue

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(tr[feature_cols].replace([np.inf, -np.inf], np.nan))
        X_te = imp.transform(te[feature_cols].replace([np.inf, -np.inf], np.nan))
        y_tr, y_te = tr["y_t3_int"].values, te["y_t3_int"].values

        w_tr = time_decay_weights(tr["_DateKey"], HALF_LIFE_WEEKS).values if USE_TIME_DECAY else None

        spw = max(1, int((y_tr == 0).sum())) / max(1, int((y_tr == 1).sum()))

        # CLF
        clf = build_clf(spw, SEED)
        clf.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_te, y_te)], eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        p_te = clf.predict_proba(X_te)[:, 1]
        oof_clf[te.index.values] = p_te
        best_iters_clf.append(getattr(clf, "best_iteration_", None))

        # RANKER (multi-level relevance, FIX B1)
        tr_s = tr.sort_values(["_DateKey", TICK_COL])
        te_s = te.sort_values(["_DateKey", TICK_COL])
        g_tr = tr_s.groupby("_DateKey").size().values
        g_te = te_s.groupby("_DateKey").size().values
        X_tr_r = imp.transform(tr_s[feature_cols].replace([np.inf, -np.inf], np.nan))
        X_te_r = imp.transform(te_s[feature_cols].replace([np.inf, -np.inf], np.nan))

        y_tr_rel = compute_relevance_labels(tr_s["y_t3_int"].values, clf.predict_proba(X_tr_r)[:, 1])
        y_te_rel = compute_relevance_labels(te_s["y_t3_int"].values, clf.predict_proba(X_te_r)[:, 1])

        rnk = build_ranker(SEED)
        rnk.fit(
            X_tr_r, y_tr_rel, group=g_tr,
            eval_set=[(X_te_r, y_te_rel)], eval_group=[g_te],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        s_te = rnk.predict(X_te_r)
        oof_rnk[te.index.values] = pd.Series(s_te, index=te_s.index).loc[te.index].values
        best_iters_rnk.append(getattr(rnk, "best_iteration_", None))

        # Búsqueda de peso por fold (anidada, sin look-ahead)
        te_eval = te[["_DateKey", "y_t3_int"]].copy()
        te_eval["Prob_Clf"]     = p_te
        te_eval["Score_Ranker"] = oof_rnk[te.index.values]
        te_eval["RankerPct"]    = te_eval.groupby("_DateKey")["Score_Ranker"].rank(pct=True)

        best_w_fold, best_p_fold = W_CLF_INIT, -1.0
        for w in np.arange(0.35, 0.86, 0.05):
            tmp = te_eval.copy()
            tmp["S"] = w * tmp["Prob_Clf"] + (1 - w) * tmp["RankerPct"]
            p = precision_at_k_by_date(tmp, "_DateKey", "y_t3_int", "S", TOP_K)
            if p > best_p_fold:
                best_p_fold, best_w_fold = p, float(w)

        te_eval["RankScore"] = best_w_fold * te_eval["Prob_Clf"] + (1 - best_w_fold) * te_eval["RankerPct"]
        auc = roc_auc_score(y_te, p_te)
        ap  = average_precision_score(y_te, p_te)
        p20 = precision_at_k_by_date(te_eval, "_DateKey", "y_t3_int", "RankScore", TOP_K)

        fold_rows.append([k, len(tr), len(te), auc, ap, p20, best_w_fold])
        print(f"[{run_label}] Fold {k}: AUC={auc:.4f}  PR-AUC={ap:.4f}  "
              f"P@{TOP_K}={p20:.3f}  W_CLF={best_w_fold:.2f}")

    # ── OOF aggregation ─────────────────────────────────────────
    valid = np.isfinite(oof_clf) & np.isfinite(oof_rnk)
    if valid.sum() == 0:
        raise ValueError(f"[{run_label}] Sin predicciones OOF válidas.")

    y_oof  = train_df.loc[valid, "y_t3_int"].values
    df_oof = train_df.loc[valid, ["_DateKey"]].copy()
    df_oof["y"]           = y_oof
    df_oof["Prob_Clf"]    = oof_clf[valid]
    df_oof["Score_Ranker"]= oof_rnk[valid]
    df_oof["RankerPct"]   = df_oof.groupby("_DateKey")["Score_Ranker"].rank(pct=True)

    # Peso final = mediana de pesos por fold (no re-optimizado en OOF completo)
    fold_weights = [r[6] for r in fold_rows]
    W_CLF = float(np.median(fold_weights)) if fold_weights else W_CLF_INIT
    W_RNK = 1.0 - W_CLF
    df_oof["RankScore"] = W_CLF * df_oof["Prob_Clf"] + W_RNK * df_oof["RankerPct"]

    print(f"\n[{run_label}] Pesos (mediana folds): W_CLF={W_CLF:.2f} | W_RNK={W_RNK:.2f}")

    # Platt calibration
    platt = LogisticRegression(max_iter=2000, random_state=SEED)
    platt.fit(df_oof[["RankScore"]].values, y_oof)

    auc_oof_clf   = roc_auc_score(y_oof, df_oof["Prob_Clf"].values)
    auc_oof_final = roc_auc_score(y_oof, df_oof["RankScore"].values)
    ap_oof        = average_precision_score(y_oof, df_oof["RankScore"].values)
    base_rate     = float(y_oof.mean())
    oof_p20       = precision_at_k_by_date(df_oof, "_DateKey", "y", "RankScore", TOP_K)
    oof_p10       = precision_at_k_by_date(df_oof, "_DateKey", "y", "RankScore", 10)

    # Test de permutación (500 perms para MVP — aumentar a 1000 en producción)
    _, _, pval = permutation_test_p_at_k(df_oof, "_DateKey", "y", "RankScore", TOP_K, 500, SEED)

    print(f"[{run_label}] OOF AUC(Clf)={auc_oof_clf:.4f} | AUC(Final)={auc_oof_final:.4f}")
    print(f"[{run_label}] OOF P@{TOP_K}={oof_p20:.3f}  P@10={oof_p10:.3f}  pval={pval:.4f}")
    print(f"[{run_label}] BaseRate={base_rate:.2%}")

    # ── Modelo final (fit en todos los datos de train) ───────────
    n_clf = safe_median_best(best_iters_clf, 2500)
    n_rnk = safe_median_best(best_iters_rnk, 1200)

    imp_final = SimpleImputer(strategy="median")
    X_full = imp_final.fit_transform(train_df[feature_cols].replace([np.inf, -np.inf], np.nan))
    y_full = train_df["y_t3_int"].values
    w_full = time_decay_weights(train_df["_DateKey"], HALF_LIFE_WEEKS).values if USE_TIME_DECAY else None
    spw_full = max(1, int((y_full == 0).sum())) / max(1, int((y_full == 1).sum()))

    clf_final = LGBMClassifier(
        objective="binary", n_estimators=n_clf, learning_rate=0.02,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.3, reg_lambda=0.6, min_child_samples=70,
        scale_pos_weight=spw_full, n_jobs=-1, random_state=SEED,
    )
    clf_final.fit(X_full, y_full, sample_weight=w_full)

    # Feature importance (con guard de dedup)
    n_fi = len(clf_final.feature_importances_)
    fi_names = (
        feature_cols if n_fi == len(feature_cols)
        else (feature_cols + [f"_extra_{i}" for i in range(n_fi)])[:n_fi]
    )
    feat_imp_df = (
        pd.DataFrame({"Feature": fi_names, "Importance": clf_final.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    feat_imp_df["Rank"] = np.arange(1, len(feat_imp_df) + 1)

    train_s = train_df.sort_values(["_DateKey", TICK_COL])
    X_full_r = imp_final.transform(train_s[feature_cols].replace([np.inf, -np.inf], np.nan))
    y_full_rel = compute_relevance_labels(
        train_s["y_t3_int"].values, clf_final.predict_proba(X_full_r)[:, 1]
    )
    g_full = train_s.groupby("_DateKey").size().values

    rnk_final = LGBMRanker(
        objective="lambdarank", n_estimators=n_rnk, learning_rate=0.03,
        num_leaves=63, min_data_in_leaf=60, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=0.8, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    rnk_final.fit(X_full_r, y_full_rel, group=g_full)

    # ── Scoring del holdout ──────────────────────────────────────
    holdout_df = holdout_df_raw.copy()
    holdout_df = holdout_df.dropna(subset=["_DateKey", "y_t3"])
    holdout_df["y_t3_int"] = holdout_df["y_t3"].astype(int)

    if len(holdout_df) == 0:
        raise ValueError(f"[{run_label}] Holdout vacío tras filtros.")

    X_hold = imp_final.transform(
        holdout_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    )
    p_hold = clf_final.predict_proba(X_hold)[:, 1]

    hold_s = holdout_df.sort_values(["_DateKey", TICK_COL])
    X_hold_r = imp_final.transform(hold_s[feature_cols].replace([np.inf, -np.inf], np.nan))
    s_hold_r = rnk_final.predict(X_hold_r)
    s_hold_aligned = pd.Series(s_hold_r, index=hold_s.index).loc[holdout_df.index].values

    holdout_scored = holdout_df[[TICK_COL, "_DateKey", PRICE_COL, "y_t3_int"]].copy()
    if RET_REAL_COL and RET_REAL_COL in holdout_df.columns:
        holdout_scored[RET_REAL_COL] = holdout_df[RET_REAL_COL].values
    if SECTOR_COL and SECTOR_COL in holdout_df.columns:
        holdout_scored[SECTOR_COL] = holdout_df[SECTOR_COL].values

    holdout_scored["Prob_Clf"]     = p_hold
    holdout_scored["Score_Ranker"] = s_hold_aligned
    holdout_scored["RankerPct"]    = holdout_scored.groupby("_DateKey")["Score_Ranker"].rank(pct=True)
    holdout_scored["RankScore"]    = W_CLF * holdout_scored["Prob_Clf"] + W_RNK * holdout_scored["RankerPct"]
    holdout_scored["Prob_T3_FINAL"]= platt.predict_proba(holdout_scored[["RankScore"]].values)[:, 1]

    # ── Métricas de holdout ──────────────────────────────────────
    hold_p20  = precision_at_k_by_date(holdout_scored, "_DateKey", "y_t3_int", "RankScore", TOP_K)
    hold_p10  = precision_at_k_by_date(holdout_scored, "_DateKey", "y_t3_int", "RankScore", 10)
    hold_ndcg = ndcg_at_k_by_date(holdout_scored, "_DateKey", "y_t3_int", "RankScore", TOP_K)
    hold_lift = hold_p20 / base_rate if base_rate > 0 else np.nan
    oof_hold_gap = oof_p20 - hold_p20

    # Avg Ret Top20 - Bot20
    if RET_REAL_COL and RET_REAL_COL in holdout_scored.columns:
        try:
            ret_top20 = holdout_scored.groupby("_DateKey").apply(
                lambda g: g.nlargest(TOP_K, "RankScore")[RET_REAL_COL].mean()
            ).mean()
            ret_bot20 = holdout_scored.groupby("_DateKey").apply(
                lambda g: g.nsmallest(TOP_K, "RankScore")[RET_REAL_COL].mean()
            ).mean()
            avg_ret_spread = float(ret_top20 - ret_bot20)
        except (TypeError, ValueError):
            avg_ret_spread = np.nan
    else:
        # Proxy: diferencia de tasas de acierto
        p_top20 = hold_p20
        p_bot20 = precision_at_k_by_date(
            holdout_scored.copy().assign(RankScore=-holdout_scored["RankScore"]),
            "_DateKey", "y_t3_int", "RankScore", TOP_K,
        )
        avg_ret_spread = float(p_top20 - p_bot20)

    # Hit rate Top20 vs SPY (si SPY tiene retorno en holdout)
    # groupby+mean garantiza índice único → .get() devuelve scalar, no Series
    spy_mask = holdout_scored[TICK_COL] == "SPY"
    spy_rets  = holdout_scored[spy_mask]
    if RET_REAL_COL and RET_REAL_COL in holdout_scored.columns and len(spy_rets):
        spy_ret_by_date = spy_rets.groupby("_DateKey")[RET_REAL_COL].mean()
        hit_rates = []
        for dt, g in holdout_scored.groupby("_DateKey"):
            if RET_REAL_COL not in g.columns:
                continue
            top20 = g.nlargest(TOP_K, "RankScore")
            spy_r_raw = spy_ret_by_date.get(dt, np.nan)
            # .get() en Series con índice único devuelve scalar; float() lo fuerza
            spy_r = float(spy_r_raw) if pd.notna(spy_r_raw) else np.nan
            if not np.isnan(spy_r):
                hit_rates.append(float((top20[RET_REAL_COL] > spy_r).mean()))
        hit_rate_vs_spy = float(np.mean(hit_rates)) if hit_rates else np.nan
    else:
        hit_rate_vs_spy = np.nan

    # HHI sectorial Top20
    hhi_top20 = np.nan
    if SECTOR_COL and SECTOR_COL in holdout_scored.columns:
        hhi_per_week = holdout_scored.groupby("_DateKey").apply(
            lambda g: hhi_sector(g.nlargest(TOP_K, "RankScore"), SECTOR_COL)
        )
        hhi_top20 = float(hhi_per_week.mean())

    # Rank stability: corr(rank_predicho, rank_actual) por semana
    rank_stabs = []
    for _, g in holdout_scored.groupby("_DateKey"):
        if len(g) < 5:
            continue
        pred_rank   = g["RankScore"].rank(ascending=False)
        actual_rank = g["y_t3_int"].rank(ascending=False)
        rs = pred_rank.corr(actual_rank, method="spearman")
        if pd.notna(rs):
            rank_stabs.append(rs)
    rank_stability = float(np.mean(rank_stabs)) if rank_stabs else np.nan

    # P@20 por semana del holdout
    p20_by_week = {}
    for dt, g in holdout_scored.groupby("_DateKey"):
        top = g.nlargest(TOP_K, "RankScore")
        p20_by_week[dt] = float(top["y_t3_int"].mean()) if len(top) else np.nan

    print(f"[{run_label}] Holdout P@{TOP_K}={hold_p20:.3f}  P@10={hold_p10:.3f}  "
          f"NDCG@{TOP_K}={hold_ndcg:.3f}  Lift={hold_lift:.2f}x")
    print(f"[{run_label}] OOF-Holdout gap: {oof_hold_gap:+.3f}")
    t_total = time.time() - t_run_start

    # ── Scoring del último cierre ────────────────────────────────
    df_last = df_all[df_all["_DateKey"] == GLOBAL_ANCHOR_DATE].copy()
    df_last = df_last.sort_values([TICK_COL, DATE_COL]).drop_duplicates(TICK_COL, keep="last")
    # Aplicar filtro de universo al scoring final
    df_last = df_last[df_last[universe_mask_col].fillna(False)].copy()
    total_tickers = df_all[TICK_COL].nunique()

    nan_frac = df_last[feature_cols].isna().mean(axis=1)
    df_last = df_last[nan_frac <= MAX_NAN_FRAC_LAST].copy()

    if len(df_last) == 0:
        print(f"  [WARN] Sin tickers en último cierre para {run_label}.")
        out_df = pd.DataFrame()
    else:
        X_last = imp_final.transform(df_last[feature_cols].replace([np.inf, -np.inf], np.nan))
        p_last = clf_final.predict_proba(X_last)[:, 1]

        dl_s = df_last.sort_values([TICK_COL])
        X_last_r = imp_final.transform(dl_s[feature_cols].replace([np.inf, -np.inf], np.nan))
        s_last_r = rnk_final.predict(X_last_r)
        s_last_aligned = pd.Series(s_last_r, index=dl_s.index).loc[df_last.index].values

        out_df = df_last[[TICK_COL, DATE_COL, PRICE_COL]].copy()
        if BASELINE_COL and BASELINE_COL in df_last.columns:
            out_df["Baseline_Prob"] = pd.to_numeric(df_last[BASELINE_COL], errors="coerce")
        out_df["Prob_Clf"]      = p_last
        out_df["Score_Ranker"]  = s_last_aligned
        out_df["RankerPct"]     = out_df["Score_Ranker"].rank(pct=True)
        out_df["RankScore"]     = W_CLF * out_df["Prob_Clf"] + W_RNK * out_df["RankerPct"]
        out_df["Prob_T3_FINAL"] = platt.predict_proba(out_df[["RankScore"]].values)[:, 1]
        out_df = out_df.sort_values("RankScore", ascending=False).reset_index(drop=True)
        out_df.insert(0, "Rank", np.arange(1, len(out_df) + 1))

    # ── Export ───────────────────────────────────────────────────
    if out_file and len(out_df) > 0:
        topk     = out_df.head(TOP_K)
        bottom_k = out_df.tail(TOP_K).sort_values("RankScore").reset_index(drop=True)
        fold_df  = pd.DataFrame(
            fold_rows,
            columns=["Fold", "TrainRows", "TestRows", "AUC", "PR_AUC", f"P@{TOP_K}", "W_CLF_fold"],
        )
        summary = pd.DataFrame({
            "Metric": [
                "RunLabel", "PipelineVersion", "InputHash", "UniverseMask",
                "N_Features", "TrainRows", "TrainTickers",
                "HoldoutWeeks", "Holdout_P@20", "Holdout_P@10", "Holdout_NDCG@20",
                "Holdout_Lift", "OOF_P@20", "OOF_AUC_Final", "OOF_PR_AUC",
                "BaseRate", "Permutation_pval", "W_CLF", "W_RNK",
                "TrainingTime_s",
            ],
            "Value": [
                run_label, "v5.0", INPUT_HASH, universe_mask_col,
                len(feature_cols), len(train_df), train_df[TICK_COL].nunique(),
                HOLDOUT_N_WEEKS_ACTUAL, hold_p20, hold_p10, hold_ndcg,
                hold_lift, oof_p20, auc_oof_final, ap_oof,
                base_rate, pval, W_CLF, W_RNK,
                round(t_total, 1),
            ],
        })
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            topk.to_excel(writer, sheet_name=f"Top_{TOP_K}_Last", index=False)
            bottom_k.to_excel(writer, sheet_name=f"Bottom_{TOP_K}_Short", index=False)
            out_df.to_excel(writer, sheet_name="All_Last", index=False)
            holdout_scored.to_excel(writer, sheet_name="Holdout_Scored", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
            fold_df.to_excel(writer, sheet_name="Fold_Metrics", index=False)
            feat_imp_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
        print(f"[{run_label}] Exportado: {out_file}")

    metrics = {
        "RunLabel":        run_label,
        "UniverseMask":    universe_mask_col,
        "N_Features":      len(feature_cols),
        "TrainRows":       len(train_df),
        "TrainTickers":    int(train_df[TICK_COL].nunique()),
        "BaseRate":        base_rate,
        "OOF_P20":         oof_p20,
        "OOF_P10":         oof_p10,
        "Holdout_P20":     hold_p20,
        "Holdout_P10":     hold_p10,
        "Holdout_NDCG20":  hold_ndcg,
        "Holdout_Lift":    hold_lift,
        "AvgRet_Spread":   avg_ret_spread,
        "HitRate_vs_SPY":  hit_rate_vs_spy,
        "HHI_Sectorial":   hhi_top20,
        "OOF_Hold_Gap":    oof_hold_gap,
        "RankStability":   rank_stability,
        "Perm_pval":       pval,
        "TrainingTime_s":  round(t_total, 1),
        "N_Trees_clf":     n_clf,
        "N_Trees_rnk":     n_rnk,
    }

    return holdout_scored, feat_imp_df, metrics, p20_by_week


# ╔══════════════════════════════════════════════════════════════════╗
# ║              BACKTEST WALK-FORWARD (v6.0)                       ║
# ╚══════════════════════════════════════════════════════════════════╝

def _compute_attribution_metrics(
    scored_df: pd.DataFrame,
    spy_ret_val: float,
    anchor_date: pd.Timestamp,
) -> dict:
    """
    Métricas de atribución calculadas sobre el universo COMPLETO rankeado,
    ANTES de cortar Top-K. scored_df debe tener RankScore y Ret_Real_NextWeek
    en su índice original (sin reset_index, sin sort previo).

    Métricas:
      spearman        : corr Spearman(RankScore, Ret_Real_NextWeek) — KPI skill ranking
      decile_spread   : ret promedio decil 10 − decil 1 (>0 = modelo discrimina)
      top{1,3,5,10,20}_ret : equal-weight por concentración de posición
      top20_lw_ret    : linear-weighted (pesos ∝ 21-rank, normalizados)
      top3_beat_spy / top10_beat_spy : hit-rate vs SPY
    """
    def _r(v):
        try:
            fv = float(v)
            return round(fv, 4) if np.isfinite(fv) else np.nan
        except (TypeError, ValueError):
            return np.nan

    rs  = scored_df["RankScore"].values.astype(float)
    ret = pd.to_numeric(scored_df.get("Ret_Real_NextWeek", pd.Series(dtype=float)),
                        errors="coerce").values.astype(float)

    valid = np.isfinite(rs) & np.isfinite(ret)
    n_valid = int(valid.sum())
    rs_v, ret_v = rs[valid], ret[valid]

    # 1) Spearman
    if n_valid >= 20:
        spearman = float(spearmanr(rs_v, ret_v).correlation)
    else:
        spearman = np.nan

    # 2) Decile spread (sobre subconjunto válido)
    decile_top_ret = decile_bot_ret = decile_spread = np.nan
    if n_valid >= 20:
        try:
            labels = pd.qcut(pd.Series(rs_v), 10, labels=False, duplicates="drop")
            groups = pd.Series(ret_v).groupby(labels).mean()
            if len(groups) >= 2:
                decile_top_ret = float(groups.iloc[-1])
                decile_bot_ret = float(groups.iloc[0])
                decile_spread  = decile_top_ret - decile_bot_ret
        except Exception:
            pass

    # 3) Position-sized returns (ordenar por RankScore desc, incluyendo NaN-ret)
    sort_idx   = np.argsort(-rs)          # orden descendente por score
    ret_sorted = ret[sort_idx]            # retornos en ese orden

    def _topn(n):
        sub = ret_sorted[:n]
        sub = sub[np.isfinite(sub)]
        return float(np.mean(sub)) if len(sub) else np.nan

    top1_ret  = _topn(1)
    top3_ret  = _topn(3)
    top5_ret  = _topn(5)
    top10_ret = _topn(10)
    top20_ret = _topn(20)

    # Linear-weighted Top-20: w ∝ 21 − rank (rank 1 = peso mayor), normalizados
    sub20    = ret_sorted[:20]
    w_lw     = np.arange(20, 0, -1, dtype=float)
    valid_lw = np.isfinite(sub20)
    if valid_lw.any():
        w = w_lw[valid_lw]; w = w / w.sum()
        top20_lw_ret = float(np.dot(sub20[valid_lw], w))
    else:
        top20_lw_ret = np.nan

    # 4) Hit rate vs SPY para top-3 y top-10
    def _beat_spy(n):
        if np.isnan(spy_ret_val):
            return np.nan
        sub = ret_sorted[:n]
        sub = sub[np.isfinite(sub)]
        return float(np.mean(sub > spy_ret_val)) if len(sub) else np.nan

    return {
        "Anchor_Date":    str(anchor_date.date()),
        "n_valid":        n_valid,
        "spearman":       _r(spearman),
        "decile_top_ret": _r(decile_top_ret),
        "decile_bot_ret": _r(decile_bot_ret),
        "decile_spread":  _r(decile_spread),
        "top1_ret":       _r(top1_ret),
        "top3_ret":       _r(top3_ret),
        "top5_ret":       _r(top5_ret),
        "top10_ret":      _r(top10_ret),
        "top20_ret":      _r(top20_ret),
        "top20_lw_ret":   _r(top20_lw_ret),
        "top3_beat_spy":  _r(_beat_spy(3)),
        "top10_beat_spy": _r(_beat_spy(10)),
        "spy_ret":        _r(spy_ret_val),
    }


def run_single_backtest(
    df_all: pd.DataFrame,
    anchor_date: pd.Timestamp,
    feature_cols: list,
    universe_mask_col: str,
    run_label: str,
) -> Optional[pd.DataFrame]:
    """
    Entrena con datos ESTRICTAMENTE anteriores a anchor_date y puntúa anchor_date.

    Regla de no look-ahead:
      - y_t3 en fecha X se observa en X+1w → sólo disponible para fechas <= anchor_date - 7d.
      - features ya son point-in-time (computadas en el df global por fecha).
      - ranks cross-sectional: se computan por fecha en el df global → sin leakage cross-date.

    Retorna DataFrame Top-20 o None si la fecha no existe o hay < 50 tickers.
    """
    cutoff = anchor_date - pd.Timedelta(weeks=1)

    # ── Universo para esta anchor_date ──────────────────────────
    mask = df_all[universe_mask_col].fillna(False)
    df_filt = df_all[mask].copy()

    # ── Scoring set: sólo rows en anchor_date ────────────────────
    scoring_df = df_filt[df_filt["_DateKey"] == anchor_date].copy()
    if len(scoring_df) == 0:
        print(f"  [BT {run_label}] anchor={anchor_date.date()} → SKIP: fecha no existe en _DateKey.")
        return None, {}
    if len(scoring_df) < 50:
        print(f"  [BT {run_label}] anchor={anchor_date.date()} → WARNING: universo={len(scoring_df)} < 50 tickers.")

    # ── Train set: fechas <= cutoff con y_t3 conocido ───────────
    train_df = df_filt[
        (df_filt["_DateKey"] <= cutoff) & df_filt["y_t3"].notna()
    ].copy()
    train_df = train_df.dropna(subset=["_DateKey"])
    train_df["y_t3_int"] = train_df["y_t3"].astype(int)
    train_df = train_df.reset_index(drop=True)

    # ── Assert obligatorio: sin look-ahead ───────────────────────
    assert train_df["_DateKey"].max() <= cutoff, \
        f"LOOK-AHEAD DETECTED at anchor={anchor_date.date()}"

    if len(train_df) < 2000:
        print(f"  [BT {run_label}] anchor={anchor_date.date()} → SKIP: train muy pequeño ({len(train_df)} rows).")
        return None, {}

    unique_dates = sorted(train_df["_DateKey"].unique())
    splits = make_walkforward_splits(
        unique_dates, N_FOLDS, EMBARGO_DATES, MIN_TRAIN_DATES, TEST_DATES_PER_FOLD
    )
    if not splits:
        print(f"  [BT {run_label}] anchor={anchor_date.date()} → SKIP: sin splits walk-forward.")
        return None, {}

    oof_clf = np.full(len(train_df), np.nan)
    oof_rnk = np.full(len(train_df), np.nan)
    best_iters_clf, best_iters_rnk, fold_weights = [], [], []

    for k, (tr_dates, te_dates) in enumerate(splits, 1):
        tr = train_df[train_df["_DateKey"].isin(tr_dates)].copy()
        te = train_df[train_df["_DateKey"].isin(te_dates)].copy()
        if len(tr) < 2000 or len(te) < 200:
            continue

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(tr[feature_cols].replace([np.inf, -np.inf], np.nan))
        X_te = imp.transform(te[feature_cols].replace([np.inf, -np.inf], np.nan))
        y_tr, y_te = tr["y_t3_int"].values, te["y_t3_int"].values
        w_tr = time_decay_weights(tr["_DateKey"], HALF_LIFE_WEEKS).values if USE_TIME_DECAY else None
        spw = max(1, int((y_tr == 0).sum())) / max(1, int((y_tr == 1).sum()))

        clf = build_clf(spw, SEED)
        clf.fit(
            X_tr, y_tr, sample_weight=w_tr,
            eval_set=[(X_te, y_te)], eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        p_te = clf.predict_proba(X_te)[:, 1]
        oof_clf[te.index.values] = p_te
        best_iters_clf.append(getattr(clf, "best_iteration_", None))

        tr_s = tr.sort_values(["_DateKey", TICK_COL])
        te_s = te.sort_values(["_DateKey", TICK_COL])
        g_tr = tr_s.groupby("_DateKey").size().values
        g_te = te_s.groupby("_DateKey").size().values
        X_tr_r = imp.transform(tr_s[feature_cols].replace([np.inf, -np.inf], np.nan))
        X_te_r = imp.transform(te_s[feature_cols].replace([np.inf, -np.inf], np.nan))
        y_tr_rel = compute_relevance_labels(tr_s["y_t3_int"].values, clf.predict_proba(X_tr_r)[:, 1])
        y_te_rel = compute_relevance_labels(te_s["y_t3_int"].values, clf.predict_proba(X_te_r)[:, 1])

        rnk = build_ranker(SEED)
        rnk.fit(
            X_tr_r, y_tr_rel, group=g_tr,
            eval_set=[(X_te_r, y_te_rel)], eval_group=[g_te],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        s_te = rnk.predict(X_te_r)
        oof_rnk[te.index.values] = pd.Series(s_te, index=te_s.index).loc[te.index].values
        best_iters_rnk.append(getattr(rnk, "best_iteration_", None))

        te_eval = te[["_DateKey", "y_t3_int"]].copy()
        te_eval["Prob_Clf"]     = p_te
        te_eval["Score_Ranker"] = oof_rnk[te.index.values]
        te_eval["RankerPct"]    = te_eval.groupby("_DateKey")["Score_Ranker"].rank(pct=True)
        best_w_fold, best_p_fold = W_CLF_INIT, -1.0
        for w in np.arange(0.35, 0.86, 0.05):
            tmp = te_eval.copy()
            tmp["S"] = w * tmp["Prob_Clf"] + (1 - w) * tmp["RankerPct"]
            p = precision_at_k_by_date(tmp, "_DateKey", "y_t3_int", "S", TOP_K)
            if p > best_p_fold:
                best_p_fold, best_w_fold = p, float(w)
        fold_weights.append(best_w_fold)

    valid = np.isfinite(oof_clf) & np.isfinite(oof_rnk)
    if valid.sum() == 0:
        print(f"  [BT {run_label}] anchor={anchor_date.date()} → SKIP: sin OOF válidas.")
        return None, {}

    y_oof  = train_df.loc[valid, "y_t3_int"].values
    df_oof = train_df.loc[valid, ["_DateKey"]].copy()
    df_oof["Prob_Clf"]     = oof_clf[valid]      # índice original de train_df
    df_oof["Score_Ranker"] = oof_rnk[valid]      # ídem → alineación correcta
    df_oof["RankerPct"]    = df_oof.groupby("_DateKey")["Score_Ranker"].rank(pct=True)

    W_CLF_bt = float(np.median(fold_weights)) if fold_weights else W_CLF_INIT
    W_RNK_bt = 1.0 - W_CLF_bt
    df_oof["RankScore"] = W_CLF_bt * df_oof["Prob_Clf"] + W_RNK_bt * df_oof["RankerPct"]

    # Platt calibration en OOF
    platt_bt = LogisticRegression(max_iter=2000, random_state=SEED)
    platt_bt.fit(df_oof[["RankScore"]].values, y_oof)

    # ── Modelo final entrenado en TODOS los datos de train ───────
    n_clf = safe_median_best(best_iters_clf, 2500)
    n_rnk = safe_median_best(best_iters_rnk, 1200)

    imp_final = SimpleImputer(strategy="median")
    X_full = imp_final.fit_transform(train_df[feature_cols].replace([np.inf, -np.inf], np.nan))
    y_full = train_df["y_t3_int"].values
    w_full = time_decay_weights(train_df["_DateKey"], HALF_LIFE_WEEKS).values if USE_TIME_DECAY else None
    spw_full = max(1, int((y_full == 0).sum())) / max(1, int((y_full == 1).sum()))

    clf_final = LGBMClassifier(
        objective="binary", n_estimators=n_clf, learning_rate=0.02,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.3, reg_lambda=0.6, min_child_samples=70,
        scale_pos_weight=spw_full, n_jobs=-1, random_state=SEED,
    )
    clf_final.fit(X_full, y_full, sample_weight=w_full)

    train_s = train_df.sort_values(["_DateKey", TICK_COL])
    X_full_r = imp_final.transform(train_s[feature_cols].replace([np.inf, -np.inf], np.nan))
    y_full_rel = compute_relevance_labels(
        train_s["y_t3_int"].values, clf_final.predict_proba(X_full_r)[:, 1]
    )
    g_full = train_s.groupby("_DateKey").size().values

    rnk_final = LGBMRanker(
        objective="lambdarank", n_estimators=n_rnk, learning_rate=0.03,
        num_leaves=63, min_data_in_leaf=60, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.5, reg_lambda=0.8, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    rnk_final.fit(X_full_r, y_full_rel, group=g_full)

    # ── Scoring en anchor_date ────────────────────────────────────
    scoring_df = scoring_df.copy()
    nan_frac = scoring_df[feature_cols].isna().mean(axis=1)
    scoring_df = scoring_df[nan_frac <= MAX_NAN_FRAC_LAST].copy()

    if len(scoring_df) == 0:
        print(f"  [BT {run_label}] anchor={anchor_date.date()} → SKIP: scoring vacío tras NaN filter.")
        return None, {}

    X_sc = imp_final.transform(scoring_df[feature_cols].replace([np.inf, -np.inf], np.nan))
    p_sc = clf_final.predict_proba(X_sc)[:, 1]

    sc_s  = scoring_df.sort_values(TICK_COL)
    X_sc_r = imp_final.transform(sc_s[feature_cols].replace([np.inf, -np.inf], np.nan))
    s_sc_r = rnk_final.predict(X_sc_r)
    s_sc_aligned = pd.Series(s_sc_r, index=sc_s.index).loc[scoring_df.index].values

    # Construir out con índice original (igual que scoring_df) — NO se resetea todavía
    _extra_cols = [c for c in [TICK_COL, PRICE_COL, "y_t3"] if c in scoring_df.columns]
    out = scoring_df[_extra_cols].copy()
    out["Prob_Clf"]      = p_sc
    out["Score_Ranker"]  = s_sc_aligned
    out["RankerPct"]     = out["Score_Ranker"].rank(pct=True)
    out["RankScore"]     = W_CLF_bt * out["Prob_Clf"] + W_RNK_bt * out["RankerPct"]
    out["Prob_T3_FINAL"] = platt_bt.predict_proba(out[["RankScore"]].values)[:, 1]

    # Retorno real: join por índice ANTES de ordenar (evita desalineación post-reset_index)
    if RET_REAL_COL and RET_REAL_COL in scoring_df.columns:
        out["Ret_Real_NextWeek"] = pd.to_numeric(scoring_df[RET_REAL_COL], errors="coerce")
    else:
        out["Ret_Real_NextWeek"] = np.nan

    # SPY benchmark: escalar — loguear si hay variación intra-fecha inesperada
    spy_mask = scoring_df[TICK_COL] == "SPY"
    if spy_mask.any() and RET_REAL_COL and RET_REAL_COL in scoring_df.columns:
        spy_rets_raw = pd.to_numeric(scoring_df.loc[spy_mask, RET_REAL_COL], errors="coerce")
        if spy_rets_raw.nunique() > 1:
            print(f"  [WARN] SPY_Ret varía dentro de anchor={anchor_date.date()}: "
                  f"{spy_rets_raw.nunique()} valores distintos.")
        spy_ret_val = float(spy_rets_raw.mean())
    else:
        spy_ret_val = np.nan

    out["SPY_Ret_NextWeek"] = spy_ret_val
    out["Beat_SPY"] = np.nan if np.isnan(spy_ret_val) else (
        (out["Ret_Real_NextWeek"] > spy_ret_val)
        .where(out["Ret_Real_NextWeek"].notna())
        .astype(float)
    )

    # ── Attribution metrics sobre universo COMPLETO (antes de cortar Top-K) ──
    attr = _compute_attribution_metrics(out, spy_ret_val, anchor_date)

    # Ahora sí: ordenar y exponer rank
    out = out.sort_values("RankScore", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    return out.head(TOP_K), attr


def run_backtest_loop(
    df_all: pd.DataFrame,
    feature_cols: list,
    universe_mask_col: str,
    run_label: str,
    out_file: Path,
    backtest_dates: Optional[List[str]] = None,
) -> dict:
    """
    Loopea backtest_dates (o BACKTEST_DATES del módulo si no se pasa),
    llama a run_single_backtest por cada fecha, consolida en un Excel con una
    sheet por fecha + Summary + Attribution + Config.

    Tras el primer run imprime estimación de tiempo. Si > BACKTEST_MAX_HOURS_AUTO horas,
    pide confirmación (input con timeout de 30 s; si no hay respuesta, continúa).

    Retorna dict de métricas por fecha.
    """
    import sys, select

    dates = backtest_dates if backtest_dates is not None else BACKTEST_DATES

    print(f"\n{'='*80}")
    print(f"BACKTEST WALK-FORWARD: {run_label}  |  universe: {universe_mask_col}")
    print(f"  Fechas: {len(dates)}  |  Output: {out_file.name}")
    print("=" * 80)

    # Normalizar fechas a Timestamps (mismo mecanismo que _DateKey)
    bt_timestamps = {
        raw: pd.Timestamp(raw).to_period("W").end_time.normalize()
        for raw in dates
    }

    # Verificar cuántas fechas matchean
    available_dates = set(df_all["_DateKey"].dropna().unique())
    missing = [raw for raw, ts in bt_timestamps.items() if ts not in available_dates]
    if len(missing) > 2:
        print(f"\n  *** ALERTA: {len(missing)} fechas del backtest NO matchean en _DateKey:")
        for m in missing:
            print(f"      {m} → normalizado: {bt_timestamps[m].date()}")
        print("  Verificar normalización de fechas antes de continuar.")
    elif missing:
        print(f"  [WARN] {len(missing)} fecha(s) sin match en _DateKey: {[bt_timestamps[m].date() for m in missing]}")

    results: Dict[str, pd.DataFrame] = {}
    summary_rows = []
    attribution_rows: list = []
    first_elapsed = None
    total_to_run = len(dates)

    for i, raw_date in enumerate(dates, 1):
        anchor_ts = bt_timestamps[raw_date]
        t0 = time.time()

        top20, attr = run_single_backtest(
            df_all=df_all,
            anchor_date=anchor_ts,
            feature_cols=feature_cols,
            universe_mask_col=universe_mask_col,
            run_label=run_label,
        )
        elapsed = time.time() - t0

        if top20 is None:
            print(f"[BT {run_label}] {i:02d}/{total_to_run}  anchor={anchor_ts.date()}  → SKIPPED")
            continue

        results[raw_date] = top20
        if attr:
            attribution_rows.append(attr)

        # Métricas compactas para el log
        n_univ = int(df_all[
            (df_all["_DateKey"] == anchor_ts) & df_all[universe_mask_col].fillna(False)
        ].shape[0])
        cutoff = anchor_ts - pd.Timedelta(weeks=1)
        n_train = int(df_all[
            (df_all["_DateKey"] <= cutoff) & df_all[universe_mask_col].fillna(False) & df_all["y_t3"].notna()
        ].shape[0])

        avg_score  = float(top20["RankScore"].mean())
        avg_ret    = float(top20["Ret_Real_NextWeek"].mean()) if top20["Ret_Real_NextWeek"].notna().any() else np.nan
        spy_ret    = float(top20["SPY_Ret_NextWeek"].iloc[0]) if pd.notna(top20["SPY_Ret_NextWeek"].iloc[0]) else np.nan
        beat_n     = int(top20["Beat_SPY"].sum()) if top20["Beat_SPY"].notna().any() else 0
        top1_tick  = top20.iloc[0][TICK_COL]
        top1_score = float(top20.iloc[0]["Prob_T3_FINAL"])
        spr_str    = f"{attr['spearman']:+.3f}" if attr and not _isnan_safe(attr.get("spearman")) else "N/A"

        avg_ret_str = f"{avg_ret:+.1%}" if not np.isnan(avg_ret) else "N/A"
        spy_str     = f"{spy_ret:+.1%}" if not np.isnan(spy_ret) else "N/A"
        beat_str    = f"{beat_n}/{TOP_K}" if top20["Beat_SPY"].notna().any() else "N/A"

        print(f"[BT {run_label}] {i:02d}/{total_to_run}  anchor={anchor_ts.date()}  "
              f"univ={n_univ:,}  train={n_train:,}  fit={elapsed:.1f}s")
        print(f"              top1={top1_tick}({top1_score:.2f})  top20_ret={avg_ret_str}  "
              f"spy={spy_str}  beat_spy={beat_str}  spearman={spr_str}")

        # Estimación de tiempo tras el primer run
        if first_elapsed is None:
            first_elapsed = elapsed
            total_est_h = (first_elapsed * total_to_run) / 3600.0
            print(f"\n  Tiempo del primer backtest single: {first_elapsed:.1f} s")
            print(f"  Estimación total backtest: {total_to_run} corridas × {first_elapsed:.1f} s "
                  f"≈ {total_est_h:.1f} horas")
            if total_est_h > BACKTEST_MAX_HOURS_AUTO:
                print(f"  Estimación > {BACKTEST_MAX_HOURS_AUTO} h. ¿Procedo con el backtest completo? (y/n) "
                      f"[30 s para responder, default=y] ", end="", flush=True)
                try:
                    rdy, _, _ = select.select([sys.stdin], [], [], 30)
                    ans = sys.stdin.readline().strip().lower() if rdy else ""
                except Exception:
                    ans = ""
                if ans == "n":
                    print("\n  Backtest abortado por el usuario.")
                    break
                else:
                    print("\n  Continuando..." if ans else "\n  Sin respuesta en 30 s → continuando...")

        # Acumular para summary
        p20_realized = float(top20["y_t3"].mean()) if "y_t3" in top20.columns else np.nan
        best_ticker = top20.sort_values("Ret_Real_NextWeek", ascending=False).iloc[0][TICK_COL] \
            if top20["Ret_Real_NextWeek"].notna().any() else "N/A"
        worst_ticker = top20.sort_values("Ret_Real_NextWeek", ascending=True).iloc[0][TICK_COL] \
            if top20["Ret_Real_NextWeek"].notna().any() else "N/A"
        hit_rate = float(top20["Beat_SPY"].mean()) if top20["Beat_SPY"].notna().any() else np.nan

        summary_rows.append({
            "Anchor_Date":      raw_date,
            "N_Universe":       n_univ,
            "Top20_AvgScore":   round(avg_score, 4),
            "Top20_AvgRet":     round(avg_ret, 4) if not np.isnan(avg_ret) else np.nan,
            "SPY_Ret":          round(spy_ret, 4) if not np.isnan(spy_ret) else np.nan,
            "HitRate_vs_SPY":   round(hit_rate, 4) if not np.isnan(hit_rate) else np.nan,
            "P@20_realized":    round(p20_realized, 4) if not np.isnan(p20_realized) else np.nan,
            "Best_Ticker":      best_ticker,
            "Worst_Ticker":     worst_ticker,
            "Fit_Time_s":       round(elapsed, 1),
        })

    # ── Summary print ────────────────────────────────────────────
    summary_df    = pd.DataFrame(summary_rows)    if summary_rows    else pd.DataFrame()
    attribution_df = pd.DataFrame(attribution_rows) if attribution_rows else pd.DataFrame()
    processed     = len(results)

    print(f"\n=== BACKTEST SUMMARY: {run_label} ===")
    print(f"Fechas procesadas        : {processed}")

    if not summary_df.empty:
        avg_p20     = summary_df["P@20_realized"].mean()
        avg_top_ret = summary_df["Top20_AvgRet"].mean()
        avg_spy_ret = summary_df["SPY_Ret"].mean()
        avg_hit     = summary_df["HitRate_vs_SPY"].mean()

        print(f"Avg P@20 realizado       : {avg_p20:.3f}" if not _isnan_safe(avg_p20) else "Avg P@20 realizado       : N/A")
        print(f"Avg Top20 next-week ret  : {avg_top_ret:+.2%}" if not _isnan_safe(avg_top_ret) else "Avg Top20 next-week ret  : N/A")
        print(f"Avg SPY next-week ret    : {avg_spy_ret:+.2%}" if not _isnan_safe(avg_spy_ret) else "Avg SPY next-week ret    : N/A")
        print(f"Avg HitRate vs SPY       : {avg_hit:.0%}" if not _isnan_safe(avg_hit) else "Avg HitRate vs SPY       : N/A")

        ret_col_s = summary_df["Top20_AvgRet"].dropna()
        if len(ret_col_s):
            best_idx  = ret_col_s.idxmax()
            worst_idx = ret_col_s.idxmin()
            bd = summary_df.loc[best_idx,  "Anchor_Date"]
            br = summary_df.loc[best_idx,  "Top20_AvgRet"]
            bs = summary_df.loc[best_idx,  "SPY_Ret"]
            wd = summary_df.loc[worst_idx, "Anchor_Date"]
            wr = summary_df.loc[worst_idx, "Top20_AvgRet"]
            ws = summary_df.loc[worst_idx, "SPY_Ret"]
            print(f"Mejor semana             : {bd} (top20={br:+.1%} vs spy={bs:+.1%})")
            print(f"Peor semana              : {wd} (top20={wr:+.1%} vs spy={ws:+.1%})")

    # ── Attribution aggregations ─────────────────────────────────
    def _col_mean(df, col):
        if df.empty or col not in df.columns:
            return np.nan
        return float(df[col].mean(skipna=True))

    attr_agg = {}
    if not attribution_df.empty:
        spr = attribution_df["spearman"].dropna()
        attr_agg = {
            "spearman_mean":      _col_mean(attribution_df, "spearman"),
            "spearman_median":    float(spr.median()) if len(spr) else np.nan,
            "spearman_pct_pos":   float((spr > 0).mean()) if len(spr) else np.nan,
            "decile_spread_mean": _col_mean(attribution_df, "decile_spread"),
            "top3_mean":          _col_mean(attribution_df, "top3_ret"),
            "top20_mean":         _col_mean(attribution_df, "top20_ret"),
        }
        t3   = attribution_df["top3_ret"].dropna()
        t20  = attribution_df["top20_ret"].dropna()
        common_idx = t3.index.intersection(t20.index)
        attr_agg["top3_minus_top20"] = float((t3.loc[common_idx] - t20.loc[common_idx]).mean()) \
            if len(common_idx) else np.nan

        def _pct(v): return f"{v:.1%}" if not _isnan_safe(v) else "N/A"
        def _flt(v): return f"{v:+.4f}" if not _isnan_safe(v) else "N/A"

        print(f"\n  === Attribution (universo completo) ===")
        print(f"  spearman_mean       : {_flt(attr_agg['spearman_mean'])}")
        print(f"  spearman_median     : {_flt(attr_agg['spearman_median'])}")
        print(f"  spearman_pct_pos    : {_pct(attr_agg['spearman_pct_pos'])}")
        print(f"  decile_spread_mean  : {_pct(attr_agg['decile_spread_mean'])}")
        print(f"  top3_mean           : {_pct(attr_agg['top3_mean'])}")
        print(f"  top20_mean          : {_pct(attr_agg['top20_mean'])}")
        print(f"  top3_minus_top20    : {_pct(attr_agg['top3_minus_top20'])}  ← KPI concentración")

    print(f"Excel: {out_file}")

    # ── Export Excel ──────────────────────────────────────────────
    if results:
        config_df = pd.DataFrame({
            "Parametro": [
                "run_label", "universe_mask_col", "BACKTEST_DATES",
                "TOP_K", "N_FOLDS", "EMBARGO_DATES", "HOLDOUT_N_WEEKS",
                "SEED", "INPUT_HASH", "DATE_STAMP",
                "UNIV_MIN_CLOSE", "UNIV_MIN_DOLLAR_VOL_20D", "UNIV_MAX_ZERO_RET_PCT_60D",
            ],
            "Valor": [
                run_label, universe_mask_col, ", ".join(dates),
                TOP_K, N_FOLDS, EMBARGO_DATES, HOLDOUT_N_WEEKS,
                SEED, INPUT_HASH, DATE_STAMP,
                UNIV_MIN_CLOSE, UNIV_MIN_DOLLAR_VOL_20D, UNIV_MAX_ZERO_RET_PCT_60D,
            ],
        })

        # Agregar agregados de Attribution al final del Summary
        if attr_agg and not summary_df.empty:
            agg_rows = pd.DataFrame([{
                "Anchor_Date": "=== ATTRIBUTION AGGREGATES ===",
                **{k: v for k, v in attr_agg.items()},
            }])
            summary_export = pd.concat([summary_df, agg_rows], ignore_index=True)
        else:
            summary_export = summary_df

        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            for raw_date, top20 in results.items():
                sheet_name = raw_date[:10]
                top20.to_excel(writer, sheet_name=sheet_name, index=False)
            if not summary_export.empty:
                summary_export.to_excel(writer, sheet_name="Summary", index=False)
            if not attribution_df.empty:
                attribution_df.to_excel(writer, sheet_name="Attribution", index=False)
            config_df.to_excel(writer, sheet_name="Config", index=False)
        print(f"[BT {run_label}] Exportado: {out_file}")
    else:
        print(f"[BT {run_label}] Sin resultados: Excel no generado.")

    return {r["Anchor_Date"]: r for r in summary_rows}
