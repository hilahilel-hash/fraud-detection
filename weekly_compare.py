# -*- coding: utf-8 -*-
"""
Local comparison runner: baseline (v11.1) vs fixed (v12) on the cached CSV.
Does NOT touch weekly.py. Reads /tmp/fraud_data/weekly_data.csv to avoid BQ.

Fixes applied in FIXED mode:
  F1. manual_risk_score: fit min/max on TRAIN, apply to val/test
  F2. normalized_iforest: fit mean/std on TRAIN, apply to val/test
  F3. Exclude manual_risk_score from XGBoost features (avoid double-counting in blend)
  F4. scale_pos_weight = sqrt(neg/pos) instead of neg/pos (better PR-AUC calibration)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import IsolationForest

DATA_PATH = "/tmp/fraud_data/weekly_data.csv"

# =================== CONFIG ===================
LABEL_COL = "has_chargeback"
DATE_COL = "date_hour"
VAL_DAYS, TEST_DAYS = 30, 90
IGNORE_RECENT_DAYS = 40
SEED = 42

CAT_COLS = ["country_code", "http_request_country", "clean_os", "payer_provider_payer_country"]

HIGH_RISK_COUNTRIES = [
    "Tunisia", "Bangladesh", "Indonesia", "Jamaica", "Nigeria", "Pakistan",
    "Algeria", "Morocco", "India", "Saudi Arabia", "Oman", "Jordan", "Mexico", "Italy",
]

ID_COLS_TO_EXCLUDE = [
    "token", "user_id", "seller_id", "gig_id", "order_id",
    "email", "ip", "payer_provider_payer_id",
]


# =================== HELPERS (identical to weekly.py) ===================
def ensure_datetime(df):
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)
    df[DATE_COL] = df[DATE_COL].fillna(df[DATE_COL].dropna().min())
    return df


def safe_fillna(df):
    NULL_STRINGS = {"", "nan", "NaN", "None", "NULL", "<NA>", "<null>", "N/A", "n/a"}
    for c in df.columns:
        if c == DATE_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("boolean")
        else:
            s = df[c].astype(str).str.strip()
            df[c] = s.where(~s.str.lower().isin({x.lower() for x in NULL_STRINGS}), np.nan)
    return df


def time_split(df):
    df = df.sort_values(DATE_COL)
    max_ts = df[DATE_COL].max()
    cut_test = max_ts - pd.Timedelta(days=TEST_DAYS)
    cut_val = cut_test - pd.Timedelta(days=VAL_DAYS)
    m_train = df[DATE_COL] < cut_val
    m_val = (df[DATE_COL] >= cut_val) & (df[DATE_COL] < cut_test)
    m_test = df[DATE_COL] >= cut_test
    return m_train, m_val, m_test


def calculate_adaptive_thresholds(df_train):
    t = {}
    t["unique_ips_last_24h"] = df_train["unique_ips_last_24h"].quantile(0.90)
    t["user_txns_1h"] = df_train["user_txns_1h"].quantile(0.95)
    t["user_txns_24h"] = df_train["user_txns_24h"].quantile(0.90)
    t["buyer_count_clone"] = df_train["buyer_count_clone"].quantile(0.90)
    t["user_txns_30d_high"] = df_train["user_txns_30d"].quantile(0.80)
    t["payment_amount_high"] = df_train["payment_amount"].quantile(0.95)
    t["total_orders_buyer_low"] = (
        df_train["total_orders_buyer"].quantile(0.15)
        if "total_orders_buyer" in df_train.columns else 3
    )
    t["total_orders_seller_low"] = (
        df_train["total_orders_seller"].quantile(0.15)
        if "total_orders_seller" in df_train.columns else 3
    )
    if "total_payment_buyer_seller_24h" in df_train.columns:
        mask = (df_train["total_orders_buyer_seller"] >= 3) & (df_train["user_txns_24h"] >= 3)
        filtered = df_train.loc[mask, "total_payment_buyer_seller_24h"]
        t["total_payment_buyer_seller_24h"] = filtered.quantile(0.95) if not filtered.empty else 300
    else:
        t["total_payment_buyer_seller_24h"] = 300

    defaults = {
        "unique_ips_last_24h": 2, "user_txns_1h": 1, "user_txns_24h": 2,
        "buyer_count_clone": 5, "user_txns_30d_high": 5, "payment_amount_high": 500,
        "secs_since_prev": 60, "seller_txn_ratio_high": 5,
    }
    for k, v in defaults.items():
        t.setdefault(k, v)
    return t


def build_rule_columns(df, t):
    df = df.copy()

    def num(c):
        return pd.to_numeric(df.get(c, np.nan), errors="coerce")

    df["rule_user_has_multiple_payers"] = (num("user_has_multiple_payers") == 1)

    if "unique_users_per_payer" in df.columns:
        n = num("unique_users_per_payer")
        df["rule_unique_users_per_payer_score"] = np.select(
            [n <= 1, (n > 1) & (n <= 4), (n > 4) & (n <= 10), n > 10],
            [0.0, 0.4, 0.7, 1.0],
        )
    else:
        df["rule_unique_users_per_payer_score"] = 0.0

    df["rule_high_risk_country_seller"] = (
        df["valid_seller_country"].astype(str).isin(HIGH_RISK_COUNTRIES)
        if "valid_seller_country" in df.columns else False
    )
    df["rule_is_team"] = (df["is_team"] == 1) & (df["total_orders_buyer"] <= 10)
    df["rule_is_new_user_7d"] = (num("is_new_user_7d") == 1)

    if "buyer_country" in df.columns:
        df["rule_new_user_from_high_risk_country"] = (
            df["buyer_country"].astype(str).isin(HIGH_RISK_COUNTRIES) &
            (df["rule_is_new_user_7d"] == True)
        ).astype(int)
    else:
        df["rule_new_user_from_high_risk_country"] = 0

    df["rule_seller_fraud_14d"] = (num("seller_fraud_14d") > 0)
    df["rule_seller_fraud_30d"] = (num("seller_fraud_30d") > 0)
    df["rule_is_paypal_after_decline"] = (num("is_paypal_after_decline") == 1)
    df["rule_unique_ips_last_24h"] = (num("unique_ips_last_24h") > t["unique_ips_last_24h"])
    df["rule_user_txns_1h"] = (num("user_txns_1h") > t["user_txns_1h"])
    df["rule_user_txns_24h"] = (num("user_txns_24h") > t["user_txns_24h"])
    df["rule_buyer_count_clone"] = (num("buyer_count_clone") > t["buyer_count_clone"])
    df["rule_is_fake_location"] = (df["is_fake_location"] == True) if "is_fake_location" in df.columns else False

    df["rule_high_volume"] = (
        (num("payment_amount") > t["payment_amount_high"]) &
        (num("user_txns_30d") > t["user_txns_30d_high"])
    ).astype(int)
    df["rule_high_volume_score"] = np.where(df["rule_high_volume"], 0.99, 0).astype(float)

    if "days_since_signup" in df.columns:
        df["rule_new_user_60d"] = (num("days_since_signup") < 60)

    df["combo_high_volume"] = df["rule_high_volume"]
    df["combo_new_user_high_volume"] = (
        (df["rule_is_new_user_7d"] == True) &
        (num("user_txns_30d") > t["user_txns_30d_high"])
    ).astype(int)
    df["combo_repeat_pair_high_value"] = (
        (num("total_orders_buyer_seller") >= 3) &
        (num("user_txns_24h") >= 3) &
        (num("total_payment_buyer_seller_24h") > t["total_payment_buyer_seller_24h"])
    ).astype(int)
    df["combo_low_buyer_seller"] = (
        (num("total_orders_buyer") <= t["total_orders_buyer_low"]) &
        (num("total_orders_seller") <= t["total_orders_seller_low"])
    ).astype(int)

    return df


def _weight_from_lift(lift, corr):
    return float(np.clip((lift ** 1.5) * (1 + 3 * corr), 0.5, 25))


def compute_rule_weights(df_train):
    """Original v11.1 weights: lift computed on the full train (no CV, no smoothing)."""
    rule_cols = [c for c in df_train.columns if c.startswith("rule_")]
    global_rate = max(df_train[LABEL_COL].mean(), 1e-9)
    weights = {}
    for c in rule_cols:
        col_values = df_train[c]
        valid_mask = col_values.notna()
        col_non_na = col_values[valid_mask]
        if col_non_na.dropna().isin([0, 1, True, False]).all():
            mask = col_non_na.astype(bool)
        else:
            mask = col_non_na > 0
        if mask.sum() == 0:
            lift, corr = 1.0, 0
        else:
            lift = df_train.loc[valid_mask].loc[mask, LABEL_COL].mean() / global_rate
            corr = abs(np.corrcoef(col_non_na.astype(float), df_train.loc[valid_mask, LABEL_COL])[0, 1])
            if np.isnan(corr):
                corr = 0
        weights[c] = _weight_from_lift(lift, corr)
    return weights


def compute_rule_weights_oof(df_train, n_splits=5, prior_strength=50.0):
    """K-fold OOF weights with Bayesian smoothing on lift.

    Time-ordered folds (no shuffle). For each fold, fit on the other K-1 folds.
    Bayesian smoothing pulls the lift toward 1.0 when a rule has few firings:
        rate_smoothed = (rate_when_fires * n_fires + global_rate * prior_strength)
                        / (n_fires + prior_strength)
    Final weight = mean across folds.
    """
    rule_cols = [c for c in df_train.columns if c.startswith("rule_")]
    df_sorted = df_train.sort_values(DATE_COL).reset_index(drop=True)
    fold_idx = np.array_split(np.arange(len(df_sorted)), n_splits)

    per_fold = {c: [] for c in rule_cols}
    for k in range(n_splits):
        train_rows = np.concatenate([fold_idx[i] for i in range(n_splits) if i != k])
        fold_train = df_sorted.iloc[train_rows]
        fold_rate = max(fold_train[LABEL_COL].mean(), 1e-9)

        for c in rule_cols:
            col_values = fold_train[c]
            valid_mask = col_values.notna()
            col_non_na = col_values[valid_mask]
            if col_non_na.empty:
                per_fold[c].append(_weight_from_lift(1.0, 0))
                continue
            if col_non_na.dropna().isin([0, 1, True, False]).all():
                mask = col_non_na.astype(bool)
            else:
                mask = col_non_na > 0
            n_fires = int(mask.sum())
            if n_fires == 0:
                per_fold[c].append(_weight_from_lift(1.0, 0))
                continue
            rate_when_fires = fold_train.loc[valid_mask].loc[mask, LABEL_COL].mean()
            rate_smoothed = (
                (rate_when_fires * n_fires + fold_rate * prior_strength) /
                (n_fires + prior_strength)
            )
            lift = rate_smoothed / fold_rate
            try:
                corr = abs(np.corrcoef(
                    col_non_na.astype(float),
                    fold_train.loc[valid_mask, LABEL_COL],
                )[0, 1])
                if np.isnan(corr):
                    corr = 0
            except Exception:
                corr = 0
            per_fold[c].append(_weight_from_lift(lift, corr))

    weights = {c: float(np.mean(vals)) for c, vals in per_fold.items()}
    return weights


# =================== MANUAL RISK (two variants) ===================
def compute_manual_risk_baseline(df, weights):
    """v11.1 behavior: min/max per split (LEAKY)."""
    df = df.copy()
    df["manual_risk_raw"] = 0.0
    for c, w in weights.items():
        if c not in df.columns:
            continue
        col = df[c]
        if col.isna().all():
            continue
        valid_mask = col.notna()
        valid_vals = col[valid_mask]
        if pd.api.types.is_bool_dtype(valid_vals):
            contrib = valid_vals.astype(bool).astype(int) * w
        elif valid_vals.dropna().isin([0, 1, True, False]).all():
            contrib = valid_vals.astype(bool).astype(int) * w
        elif valid_vals.dropna().between(0, 1).all():
            contrib = valid_vals.astype(float) * w
        else:
            contrib = valid_vals.astype(float).clip(0, 1) * w
        df.loc[valid_mask, "manual_risk_raw"] += contrib
    mn, mx = df["manual_risk_raw"].min(), df["manual_risk_raw"].max()
    df["manual_risk_score"] = (df["manual_risk_raw"] - mn) / (mx - mn + 1e-9)
    df.drop(columns=["manual_risk_raw"], inplace=True)
    return df


def compute_manual_risk_raw(df, weights):
    """Compute the raw score without normalizing. Used by fixed variant."""
    df = df.copy()
    df["manual_risk_raw"] = 0.0
    for c, w in weights.items():
        if c not in df.columns:
            continue
        col = df[c]
        if col.isna().all():
            continue
        valid_mask = col.notna()
        valid_vals = col[valid_mask]
        if pd.api.types.is_bool_dtype(valid_vals):
            contrib = valid_vals.astype(bool).astype(int) * w
        elif valid_vals.dropna().isin([0, 1, True, False]).all():
            contrib = valid_vals.astype(bool).astype(int) * w
        elif valid_vals.dropna().between(0, 1).all():
            contrib = valid_vals.astype(float) * w
        else:
            contrib = valid_vals.astype(float).clip(0, 1) * w
        df.loc[valid_mask, "manual_risk_raw"] += contrib
    return df


def apply_manual_risk_normalization(df, mn, mx):
    df = df.copy()
    df["manual_risk_score"] = (df["manual_risk_raw"] - mn) / (mx - mn + 1e-9)
    df.drop(columns=["manual_risk_raw"], inplace=True)
    return df


# =================== DERIVED FEATURES (two variants) ===================
def add_derived_features_baseline(df):
    """v11.1: normalize iforest using THIS split's mean/std (LEAKY)."""
    df = df.copy()
    def num(c):
        return pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df["txn_rate_signup"] = num("user_txns_30d") / (num("days_since_signup") + 1)
    df["amt_ratio_to_mean"] = np.where(
        pd.notna(num("user_amt_mean_30d")) & (num("user_amt_mean_30d") > 0),
        num("payment_amount") / num("user_amt_mean_30d"),
        np.nan,
    )
    df["normalized_iforest"] = (
        (num("iforest_score") - num("iforest_score").mean()) /
        (num("iforest_score").std() + 1e-9)
    )
    return df


def add_derived_features_fixed(df, iforest_mean=None, iforest_std=None):
    df = df.copy()
    def num(c):
        return pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df["txn_rate_signup"] = num("user_txns_30d") / (num("days_since_signup") + 1)
    df["amt_ratio_to_mean"] = np.where(
        pd.notna(num("user_amt_mean_30d")) & (num("user_amt_mean_30d") > 0),
        num("payment_amount") / num("user_amt_mean_30d"),
        np.nan,
    )
    s = num("iforest_score")
    df["normalized_iforest"] = (s - iforest_mean) / (iforest_std + 1e-9)
    return df


# =================== PIPELINE ===================
def prepare_data(df_in, mode, extra_skip_days=0, verbose=True):
    """Everything up to (but not including) XGBoost training.
    Returns dict with Xtr, ytr, Xva, yva, Xte, yte, manual_risk_score arrays, scale_pos_weight."""
    if verbose:
        print(f"\n{'=' * 80}\n  MODE: {mode} (extra_skip_days={extra_skip_days})\n{'=' * 80}")
    df = df_in.copy()
    df = ensure_datetime(df)
    df = safe_fillna(df)

    max_ts = df[DATE_COL].max()
    cutoff = max_ts - pd.Timedelta(days=IGNORE_RECENT_DAYS + extra_skip_days)
    df = df[df[DATE_COL] < cutoff].copy()

    if "seller_pro" in df.columns:
        df["seller_pro"] = df["seller_pro"].fillna(False).astype(bool)

    m_train, m_val, m_test = time_split(df)
    df_train = df[m_train].copy()
    df_val = df[m_val].copy()
    df_test = df[m_test].copy()
    if verbose:
        print(f"Train: {df_train.shape}  Val: {df_val.shape}  Test: {df_test.shape}")
        print(f"Fraud rate (train): {df_train[LABEL_COL].mean():.4f}")

    thresholds = calculate_adaptive_thresholds(df_train)
    df_train = build_rule_columns(df_train, thresholds)
    df_val = build_rule_columns(df_val, thresholds)
    df_test = build_rule_columns(df_test, thresholds)

    encoders = {}
    for col in CAT_COLS:
        if col in df_train.columns:
            le = LabelEncoder()
            train_vals = df_train[col].astype(str).fillna("UNKNOWN").unique().tolist()
            if "UNKNOWN" not in train_vals:
                train_vals.append("UNKNOWN")
            le.fit(train_vals)
            encoders[col] = le
            known = set(le.classes_)
            for part in (df_train, df_val, df_test):
                part[col] = (
                    part[col].astype(str).fillna("UNKNOWN")
                    .apply(lambda x: x if x in known else "UNKNOWN")
                )
                part[col] = le.transform(part[col])

    num_cols = [
        c for c in df_train.select_dtypes(include=[np.number]).columns
        if c not in [LABEL_COL] + ID_COLS_TO_EXCLUDE
    ]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED, n_jobs=-1)
    iso.fit(df_train[num_cols])
    for part in (df_train, df_val, df_test):
        part["iforest_score"] = -iso.score_samples(part[num_cols])

    if mode == "baseline":
        df_train = add_derived_features_baseline(df_train)
        df_val = add_derived_features_baseline(df_val)
        df_test = add_derived_features_baseline(df_test)
        ifm = None
        ifs = None
    else:
        ifm = float(df_train["iforest_score"].mean())
        ifs = float(df_train["iforest_score"].std())
        df_train = add_derived_features_fixed(df_train, ifm, ifs)
        df_val = add_derived_features_fixed(df_val, ifm, ifs)
        df_test = add_derived_features_fixed(df_test, ifm, ifs)

    if mode == "fixed_v2":
        weights = compute_rule_weights_oof(df_train, n_splits=5, prior_strength=50.0)
    else:
        weights = compute_rule_weights(df_train)

    if mode == "baseline":
        df_train = compute_manual_risk_baseline(df_train, weights)
        df_val = compute_manual_risk_baseline(df_val, weights)
        df_test = compute_manual_risk_baseline(df_test, weights)
        manual_risk_mn = None
        manual_risk_mx = None
    else:
        df_train = compute_manual_risk_raw(df_train, weights)
        df_val = compute_manual_risk_raw(df_val, weights)
        df_test = compute_manual_risk_raw(df_test, weights)
        manual_risk_mn = float(df_train["manual_risk_raw"].min())
        manual_risk_mx = float(df_train["manual_risk_raw"].max())
        df_train = apply_manual_risk_normalization(df_train, manual_risk_mn, manual_risk_mx)
        df_val = apply_manual_risk_normalization(df_val, manual_risk_mn, manual_risk_mx)
        df_test = apply_manual_risk_normalization(df_test, manual_risk_mn, manual_risk_mx)

    exclude_cols = [LABEL_COL, DATE_COL] + ID_COLS_TO_EXCLUDE
    if mode in ("fixed", "fixed_v2", "fixed_v3"):
        exclude_cols = exclude_cols + ["manual_risk_score"]
    if mode == "fixed_v3":
        rule_and_combo = [c for c in df_train.columns if c.startswith("rule_") or c.startswith("combo_")]
        exclude_cols = exclude_cols + rule_and_combo
    feats = [c for c in df_train.columns if c not in exclude_cols]

    Xtr = df_train[feats].select_dtypes(include=[np.number])
    ytr = df_train[LABEL_COL].astype(int)
    Xva = df_val[feats].select_dtypes(include=[np.number])
    yva = df_val[LABEL_COL].astype(int)
    Xte = df_test[feats].select_dtypes(include=[np.number])
    yte = df_test[LABEL_COL].astype(int)

    Xva = Xva.reindex(columns=Xtr.columns)
    Xte = Xte.reindex(columns=Xtr.columns)

    if verbose:
        print(f"#features used: {Xtr.shape[1]}")

    if mode == "baseline":
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    else:
        spw = float(np.sqrt((len(ytr) - ytr.sum()) / max(ytr.sum(), 1)))

    return {
        "mode": mode,
        "Xtr": Xtr, "ytr": ytr,
        "Xva": Xva, "yva": yva,
        "Xte": Xte, "yte": yte,
        "manual_risk_val": df_val["manual_risk_score"].to_numpy(),
        "manual_risk_test": df_test["manual_risk_score"].to_numpy(),
        "scale_pos_weight": float(spw),
        "n_features": int(Xtr.shape[1]),
        # artifacts for daily scoring
        "encoders": encoders,
        "iso": iso,
        "iso_num_cols": list(num_cols),
        "thresholds": thresholds,
        "weights": weights,
        "manual_risk_mn": manual_risk_mn,
        "manual_risk_mx": manual_risk_mx,
        "iforest_mean": ifm,
        "iforest_std": ifs,
        "feature_columns": list(Xtr.columns),
    }


def train_and_eval(prep, xgb_params=None, num_boost_round=1500, early_stopping_rounds=50, verbose=False):
    """Train XGBoost and compute blend + return metrics dict."""
    dtrain = xgb.DMatrix(prep["Xtr"], label=prep["ytr"], missing=np.nan)
    dval = xgb.DMatrix(prep["Xva"], label=prep["yva"], missing=np.nan)
    dtest = xgb.DMatrix(prep["Xte"], label=prep["yte"], missing=np.nan)

    default_params = {
        "max_depth": 7, "learning_rate": 0.05, "subsample": 0.7,
        "colsample_bytree": 0.8, "objective": "binary:logistic",
        "eval_metric": ["aucpr"], "tree_method": "hist",
        "scale_pos_weight": prep["scale_pos_weight"], "seed": SEED,
    }
    if xgb_params:
        default_params.update(xgb_params)

    model = xgb.train(
        default_params, dtrain, num_boost_round=num_boost_round,
        evals=[(dval, "val")], early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )

    preds_model_val = model.predict(dval)
    preds_rules_val = prep["manual_risk_val"]
    best_alpha, best_metric = 0.5, -1
    for a in np.arange(0.1, 0.95, 0.05):
        blend = a * preds_model_val + (1 - a) * preds_rules_val
        score = average_precision_score(prep["yva"], blend)
        if score > best_metric:
            best_alpha, best_metric = a, score

    preds_test_model = model.predict(dtest)
    preds_test_blend = (
        best_alpha * preds_test_model +
        (1 - best_alpha) * prep["manual_risk_test"]
    )

    yte = prep["yte"]
    metrics = {
        "mode": prep["mode"],
        "n_features": prep["n_features"],
        "scale_pos_weight": prep["scale_pos_weight"],
        "best_alpha": float(best_alpha),
        "val_auprc_at_alpha": float(best_metric),
        "model": model,
        "test_roc_auc_model_only": float(roc_auc_score(yte, preds_test_model)),
        "test_auprc_model_only": float(average_precision_score(yte, preds_test_model)),
        "test_roc_auc_rules_only": float(roc_auc_score(yte, prep["manual_risk_test"])),
        "test_auprc_rules_only": float(average_precision_score(yte, prep["manual_risk_test"])),
        "test_roc_auc_blend": float(roc_auc_score(yte, preds_test_blend)),
        "test_auprc_blend": float(average_precision_score(yte, preds_test_blend)),
        "best_iteration": int(model.best_iteration),
    }
    return metrics


def run_pipeline(df_in, mode, extra_skip_days=0, xgb_params=None, verbose=True):
    """Wrapper that does prepare + train. Keeps backward compatibility."""
    prep = prepare_data(df_in, mode, extra_skip_days=extra_skip_days, verbose=verbose)
    return train_and_eval(prep, xgb_params=xgb_params)


def run_grid_search(df, mode="fixed_v3"):
    """Grid search XGBoost hyperparams on val AUPRC, then report test metrics."""
    prep = prepare_data(df, mode, extra_skip_days=0, verbose=True)

    grid = []
    for max_depth in [3, 4, 5, 6, 7]:
        for min_child_weight in [1, 5, 20]:
            for reg_lambda in [1, 5]:
                grid.append({
                    "max_depth": max_depth,
                    "min_child_weight": min_child_weight,
                    "reg_lambda": reg_lambda,
                })

    print(f"\n[grid] Running {len(grid)} combinations on mode={mode}")
    rows = []
    for params in grid:
        m = train_and_eval(prep, xgb_params=params)
        rows.append({**params, **m})
        print(f"  depth={params['max_depth']}  mcw={params['min_child_weight']:>2}  "
              f"lam={params['reg_lambda']}  "
              f"val_auprc={m['val_auprc_at_alpha']:.4f}  "
              f"test_auprc_blend={m['test_auprc_blend']:.4f}  "
              f"best_iter={m['best_iteration']}")

    rows_sorted = sorted(rows, key=lambda r: r["val_auprc_at_alpha"], reverse=True)
    print("\n[grid] Top 5 by VAL AUPRC:")
    for r in rows_sorted[:5]:
        print(f"  depth={r['max_depth']}  mcw={r['min_child_weight']:>2}  "
              f"lam={r['reg_lambda']}  val_auprc={r['val_auprc_at_alpha']:.4f}  "
              f"test_auprc_blend={r['test_auprc_blend']:.4f}  best_iter={r['best_iteration']}")

    best = rows_sorted[0]
    print(f"\n[grid] BEST by val_auprc: depth={best['max_depth']} mcw={best['min_child_weight']} "
          f"lam={best['reg_lambda']}  test_auprc_blend={best['test_auprc_blend']:.4f}")
    return best


TUNED_PARAMS = {"max_depth": 4, "min_child_weight": 20, "reg_lambda": 5}


def train_for_daily(df_weekly, mode="fixed_v3", xgb_params=None):
    """Run weekly training and return all artifacts a daily scorer needs.
    Default = fixed_v3 + tuned params (best config from our sweep)."""
    if xgb_params is None and mode != "baseline":
        xgb_params = TUNED_PARAMS
    prep = prepare_data(df_weekly, mode, extra_skip_days=0, verbose=False)
    metrics = train_and_eval(prep, xgb_params=xgb_params)
    artifacts = {
        "mode": mode,
        "model": metrics["model"],
        "feature_columns": prep["feature_columns"],
        "encoders": prep["encoders"],
        "iso": prep["iso"],
        "iso_num_cols": prep["iso_num_cols"],
        "thresholds": prep["thresholds"],
        "weights": prep["weights"],
        "manual_risk_mn": prep["manual_risk_mn"],
        "manual_risk_mx": prep["manual_risk_mx"],
        "iforest_mean": prep["iforest_mean"],
        "iforest_std": prep["iforest_std"],
        "base_alpha": metrics["best_alpha"],
        "test_auprc_blend": metrics["test_auprc_blend"],
    }
    return artifacts


def run_cv(df):
    """Rolling time-series CV: 3 windows.
    Compares baseline, fixed_v3 (default xgb), fixed_v3 (tuned xgb)."""
    windows = [0, 60, 120]
    results = {"baseline": [], "fixed_v3": [], "fixed_v3_tuned": []}
    for skip in windows:
        prep_b = prepare_data(df, "baseline", extra_skip_days=skip, verbose=True)
        prep_v3 = prepare_data(df, "fixed_v3", extra_skip_days=skip, verbose=True)

        m_b = train_and_eval(prep_b)
        m_v3 = train_and_eval(prep_v3)
        m_v3t = train_and_eval(prep_v3, xgb_params=TUNED_PARAMS)
        for d, name in [(m_b, "baseline"), (m_v3, "fixed_v3"), (m_v3t, "fixed_v3_tuned")]:
            d["extra_skip_days"] = skip
            results[name].append(d)

    print("\n" + "=" * 80)
    print("  ROLLING TIME-SERIES CV RESULTS")
    print("=" * 80)

    metric_keys = ["test_auprc_blend", "test_roc_auc_blend",
                   "test_auprc_model_only", "test_roc_auc_model_only"]

    for key in metric_keys:
        print(f"\n--- {key} ---")
        print(f"{'fold':>6}  {'baseline':>10}  {'fixed_v3':>10}  {'v3_tuned':>10}  "
              f"{'Δ v3':>9}  {'Δ tuned':>9}")
        for i in range(len(windows)):
            b = results["baseline"][i][key]
            f = results["fixed_v3"][i][key]
            t = results["fixed_v3_tuned"][i][key]
            print(f"{i+1:>6}  {b:>10.4f}  {f:>10.4f}  {t:>10.4f}  "
                  f"{f-b:>+9.4f}  {t-b:>+9.4f}")

        b_vals = [results["baseline"][i][key] for i in range(len(windows))]
        f_vals = [results["fixed_v3"][i][key] for i in range(len(windows))]
        t_vals = [results["fixed_v3_tuned"][i][key] for i in range(len(windows))]
        print(f"{'mean':>6}  {np.mean(b_vals):>10.4f}  {np.mean(f_vals):>10.4f}  "
              f"{np.mean(t_vals):>10.4f}  "
              f"{np.mean(f_vals)-np.mean(b_vals):>+9.4f}  "
              f"{np.mean(t_vals)-np.mean(b_vals):>+9.4f}")
        print(f"{'std':>6}  {np.std(b_vals):>10.4f}  {np.std(f_vals):>10.4f}  "
              f"{np.std(t_vals):>10.4f}")


def main():
    print(f"[load] reading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"[load] shape: {df.shape}")

    if os.environ.get("CV_MODE") == "1":
        run_cv(df)
        return
    if os.environ.get("GRID_MODE") == "1":
        run_grid_search(df, mode="fixed_v3")
        return

    baseline = run_pipeline(df, "baseline")
    fixed = run_pipeline(df, "fixed")
    fixed_v2 = run_pipeline(df, "fixed_v2")
    fixed_v3 = run_pipeline(df, "fixed_v3")

    print("\n" + "=" * 80)
    print("  COMPARISON")
    print("=" * 80)
    keys = [
        "n_features", "scale_pos_weight", "best_alpha", "best_iteration",
        "val_auprc_at_alpha",
        "test_roc_auc_model_only", "test_auprc_model_only",
        "test_roc_auc_rules_only", "test_auprc_rules_only",
        "test_roc_auc_blend", "test_auprc_blend",
    ]

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    width = max(len(k) for k in keys)
    header = (
        f"{'metric'.ljust(width)}  "
        f"{'baseline':>10}  {'fixed':>10}  {'fixed_v2':>10}  {'fixed_v3':>10}  "
        f"{'Δfix':>8}  {'Δv2':>8}  {'Δv3':>8}"
    )
    print(header)
    print("-" * len(header))
    for k in keys:
        b, f1, f2, f3 = baseline[k], fixed[k], fixed_v2[k], fixed_v3[k]
        if isinstance(b, float):
            d1, d2, d3 = f1 - b, f2 - b, f3 - b
            print(
                f"{k.ljust(width)}  {fmt(b):>10}  {fmt(f1):>10}  {fmt(f2):>10}  {fmt(f3):>10}  "
                f"{d1:+.4f}  {d2:+.4f}  {d3:+.4f}"
            )
        else:
            print(f"{k.ljust(width)}  {fmt(b):>10}  {fmt(f1):>10}  {fmt(f2):>10}  {fmt(f3):>10}")


if __name__ == "__main__":
    main()
