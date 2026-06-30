# -*- coding: utf-8 -*-
"""
Fiverr Fraud Detection - Weekly Training Pipeline (v12.0)
Runs every Sunday at 08:00 Israel time via GitHub Actions.
Saves trained artifacts to /tmp/fraud_artifacts/ (uploaded via GitHub Actions).

v12 changes vs v11.1 (validated on 3-fold time-series CV, +28% mean AUPRC blend):
  F1. manual_risk_score normalized with TRAIN min/max applied to val/test (was per-split).
  F2. normalized_iforest uses TRAIN mean/std applied to val/test (was per-split).
  F3. manual_risk_score excluded from XGBoost features (avoid double-counting in blend).
  F4. scale_pos_weight = sqrt(neg/pos) instead of full neg/pos (better PR-AUC calibration).
  F6. rule_* and combo_* excluded from XGBoost features (already in manual_risk_score blend).
  F7. Tuned XGBoost: max_depth=4, min_child_weight=20, reg_lambda=5 (from grid search).
  Plus: save normalization_stats artifact (manual_risk_mn/mx, iforest_mean/std) for daily.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt

import xgboost as xgb
import joblib
import re

from google.cloud import bigquery

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import BorderlineSMOTE

from drive_utils import get_credentials

# =================== AUTH ===================
SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive",
]

credentials = get_credentials()
client = bigquery.Client(project="fiverr-bq-payments-adhoc-prod", credentials=credentials)
print("BigQuery client initialized.")

LOCAL_ARTIFACT_DIR = "/tmp/fraud_artifacts"
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

# =================== CONFIG ===================
LABEL_COL = "has_chargeback"
DATE_COL = "date_hour"
VAL_DAYS, TEST_DAYS = 30, 90
SMOTE_ON = False

CAT_COLS = ["country_code", "http_request_country", "clean_os", "payer_provider_payer_country"]

HIGH_RISK_COUNTRIES = [
    "Tunisia", "Bangladesh", "Indonesia", "Jamaica", "Nigeria", "Pakistan",
    "Algeria", "Morocco", "India", "Saudi Arabia", "Oman", "Jordan", "Mexico", "Italy",
]

ID_COLS_TO_EXCLUDE = [
    "token", "user_id", "seller_id", "gig_id", "order_id",
    "email", "ip", "payer_provider_payer_id",
    # v3: user_txns_30d is inaccurate in source data (dm_paypal_fraud_daily).
    # Excluded from XGBoost features. Rules that depended on it now use
    # user_txns_24h instead (verified accurate). Same for txn_rate_signup
    # which is now computed from user_txns_total / days_since_signup.
    "user_txns_30d",
    # v7: these columns exist in dm_paypal_fraud_weekly but NOT in
    # dm_paypal_fraud_daily, so the model would train on features that are
    # missing at scoring time. Excluding so weekly only learns from features
    # daily can actually provide.
    "avg_payment_amount_14d",
    "buyer_country",
    "os_diversity_24h",
    "reused_same_paypal_payer_id_30d",
    "seller_txns_14d",
    "user_txns_1y",
    # v13: dropped — mean(|SHAP|)=0.0 on test split, XGBoost never split on them.
    # Clean drops: none feed any rule_/combo_/anti_rule_, so removing them costs
    # nothing. Cuts XGBoost features 39 -> 32. (is_paypal_after_decline and
    # seller_fraud_14d were also 0.0 but feed rules, so kept in the raw df.)
    "captured",
    "seller_service_14d",
    "seller_service_30d",
    "used_paypal_in_30d",
    "buyer_email_changed_last_10d",
    "seller_email_changed_last_10d",
    "txn_rate_signup",
]

# =================== QUERY ===================
q = r"""
SELECT *
FROM `fiverr-dwh-data-prod.dwh.dm_paypal_fraud_weekly`
WHERE _PARTITIONdate = current_date()-1
"""


# =================== DATA LOADING ===================
def load_data(query_string, bq_client):
    print("[load_data] Running BigQuery query...")
    df = bq_client.query(query_string).to_dataframe()
    print(f"[load_data] Shape: {df.shape}")
    return df


# =================== HELPERS ===================
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


# =================== ADAPTIVE THRESHOLDS ===================
def calculate_adaptive_thresholds(df_train, target_col=LABEL_COL, method="percentile"):
    thresholds = {}

    if method == "percentile":
        print("[info] Calculating percentile-based thresholds...")

        thresholds["unique_ips_last_24h"] = df_train["unique_ips_last_24h"].quantile(0.90)
        thresholds["user_txns_1h"] = df_train["user_txns_1h"].quantile(0.95)
        thresholds["user_txns_24h"] = df_train["user_txns_24h"].quantile(0.90)
        thresholds["buyer_count_clone"] = df_train["buyer_count_clone"].quantile(0.90)
        thresholds["user_txns_30d_high"] = df_train["user_txns_30d"].quantile(0.80)
        thresholds["payment_amount_high"] = df_train["payment_amount"].quantile(0.95)

        thresholds["total_orders_buyer_low"] = (
            df_train["total_orders_buyer"].quantile(0.15)
            if "total_orders_buyer" in df_train.columns else 3
        )
        thresholds["total_orders_seller_low"] = (
            df_train["total_orders_seller"].quantile(0.15)
            if "total_orders_seller" in df_train.columns else 3
        )

        mask_repeat_pairs = (
            (df_train["total_orders_buyer_seller"] >= 3) &
            (df_train["user_txns_24h"] >= 3)
        )
        if "total_payment_buyer_seller_24h" in df_train.columns:
            filtered = df_train.loc[mask_repeat_pairs, "total_payment_buyer_seller_24h"]
            thresholds["total_payment_buyer_seller_24h"] = filtered.quantile(0.95)
        else:
            thresholds["total_payment_buyer_seller_24h"] = 300

        if "seller_avg_order_amount_to_date" in df_train.columns:
            avg_c = pd.to_numeric(df_train["seller_avg_order_amount_to_date"], errors="coerce")
            curr_c = pd.to_numeric(df_train["payment_amount"], errors="coerce")
            prev_c = pd.to_numeric(df_train.get("total_orders_seller", 0), errors="coerce")
            temp_ratios = curr_c / avg_c
            valid_ratios = temp_ratios[(avg_c > 0) & (prev_c >= 3)].dropna()
            thresholds["seller_txn_ratio_high"] = (
                valid_ratios.quantile(0.95) if not valid_ratios.empty else 5
            )

        # Combo rarity (v3: use user_txns_24h, the accurate source field)
        df_train["combo_high_volume"] = (
            (df_train["payment_amount"] > thresholds["payment_amount_high"]) &
            (df_train["user_txns_24h"] > thresholds["user_txns_24h"])
        )
        thresholds["combo_high_volume_rate"] = df_train["combo_high_volume"].mean()

        if "is_new_user_7d" in df_train.columns:
            df_train["combo_new_user_high_volume"] = (
                (df_train["is_new_user_7d"] == 1) &
                (df_train["user_txns_24h"] > thresholds["user_txns_24h"])
            )
            thresholds["combo_new_user_high_volume_rate"] = df_train["combo_new_user_high_volume"].mean()
        else:
            thresholds["combo_new_user_high_volume_rate"] = 0.0

        if "total_payment_buyer_seller_24h" in df_train.columns:
            df_train["combo_repeat_pair_high_value"] = (
                (df_train["total_orders_buyer_seller"] >= 3) &
                (df_train["user_txns_24h"] >= 3) &
                (df_train["total_payment_buyer_seller_24h"] > thresholds["total_payment_buyer_seller_24h"])
            )
            thresholds["combo_repeat_pair_high_value_rate"] = df_train["combo_repeat_pair_high_value"].mean()
        else:
            thresholds["combo_repeat_pair_high_value_rate"] = 0.0

        if "total_orders_buyer" in df_train.columns and "total_orders_seller" in df_train.columns:
            df_train["combo_low_buyer_seller"] = (
                (df_train["total_orders_buyer"] <= thresholds["total_orders_buyer_low"]) &
                (df_train["total_orders_seller"] <= thresholds["total_orders_seller_low"])
            )
            thresholds["combo_low_buyer_seller_rate"] = df_train["combo_low_buyer_seller"].mean()
        else:
            thresholds["combo_low_buyer_seller_rate"] = 0.0

    defaults = {
        "unique_ips_last_24h": 2, "user_txns_1h": 1, "user_txns_24h": 2,
        "buyer_count_clone": 5, "user_txns_30d_high": 5, "payment_amount_high": 500,
        "secs_since_prev": 60, "seller_txn_ratio_high": 5,
    }
    for k, v in defaults.items():
        thresholds.setdefault(k, v)

    return thresholds, df_train


# =================== RULES ===================
def build_rule_columns(df, thresholds=None):
    df = df.copy()
    if thresholds is None:
        thresholds = {
            "unique_ips_last_24h": 2, "user_txns_1h": 1, "user_txns_24h": 2,
            "buyer_count_clone": 5, "user_txns_30d_high": 5, "payment_amount_high": 500,
            "secs_since_prev": 60, "seller_txn_ratio_high": 5,
            "total_orders_buyer_low": 3, "total_orders_seller_low": 3,
        }

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
    df["rule_unique_ips_last_24h"] = (num("unique_ips_last_24h") > thresholds["unique_ips_last_24h"])
    df["rule_user_txns_1h"] = (num("user_txns_1h") > thresholds["user_txns_1h"])
    df["rule_user_txns_24h"] = (num("user_txns_24h") > thresholds["user_txns_24h"])
    df["rule_buyer_count_clone"] = (num("buyer_count_clone") > thresholds["buyer_count_clone"])
    df["rule_is_fake_location"] = (df["is_fake_location"] == True) if "is_fake_location" in df.columns else False

    # v3: rules that used user_txns_30d now use user_txns_24h (accurate in source).
    df["rule_high_volume"] = (
        (num("payment_amount") > thresholds["payment_amount_high"]) &
        (num("user_txns_24h") > thresholds["user_txns_24h"])
    ).astype(int)
    df["rule_high_volume_score"] = np.where(
        df["rule_high_volume"], 1 - thresholds.get("combo_high_volume_rate", 0.01), 0
    ).astype(float)

    if "days_since_signup" in df.columns:
        df["rule_new_user_60d"] = (num("days_since_signup") < 60)

    df["combo_high_volume"] = (
        (num("payment_amount") > thresholds["payment_amount_high"]) &
        (num("user_txns_24h") > thresholds["user_txns_24h"])
    ).astype(int)
    df["combo_new_user_high_volume"] = (
        (df["rule_is_new_user_7d"] == True) &
        (num("user_txns_24h") > thresholds["user_txns_24h"])
    ).astype(int)
    df["combo_repeat_pair_high_value"] = (
        (num("total_orders_buyer_seller") >= 3) &
        (num("user_txns_24h") >= 3) &
        (num("total_payment_buyer_seller_24h") > thresholds["total_payment_buyer_seller_24h"])
    ).astype(int)
    df["combo_low_buyer_seller"] = (
        (num("total_orders_buyer") <= thresholds["total_orders_buyer_low"]) &
        (num("total_orders_seller") <= thresholds["total_orders_seller_low"])
    ).astype(int)

    # ====== ANTI-RULES (v2): protective signals that REDUCE risk score ======
    # Why: in Top-10 tracking, established sellers/buyers with many clones
    # kept getting flagged as FP. These rules tell the model "trust this account".
    df["anti_rule_mature_seller"] = (
        df.get("seller_level", pd.Series(dtype=str)).astype(str).isin(
            ["SELLER_LEVEL_ONE", "SELLER_LEVEL_TWO"]
        )
        & (num("total_orders_seller") > 20)
        & (num("seller_fraud_30d") == 0)
        & (num("seller_fraud_14d") == 0)
    )

    # v5: don't protect a mature buyer when there's an amount spike (potential ATO).
    # amt_ratio = payment / 30d-avg. If ratio >= 3, the buyer is paying way more
    # than usual — could be account takeover, so withhold the anti-rule protection.
    # v8: restrict to is_team=0 (non-team). Team buyers need 2 years before
    # they're protective — handled separately by anti_rule_established_team.
    _amt_ratio = num("payment_amount") / num("user_amt_mean_30d").replace(0, np.nan)
    df["anti_rule_mature_buyer"] = (
        (num("is_team") == 0)
        & (num("total_orders_buyer") > 50)
        & (num("days_since_signup") > 180)
        & ((_amt_ratio < 3) | _amt_ratio.isna())
    )

    # v5: positive rule for ATO-style amount spikes (current payment >> avg).
    df["rule_amount_spike"] = (_amt_ratio >= 5).fillna(False)

    df["anti_rule_high_volume_clean_seller"] = (
        (num("total_orders_seller") > 100)
        & (num("seller_fraud_30d") == 0)
    )

    # v4: team accounts with many orders are legitimate (they're companies/agencies).
    # v8: tightened to 730 days. Data shows teams with <2 years history have
    # LIFT = 1.0 (no protection earned). Only after 2 years do they reach lift 0.13.
    df["anti_rule_established_team"] = (
        (num("is_team") == 1)
        & (num("total_orders_buyer") > 50)
        & (num("days_since_signup") > 730)
        & (num("seller_fraud_30d") == 0)
    )

    return df


def compute_rule_weights(df_train):
    # v2: also computes negative weights for anti_rule_* columns.
    rule_cols = [c for c in df_train.columns
                 if c.startswith("rule_") or c.startswith("anti_rule_")]
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

        if c.startswith("anti_rule_"):
            # Anti-rule: low lift = strong protection. Convert to negative weight.
            # lift >= 1 => not really an anti-signal => w = 0
            # lift ~ 0 => fires only on safe txns => strong negative w
            if lift >= 1.0:
                w = 0.0
            else:
                anti_lift = 1.0 / max(lift, 0.05)
                w = -float(np.clip((anti_lift ** 0.5) * (1 + 3 * corr), 0.5, 10))
        else:
            w = float(np.clip((lift ** 1.5) * (1 + 3 * corr), 0.5, 25))
        weights[c] = w
    return weights


def compute_manual_risk(df, weights, mn=None, mx=None):
    """Compute manual_risk_score for df using `weights`.

    v2: weights for anti_rule_* are negative (already computed in compute_rule_weights),
        and contribute as subtraction from manual_risk_raw. Normalization still maps [0,1].

    F1: If mn/mx are None, they're fit from this df (use on TRAIN).
        If provided, they're applied directly (use on VAL/TEST, daily scoring).
    Returns (df, mn, mx) so the caller can persist the train-fit stats.
    """
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
        # Sign of `w` already encodes positive/negative (anti_rule_* has w<0)
        if pd.api.types.is_bool_dtype(valid_vals):
            contrib = valid_vals.astype(bool).astype(int) * w
        elif valid_vals.dropna().isin([0, 1, True, False]).all():
            contrib = valid_vals.astype(bool).astype(int) * w
        elif valid_vals.dropna().between(0, 1).all():
            contrib = valid_vals.astype(float) * w
        else:
            contrib = valid_vals.astype(float).clip(0, 1) * w
        df.loc[valid_mask, "manual_risk_raw"] += contrib
    if mn is None:
        mn = float(df["manual_risk_raw"].min())
    if mx is None:
        mx = float(df["manual_risk_raw"].max())
    df["manual_risk_score"] = (df["manual_risk_raw"] - mn) / (mx - mn + 1e-9)
    df.drop(columns=["manual_risk_raw"], inplace=True)
    return df, float(mn), float(mx)


def add_derived_features(df, iforest_mean=None, iforest_std=None):
    """F2: normalize iforest_score with TRAIN mean/std (no per-split leakage).
    If mean/std are None, they're fit from this df. If provided, applied directly.
    Returns (df, iforest_mean, iforest_std)."""
    df = df.copy()
    def num(c):
        return pd.to_numeric(df.get(c, np.nan), errors="coerce")
    # v3: use user_txns_total (accurate) instead of user_txns_30d (inaccurate in source).
    df["txn_rate_signup"] = num("user_txns_total") / (num("days_since_signup") + 1)
    df["amt_ratio_to_mean"] = np.where(
        pd.notna(num("user_amt_mean_30d")) & (num("user_amt_mean_30d") > 0),
        num("payment_amount") / num("user_amt_mean_30d"),
        np.nan,
    )
    s = num("iforest_score")
    if iforest_mean is None:
        iforest_mean = float(s.mean())
    if iforest_std is None:
        iforest_std = float(s.std())
    df["normalized_iforest"] = (s - iforest_mean) / (iforest_std + 1e-9)
    return df, float(iforest_mean), float(iforest_std)


# =================== MAIN PIPELINE ===================
def run_pipeline(df, threshold_method="percentile"):
    print("=" * 80)
    print("   Fiverr Fraud Detection Pipeline - v13.0")
    print("=" * 80)

    df = ensure_datetime(df)
    df = safe_fillna(df)

    # v6: exclude Fiverr employees — they should never be scored as fraud.
    if "is_fiverr_employee" in df.columns:
        before = len(df)
        df = df[df["is_fiverr_employee"] != True].copy()
        dropped = before - len(df)
        if dropped:
            print(f"[info] Dropped {dropped} rows where is_fiverr_employee=TRUE")

    IGNORE_RECENT_DAYS = 40
    max_ts = df[DATE_COL].max()
    cutoff = max_ts - pd.Timedelta(days=IGNORE_RECENT_DAYS)
    df = df[df[DATE_COL] < cutoff].copy()
    print(f"[info] Excluding last {IGNORE_RECENT_DAYS} days (up to {max_ts})")

    if "seller_pro" in df.columns:
        df["seller_pro"] = df["seller_pro"].fillna(False).astype(bool)

    m_train, m_val, m_test = time_split(df)
    df_train = df[m_train].copy()
    df_val = df[m_val].copy()
    df_test = df[m_test].copy()

    print(f"Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    print(f"Fraud rate (train): {df_train[LABEL_COL].mean():.4f}")

    thresholds, df_train = calculate_adaptive_thresholds(df_train, LABEL_COL, method=threshold_method)
    df_train = build_rule_columns(df_train, thresholds)
    df_val = build_rule_columns(df_val, thresholds)
    df_test = build_rule_columns(df_test, thresholds)

    label_encoders = {}
    for col in CAT_COLS:
        if col in df_train.columns:
            le = LabelEncoder()
            train_vals = df_train[col].astype(str).fillna("UNKNOWN").unique().tolist()
            if "UNKNOWN" not in train_vals:
                train_vals.append("UNKNOWN")
            le.fit(train_vals)
            label_encoders[col] = le
            known = set(le.classes_)
            for part in (df_train, df_val, df_test):
                part[col] = (
                    part[col].astype(str).fillna("UNKNOWN")
                    .apply(lambda x: x if x in known else "UNKNOWN")
                )
                part[col] = le.transform(part[col])

    print("\n[info] Training IsolationForest...")
    num_cols = [
        c for c in df_train.select_dtypes(include=[np.number]).columns
        if c not in [LABEL_COL] + ID_COLS_TO_EXCLUDE
    ]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(df_train[num_cols])
    for part in (df_train, df_val, df_test):
        part["iforest_score"] = -iso.score_samples(part[num_cols])

    # F2: fit iforest mean/std on TRAIN, apply same stats to val/test
    df_train, iforest_mean, iforest_std = add_derived_features(df_train)
    df_val, _, _ = add_derived_features(df_val, iforest_mean, iforest_std)
    df_test, _, _ = add_derived_features(df_test, iforest_mean, iforest_std)

    # F1: fit manual_risk min/max on TRAIN, apply same stats to val/test
    weights = compute_rule_weights(df_train)
    df_train, manual_risk_mn, manual_risk_mx = compute_manual_risk(df_train, weights)
    df_val, _, _ = compute_manual_risk(df_val, weights, mn=manual_risk_mn, mx=manual_risk_mx)
    df_test, _, _ = compute_manual_risk(df_test, weights, mn=manual_risk_mn, mx=manual_risk_mx)

    # F3: exclude manual_risk_score from features (it's used in the blend separately).
    # F6: exclude rule_*/combo_*/anti_rule_* from features (already in manual_risk_score).
    exclude_cols = [LABEL_COL, DATE_COL] + ID_COLS_TO_EXCLUDE + ["manual_risk_score"]
    rule_and_combo = [c for c in df_train.columns
                      if c.startswith("rule_") or c.startswith("combo_") or c.startswith("anti_rule_")]
    exclude_cols = exclude_cols + rule_and_combo
    feats = [c for c in df_train.columns if c not in exclude_cols]

    Xtr = df_train[feats].select_dtypes(include=[np.number])
    ytr = df_train[LABEL_COL].astype(int)
    Xva = df_val[feats].select_dtypes(include=[np.number])
    yva = df_val[LABEL_COL].astype(int)
    Xte = df_test[feats].select_dtypes(include=[np.number])
    yte = df_test[LABEL_COL].astype(int)

    # Align column order across splits
    Xva = Xva.reindex(columns=Xtr.columns)
    Xte = Xte.reindex(columns=Xtr.columns)

    dtrain = xgb.DMatrix(Xtr, label=ytr, missing=np.nan)
    dval = xgb.DMatrix(Xva, label=yva, missing=np.nan)
    dtest = xgb.DMatrix(Xte, label=yte, missing=np.nan)

    print(f"\n[info] Training XGBoost on {Xtr.shape[1]} features ...")
    # v13: scale_pos_weight = 1 (no weighting). Time-series CV (spw_cv.py, 5 folds)
    # showed spw=1 beats sqrt(neg/pos) on test AUPRC 5/5 folds (mean +0.0158).
    # The blend already injects fraud signal via manual_risk_score, so weighting
    # the rare class only inflates probabilities / FPs and lowers AUPRC+precision.
    # (Was sqrt(neg/pos) ~10; full ~104 was worse still — AUPRC monotonically
    # decreases as spw rises.) ROC dips ~0.007 — acceptable, AUPRC is the metric.
    # F7: tuned hyperparameters from grid search on val AUPRC.
    # v13: max_depth 4->2. Grid search (overfit_grid.py) on the 32-feature set
    # showed depth=2 cuts train->test AUPRC gap 0.297 -> 0.039 (overfit ~gone)
    # while RAISING test AUPRC 0.3512 -> 0.3676. depth=4 was memorizing deep
    # interactions that didn't generalize. Revisit if new features are added.
    params = {
        "max_depth": 2, "learning_rate": 0.05, "subsample": 0.7,
        "colsample_bytree": 0.8, "min_child_weight": 20, "reg_lambda": 5,
        "objective": "binary:logistic",
        "eval_metric": ["aucpr"], "tree_method": "hist",
        "scale_pos_weight": 1.0,
        "seed": 42,
    }
    model = xgb.train(
        params, dtrain, num_boost_round=1500,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=100,
    )

    # v13.1: normalize the ML score to [0,1] using TRAIN q01-q99, so it shares
    # manual_risk's scale before blending. With scale_pos_weight=1 the raw
    # probabilities sit near 0 (calibrated to the 0.8% base rate), so an
    # un-normalized blend was rules-dominated regardless of alpha and the top
    # fraud got an alarmingly low final_score. Monotonic transform -> ML ranking
    # unchanged; refit alpha. q01/q99 (not min/max) is robust to a single outlier.
    # ml_lo/ml_hi saved to normalization_stats so daily applies the same mapping.
    p_train = model.predict(dtrain)
    ml_lo = float(np.quantile(p_train, 0.01))
    ml_hi = float(np.quantile(p_train, 0.99))

    def ml_norm(p):
        return np.clip((p - ml_lo) / (ml_hi - ml_lo + 1e-9), 0.0, 1.0)

    preds_model_val = ml_norm(model.predict(dval))
    preds_rules_val = df_val["manual_risk_score"].to_numpy()
    best_alpha, best_metric = 0.5, -1
    for a in np.arange(0.1, 0.95, 0.05):
        blend = a * preds_model_val + (1 - a) * preds_rules_val
        score = average_precision_score(yva, blend)
        if score > best_metric:
            best_alpha, best_metric = a, score

    alpha_blend = best_alpha
    preds_blend = (
        alpha_blend * ml_norm(model.predict(dtest)) +
        (1 - alpha_blend) * df_test["manual_risk_score"].to_numpy()
    )

    auc_score = roc_auc_score(yte, preds_blend)
    auprc_score = average_precision_score(yte, preds_blend)
    print(f"\nTest ROC-AUC:  {auc_score:.4f}")
    print(f"Test AUPRC:    {auprc_score:.4f}")
    print(f"Blending Alpha: {alpha_blend:.2f}")

    # =================== OVERFIT CHECK (train vs val vs test) ===================
    # Same blend (alpha * XGB + (1-alpha) * manual_risk) evaluated on all 3 splits.
    # Big train>>test gap = overfit. XGBoost already has early-stopping on val.
    print("\n[overfit check] blended metrics per split (same alpha):")
    splits = [
        ("train", dtrain, ytr, df_train["manual_risk_score"].to_numpy()),
        ("val",   dval,   yva, df_val["manual_risk_score"].to_numpy()),
        ("test",  dtest,  yte, df_test["manual_risk_score"].to_numpy()),
    ]
    print(f"  {'split':<6} {'ROC-AUC':>9} {'AUPRC':>9}")
    gaps = {}
    for name, dmat, y, rules in splits:
        blend = alpha_blend * ml_norm(model.predict(dmat)) + (1 - alpha_blend) * rules
        roc = roc_auc_score(y, blend)
        ap = average_precision_score(y, blend)
        gaps[name] = (roc, ap)
        print(f"  {name:<6} {roc:>9.4f} {ap:>9.4f}")
    print(f"  [gap] train-test AUPRC: {gaps['train'][1] - gaps['test'][1]:+.4f}"
          f"  | train-test ROC: {gaps['train'][0] - gaps['test'][0]:+.4f}")
    # also the raw XGBoost component alone (no rules) — isolates model overfit
    print("[overfit check] XGBoost-only AUPRC (no blend):")
    for name, dmat, y, _ in splits:
        ap = average_precision_score(y, model.predict(dmat))
        print(f"  {name:<6} {ap:>9.4f}")

    # =================== SHAP FEATURE IMPACT ===================
    # Per-feature impact on the XGBoost model, computed on the test split.
    # TreeExplainer is exact + fast for XGBoost. Two artifacts:
    #   shap_summary.png  — beeswarm: direction + magnitude per feature
    #   shap_importance.png — bar: mean(|SHAP|) ranking
    try:
        import shap
        print("\n[info] Computing SHAP values (TreeExplainer) on test split...")
        # cap rows for speed; beeswarm is unreadable above a few thousand anyway
        Xte_shap = Xte if len(Xte) <= 5000 else Xte.sample(5000, random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xte_shap)

        n_feats = Xte_shap.shape[1]
        plt.figure(figsize=(10, max(8, 0.32 * n_feats)))
        shap.summary_plot(shap_values, Xte_shap, show=False, max_display=n_feats)
        plt.title("SHAP — feature impact (test split) v12.0")
        shap_summary_path = os.path.join(LOCAL_ARTIFACT_DIR, "shap_summary.png")
        plt.savefig(shap_summary_path, dpi=120, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, max(8, 0.32 * n_feats)))
        shap.summary_plot(shap_values, Xte_shap, plot_type="bar", show=False, max_display=n_feats)
        plt.title("SHAP — mean(|impact|) per feature v12.0")
        shap_bar_path = os.path.join(LOCAL_ARTIFACT_DIR, "shap_importance.png")
        plt.savefig(shap_bar_path, dpi=120, bbox_inches="tight")
        plt.close()

        # also dump the ranking as CSV
        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_rank = pd.DataFrame(
            {"feature": Xte_shap.columns, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_csv_path = os.path.join(LOCAL_ARTIFACT_DIR, "shap_importance.csv")
        shap_rank.to_csv(shap_csv_path, index=False)
        print(f"[✓] SHAP artifacts: {shap_summary_path}, {shap_bar_path}, {shap_csv_path}")
        print("\nTop 15 features by mean(|SHAP|):")
        print(shap_rank.head(15).to_string(index=False))
    except Exception as e:
        print(f"[warn] SHAP step failed: {e}")

    # Save PR curve as image (no plt.show() in CI)
    prec, rec, _ = precision_recall_curve(yte, preds_blend)
    pr_auc_val = auc(rec, prec)
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, color="darkorange", lw=2, label=f"PR Curve (AUC={pr_auc_val:.3f})")
    plt.fill_between(rec, prec, alpha=0.2, color="orange")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - v11.1")
    plt.legend(loc="lower left"); plt.grid(True, linestyle="--", alpha=0.6)
    plot_path = os.path.join(LOCAL_ARTIFACT_DIR, "pr_curve.png")
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()

    # =================== SAVE ARTIFACTS LOCALLY ===================
    print("\n[info] Saving artifacts locally...")

    def save(obj, name, ext):
        path = os.path.join(LOCAL_ARTIFACT_DIR, f"{name}.{ext}")
        if ext == "joblib":
            joblib.dump(obj, path)
        elif ext == "txt":
            with open(path, "w") as f:
                f.write(str(obj))
        return path

    # Normalization stats — daily must use these so cross-day scores stay comparable.
    normalization_stats = {
        "manual_risk_mn": manual_risk_mn,
        "manual_risk_mx": manual_risk_mx,
        "iforest_mean": iforest_mean,
        "iforest_std": iforest_std,
        # v13.1: ML-score normalization range (train q01-q99) for daily's blend.
        "ml_lo": ml_lo,
        "ml_hi": ml_hi,
    }

    artifact_paths = [
        save(label_encoders, "label_encoders", "joblib"),
        save(model, "fraud_model", "joblib"),
        save(iso, "iforest_model", "joblib"),
        save(list(num_cols), "iforest_features", "joblib"),
        save(list(Xtr.columns), "fraud_features", "joblib"),
        save(weights, "rule_weights", "joblib"),
        save(thresholds, "rule_thresholds", "joblib"),
        save(normalization_stats, "normalization_stats", "joblib"),
        save(alpha_blend, "blend_alpha", "txt"),
        plot_path,
    ]

    pd.DataFrame({"y_true": yte, "preds_blend": preds_blend}).to_csv(
        os.path.join(LOCAL_ARTIFACT_DIR, "fraud_predictions.csv"), index=False
    )
    artifact_paths.append(os.path.join(LOCAL_ARTIFACT_DIR, "fraud_predictions.csv"))

    importances = model.get_score(importance_type="weight")
    pd.DataFrame(
        sorted(importances.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "weight"],
    ).to_csv(os.path.join(LOCAL_ARTIFACT_DIR, "feature_importance.csv"), index=False)
    artifact_paths.append(os.path.join(LOCAL_ARTIFACT_DIR, "feature_importance.csv"))

    print(f"\n[✓] All artifacts saved to {LOCAL_ARTIFACT_DIR} (GitHub Actions will upload them)")
    return model, list(Xtr.columns), alpha_blend, thresholds


# =================== ENTRY POINT ===================
if __name__ == "__main__":
    print("=== Weekly Fraud Detection Pipeline: Start ===")
    df = load_data(q, client)
    model, feats, alpha, thresholds = run_pipeline(df, threshold_method="percentile")
    print("\n[Summary] Pipeline completed successfully!")
    print(f"  - Blending α: {alpha:.2f}")
    print(f"  - Artifacts saved to: {LOCAL_ARTIFACT_DIR}")
