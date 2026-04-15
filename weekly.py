# -*- coding: utf-8 -*-
"""
Fiverr Fraud Detection - Weekly Training Pipeline (v11.1)
Runs every Sunday at 08:00 Israel time via GitHub Actions.
Saves trained artifacts to Google Drive (ARTIFACTS_FOLDER_ID).
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

from google.oauth2 import service_account
from google.cloud import bigquery

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import BorderlineSMOTE

from drive_utils import get_credentials, upload_file_to_drive, get_drive_service

# =================== AUTH ===================
SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/drive",
]

credentials = get_credentials()
client = bigquery.Client(
    project="fiverr-bq-payments-adhoc-prod",
    credentials=credentials,
)
print("BigQuery client initialized.")

ARTIFACTS_FOLDER_ID = os.environ["ARTIFACTS_FOLDER_ID"]
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

        # Combo rarity
        df_train["combo_high_volume"] = (
            (df_train["payment_amount"] > thresholds["payment_amount_high"]) &
            (df_train["user_txns_30d"] > thresholds["user_txns_30d_high"])
        )
        thresholds["combo_high_volume_rate"] = df_train["combo_high_volume"].mean()

        if "is_new_user_7d" in df_train.columns:
            df_train["combo_new_user_high_volume"] = (
                (df_train["is_new_user_7d"] == 1) &
                (df_train["user_txns_30d"] > thresholds["user_txns_30d_high"])
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

    df["rule_high_volume"] = (
        (num("payment_amount") > thresholds["payment_amount_high"]) &
        (num("user_txns_30d") > thresholds["user_txns_30d_high"])
    ).astype(int)
    df["rule_high_volume_score"] = np.where(
        df["rule_high_volume"], 1 - thresholds.get("combo_high_volume_rate", 0.01), 0
    ).astype(float)

    if "days_since_signup" in df.columns:
        df["rule_new_user_60d"] = (num("days_since_signup") < 60)

    df["combo_high_volume"] = (
        (num("payment_amount") > thresholds["payment_amount_high"]) &
        (num("user_txns_30d") > thresholds["user_txns_30d_high"])
    ).astype(int)
    df["combo_new_user_high_volume"] = (
        (df["rule_is_new_user_7d"] == True) &
        (num("user_txns_30d") > thresholds["user_txns_30d_high"])
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

    return df


def compute_rule_weights(df_train):
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
        w = np.clip((lift ** 1.5) * (1 + 3 * corr), 0.5, 25)
        weights[c] = float(w)
    return weights


def compute_manual_risk(df, weights):
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


def add_derived_features(df):
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


# =================== MAIN PIPELINE ===================
def run_pipeline(df, threshold_method="percentile"):
    print("=" * 80)
    print("   Fiverr Fraud Detection Pipeline - v11.1")
    print("=" * 80)

    df = ensure_datetime(df)
    df = safe_fillna(df)

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
        if c not in [LABEL_COL, "manual_risk_score"] + ID_COLS_TO_EXCLUDE
    ]
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
    iso.fit(df_train[num_cols])
    for part in (df_train, df_val, df_test):
        part["iforest_score"] = -iso.score_samples(part[num_cols])

    df_train = add_derived_features(df_train)
    df_val = add_derived_features(df_val)
    df_test = add_derived_features(df_test)

    weights = compute_rule_weights(df_train)
    df_train = compute_manual_risk(df_train, weights)
    df_val = compute_manual_risk(df_val, weights)
    df_test = compute_manual_risk(df_test, weights)

    exclude_cols = [LABEL_COL, DATE_COL] + ID_COLS_TO_EXCLUDE
    feats = [c for c in df_train.columns if c not in exclude_cols]

    Xtr = df_train[feats].select_dtypes(include=[np.number])
    ytr = df_train[LABEL_COL].astype(int)
    Xva = df_val[feats].select_dtypes(include=[np.number])
    yva = df_val[LABEL_COL].astype(int)
    Xte = df_test[feats].select_dtypes(include=[np.number])
    yte = df_test[LABEL_COL].astype(int)

    dtrain = xgb.DMatrix(Xtr, label=ytr, missing=np.nan)
    dval = xgb.DMatrix(Xva, label=yva, missing=np.nan)
    dtest = xgb.DMatrix(Xte, label=yte, missing=np.nan)

    print("\n[info] Training XGBoost...")
    params = {
        "max_depth": 7, "learning_rate": 0.05, "subsample": 0.7,
        "colsample_bytree": 0.8, "objective": "binary:logistic",
        "eval_metric": ["aucpr"], "tree_method": "hist",
        "scale_pos_weight": (len(ytr) - ytr.sum()) / max(ytr.sum(), 1),
    }
    model = xgb.train(
        params, dtrain, num_boost_round=1500,
        evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=100,
    )

    preds_model_val = model.predict(dval)
    preds_rules_val = df_val["manual_risk_score"].to_numpy()
    best_alpha, best_metric = 0.5, -1
    for a in np.arange(0.1, 0.95, 0.05):
        blend = a * preds_model_val + (1 - a) * preds_rules_val
        score = average_precision_score(yva, blend)
        if score > best_metric:
            best_alpha, best_metric = a, score

    alpha_blend = best_alpha
    preds_blend = (
        alpha_blend * model.predict(dtest) +
        (1 - alpha_blend) * df_test["manual_risk_score"].to_numpy()
    )

    auc_score = roc_auc_score(yte, preds_blend)
    auprc_score = average_precision_score(yte, preds_blend)
    print(f"\nTest ROC-AUC:  {auc_score:.4f}")
    print(f"Test AUPRC:    {auprc_score:.4f}")
    print(f"Blending Alpha: {alpha_blend:.2f}")

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

    artifact_paths = [
        save(label_encoders, "label_encoders", "joblib"),
        save(model, "fraud_model", "joblib"),
        save(iso, "iforest_model", "joblib"),
        save(list(Xtr.columns), "fraud_features", "joblib"),
        save(weights, "rule_weights", "joblib"),
        save(thresholds, "rule_thresholds", "joblib"),
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

    # =================== UPLOAD TO DRIVE ===================
    print("\n[info] Uploading artifacts to Google Drive...")
    drive_service = get_drive_service()
    for path in artifact_paths:
        upload_file_to_drive(path, ARTIFACTS_FOLDER_ID, drive_service)

    print(f"\n[✓] All artifacts uploaded to Drive folder: {ARTIFACTS_FOLDER_ID}")
    return model, list(Xtr.columns), alpha_blend, thresholds


# =================== ENTRY POINT ===================
if __name__ == "__main__":
    print("=== Weekly Fraud Detection Pipeline: Start ===")
    df = load_data(q, client)
    model, feats, alpha, thresholds = run_pipeline(df, threshold_method="percentile")
    print("\n[Summary] Pipeline completed successfully!")
    print(f"  - Blending α: {alpha:.2f}")
    print(f"  - Artifacts uploaded to Drive folder: {ARTIFACTS_FOLDER_ID}")
