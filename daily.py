# -*- coding: utf-8 -*-
"""
Fiverr Fraud Detection - Daily Scoring Pipeline (v11.1)
Runs every day at 11:00 Israel time via GitHub Actions.
Downloads artifacts from Google Drive, scores yesterday's data, uploads Excel result.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use("Agg")

from google.cloud import bigquery

from drive_utils import get_credentials

# =================== AUTH ===================
credentials = get_credentials()
client = bigquery.Client(project="fiverr-bq-payments-adhoc-prod", credentials=credentials)
print("BigQuery client initialized.")

LOCAL_ARTIFACT_DIR = "/tmp/fraud_artifacts"
os.makedirs(LOCAL_ARTIFACT_DIR, exist_ok=True)

# =================== CONFIG ===================
LABEL_COL = "has_chargeback"
DATE_COL = "date_hour"
CAT_COLS = ["country_code", "http_request_country", "clean_os", "payer_provider_payer_country"]

CRITICAL_RULE_WEIGHTS = {
    "rule_buyer_payer_seen_in_seller": 8.0,
    "rule_buyer_seller_shared_clone": 25.0,
    "rule_has_blocked_clone": 25.0,
    "rule_status_pay": 20,
    "rule_seller_count_clone": 8.0,
    "rule_massage_activity_all": 20.0,
    "rule_massage_activity": 20.0,
}

HIGH_RISK_COUNTRIES = [
    "Tunisia", "Bangladesh", "Indonesia", "Jamaica", "Nigeria", "Pakistan",
    "Algeria", "Morocco", "India", "Saudi Arabia", "Oman", "Jordan", "Mexico", "Italy",
]

# =================== QUERY ===================
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
print(f"Yesterday: {yesterday}")

q = f"""
SELECT *
FROM `fiverr-dwh-data-prod.dwh.dm_paypal_fraud_daily`
WHERE _partitiondate >= current_date()-1
"""




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


def encode_categoricals(df, encoders, cat_cols):
    df = df.copy()
    for col in cat_cols:
        if col not in df.columns:
            continue
        le = encoders.get(col)
        if le is None:
            df[col] = -1
            continue
        df[col] = df[col].where(df[col].notna(), "UNKNOWN").astype(str)
        known = set(le.classes_)
        df[col] = df[col].where(df[col].isin(known), "UNKNOWN")
        if "UNKNOWN" not in le.classes_:
            le.classes_ = np.append(le.classes_, "UNKNOWN")
        df[col] = le.transform(df[col])
    return df


def build_rule_columns(df, thresholds):
    df = df.copy()

    def num(c):
        return pd.to_numeric(df.get(c, 0), errors="coerce")

    df["rule_has_blocked_clone"] = (num("has_blocked_clone") == 1)
    df["rule_status_pay"] = (num("payment_amount") >= 600) & (df["order_status"] == "delivered")
    df["rule_buyer_payer_seen_in_seller"] = (num("buyer_payer_seen_in_seller") == 1)
    df["rule_buyer_seller_shared_clone"] = (num("buyer_seller_shared_clone") == 1)
    df["rule_seller_count_clone"] = (num("seller_count_clone") > 0)
    df["rule_massage_activity"] = (num("messages_in_closest_order") < 7) & (df["messages_in_closest_order"].notna())

    df["rule_user_has_multiple_payers"] = (num("user_has_multiple_payers") == 1)
    df["rule_is_new_user_7d"] = (num("is_new_user_7d") == 1)
    df["rule_seller_fraud_14d"] = (num("seller_fraud_14d") > 0)
    df["rule_seller_fraud_30d"] = (num("seller_fraud_30d") > 0)
    df["rule_is_paypal_after_decline"] = (num("is_paypal_after_decline") == 1)
    df["rule_unique_ips_last_24h"] = (num("unique_ips_last_24h") > thresholds.get("unique_ips_last_24h", 2))
    df["rule_user_txns_1h"] = (num("user_txns_1h") > thresholds.get("user_txns_1h", 1))
    df["rule_user_txns_24h"] = (num("user_txns_24h") > thresholds.get("user_txns_24h", 2))
    df["rule_is_fake_location"] = (df["is_fake_location"] == True) if "is_fake_location" in df.columns else False
    df["rule_buyer_count_clone"] = (num("buyer_count_clone") > thresholds.get("buyer_count_clone", 5))
    df["rule_high_risk_country_seller"] = df.get("http_request_country", pd.Series(dtype=str)).isin(HIGH_RISK_COUNTRIES)
    df["rule_is_team"] = (df["is_team"] == 1) & (df["total_orders_buyer"] <= 10)

    df["rule_new_user_from_high_risk_country"] = (
        df.get("http_request_country", pd.Series(dtype=str)).isin(HIGH_RISK_COUNTRIES) &
        (df["rule_is_new_user_7d"] == True)
    ).astype(int)

    if "days_since_signup" in df.columns:
        df["rule_new_user_60d"] = (num("days_since_signup") < 60)

    df["combo_high_volume"] = (
        (num("payment_amount") > thresholds.get("payment_amount_high", 500)) &
        (num("user_txns_30d") > thresholds.get("user_txns_30d_high", 5))
    ).astype(int)
    df["combo_new_user_high_volume"] = (
        (df["rule_is_new_user_7d"] == True) &
        (num("user_txns_30d") > thresholds.get("user_txns_30d_high", 5))
    ).astype(int)
    df["combo_repeat_pair_high_value"] = (
        (num("total_orders_buyer_seller") >= 3) &
        (num("user_txns_24h") >= 3) &
        (num("total_payment_buyer_seller_24h") > thresholds.get("total_payment_buyer_seller_24h", 300))
    ).astype(int)
    df["combo_low_buyer_seller"] = (
        (num("total_orders_buyer") <= thresholds.get("total_orders_buyer_low", 3)) &
        (num("total_orders_seller") <= thresholds.get("total_orders_seller_low", 3))
    ).astype(int)

    df["rule_high_volume"] = df["combo_high_volume"].astype(bool)
    df["rule_new_user_high_volume"] = df["combo_new_user_high_volume"].astype(bool)
    df["rule_massage_activity_all"] = (
        (df["rule_is_new_user_7d"] == 1) &
        (num("user_txns_30d") > thresholds.get("user_txns_30d_high", 5))
    )

    return df


def merge_rule_weights(weekly_weights, critical_rules, df):
    merged = weekly_weights.copy()
    global_rate = max(df[LABEL_COL].mean() if LABEL_COL in df.columns else 0.01, 1e-6)
    for rule, base_w in critical_rules.items():
        if rule in df.columns:
            col = df[rule]
            mask = col.astype(bool) if col.dropna().isin([0, 1, True, False]).all() else (col > 0)
            lift = (
                df.loc[mask, LABEL_COL].mean() / global_rate
                if LABEL_COL in df.columns and mask.sum() > 0 else 1.5
            )
            merged[rule] = np.clip((lift ** 1.5) * base_w, base_w * 0.5, base_w * 2.5)
        else:
            merged[rule] = base_w
    return merged


def compute_manual_risk(df, weights):
    df = df.copy()
    df["manual_risk_raw"] = 0.0
    for c, w in weights.items():
        if c not in df.columns:
            continue
        col = df[c]
        if col.isna().all():
            continue
        col = col.fillna(False) if pd.api.types.is_bool_dtype(col) else col.fillna(0)
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
    df["manual_risk_score"] = (
        (df["manual_risk_raw"] - mn) / (mx - mn + 1e-9)
        if mx != mn else 0.0
    )
    df.drop(columns=["manual_risk_raw"], inplace=True)
    return df


# =================== EXPLAINABILITY ===================
def explain_transaction_scores(df, model, features, weights, thresholds):
    df = df.copy()
    rule_cols = [c for c in df.columns if c.startswith("rule_")]
    triggered_rules, rule_details, rule_contributions = [], [], []

    for _, row in df.iterrows():
        trigs, details, contribs = [], [], []
        for r in rule_cols:
            if r in weights:
                val = row.get(r, 0)
                if pd.notna(val) and bool(val):
                    w = weights[r]
                    trigs.append(r); contribs.append(w)
                    details.append(f"{r} (w={w:.1f})")
        triggered_rules.append(", ".join(trigs) if trigs else "none")
        rule_details.append(" | ".join(details) if details else "none")
        rule_contributions.append(sum(contribs))

    df["triggered_rules"] = triggered_rules
    df["rule_details"] = rule_details
    df["manual_risk_contribution"] = rule_contributions

    try:
        X = df.reindex(columns=features, fill_value=0)
        dmat = xgb.DMatrix(X[features].astype(np.float64))
        contribs_mat = model.predict(dmat, pred_contribs=True)
        details_list, pos_list, neg_list = [], [], []
        for i in range(len(df)):
            vals = contribs_mat[i][:-1]
            top_idx = np.argsort(np.abs(vals))[-8:][::-1]
            det, pos, neg = [], [], []
            for j in top_idx:
                f, v = features[j], vals[j]
                val = X.iloc[i][f]
                if pd.isna(val):
                    continue
                direction = "↑" if v > 0 else "↓"
                det.append(f"{f}={val:.2f} {direction}{abs(v):.3f}")
                if v > 0.02: pos.append(f"{f}(+{v:.3f})")
                if v < -0.02: neg.append(f"{f}({v:.3f})")
            details_list.append(" | ".join(det))
            pos_list.append("; ".join(pos) if pos else "none")
            neg_list.append("; ".join(neg) if neg else "none")
        df["ml_detailed_explanation"] = details_list
        df["ml_positive_signals"] = pos_list
        df["ml_negative_signals"] = neg_list
    except Exception:
        df["ml_detailed_explanation"] = df["ml_positive_signals"] = df["ml_negative_signals"] = "unavailable"

    def dom(row):
        m, r, a = row["fraud_score_raw"], row["manual_risk_score"], row["alpha_dynamic"]
        if a * m > (1 - a) * r * 2: return "ML_Dominant"
        if (1 - a) * r > a * m * 2: return "Rules_Dominant"
        return "Balanced_Hybrid"

    df["dominant_factor"] = df.apply(dom, axis=1)

    def explain_cat(row):
        s = row["final_score"]
        if s >= 0.8: return "CRITICAL"
        if s >= 0.6: return "HIGH"
        if s >= 0.4: return "MEDIUM"
        return "LOW"

    df["risk_explanation"] = df.apply(explain_cat, axis=1)

    def compare_thr(row):
        comps = []
        for col, label in [
            ("unique_ips_last_24h", "IPs/24h"),
            ("user_txns_24h", "Txns/24h"),
            ("payment_amount", "Amount"),
        ]:
            if col in row and col in thresholds:
                v, t = row[col], thresholds[col]
                if pd.notna(v) and v > t:
                    comps.append(f"{label}: {v:.1f}>{t:.1f}")
        return " | ".join(comps) if comps else "within normal"

    df["threshold_violations"] = df.apply(compare_thr, axis=1)
    return df


def generate_summary_report(df):
    print("\n" + "=" * 80)
    print(" FRAUD DETECTION SUMMARY REPORT")
    print("=" * 80)
    print(f"Mean final score:         {df['final_score'].mean():.4f}")
    print(f"Transactions > 0.7:       {(df['final_score'] > 0.7).sum()}")
    print(f"Transactions > 0.5:       {(df['final_score'] > 0.5).sum()}")
    print("\n[Dominant factors]\n", df["dominant_factor"].value_counts().to_string())
    print("=" * 80)


def generate_email_html(df, date_str):
    """Generate an HTML email summary and save to /tmp/email_summary.html."""
    critical = df[df["final_score"] >= 0.8]
    high     = df[(df["final_score"] >= 0.6) & (df["final_score"] < 0.8)]
    total    = len(df)

    top5 = df.head(5)
    rows_html = ""
    for _, row in top5.iterrows():
        score = row.get("final_score", 0)
        color = "#c0392b" if score >= 0.8 else "#e67e22" if score >= 0.6 else "#27ae60"
        rows_html += f"""
        <tr>
          <td>{row.get('order_id', '-')}</td>
          <td>{row.get('user_id', '-')}</td>
          <td>${row.get('payment_amount', 0):,.0f}</td>
          <td style="color:{color};font-weight:bold">{score:.3f}</td>
          <td>{row.get('risk_explanation', '-')}</td>
          <td style="font-size:11px">{str(row.get('triggered_rules', '-'))[:80]}</td>
        </tr>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#333">
    <h2 style="color:#2c3e50">🔍 Fraud Detection Daily Report — {date_str}</h2>

    <table style="border-collapse:collapse;margin-bottom:20px">
      <tr>
        <td style="padding:10px 20px;background:#c0392b;color:white;border-radius:6px;text-align:center">
          <b style="font-size:24px">{len(critical)}</b><br>CRITICAL (&ge;0.8)
        </td>
        <td style="width:12px"></td>
        <td style="padding:10px 20px;background:#e67e22;color:white;border-radius:6px;text-align:center">
          <b style="font-size:24px">{len(high)}</b><br>HIGH (0.6–0.8)
        </td>
        <td style="width:12px"></td>
        <td style="padding:10px 20px;background:#7f8c8d;color:white;border-radius:6px;text-align:center">
          <b style="font-size:24px">{total}</b><br>Total Scored
        </td>
      </tr>
    </table>

    <h3>Top 5 Riskiest Transactions</h3>
    <table border="1" cellpadding="6" cellspacing="0"
           style="border-collapse:collapse;font-size:13px;width:100%">
      <tr style="background:#2c3e50;color:white">
        <th>Order ID</th><th>User ID</th><th>Amount</th>
        <th>Score</th><th>Risk Level</th><th>Triggered Rules</th>
      </tr>
      {rows_html}
    </table>

    <p style="margin-top:20px;color:#7f8c8d;font-size:12px">
      Full Excel report saved to Google Drive → "Daily fraud" folder.<br>
      Powered by Fiverr Fraud Detection Pipeline v11.1
    </p>
    </body></html>
    """

    summary_path = "/tmp/email_summary.html"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[info] Email summary saved: {summary_path}")
    return summary_path


# =================== MAIN DAILY PIPELINE ===================
def run_daily_pipeline(df):
    print("=" * 80)
    print("   Fiverr Fraud Detection - Daily Scoring (v11.1)")
    print("=" * 80)

    model = joblib.load(os.path.join(LOCAL_ARTIFACT_DIR, "fraud_model.joblib"))
    features = joblib.load(os.path.join(LOCAL_ARTIFACT_DIR, "fraud_features.joblib"))
    iso = joblib.load(os.path.join(LOCAL_ARTIFACT_DIR, "iforest_model.joblib"))
    weights = joblib.load(os.path.join(LOCAL_ARTIFACT_DIR, "rule_weights.joblib"))
    thresholds = joblib.load(os.path.join(LOCAL_ARTIFACT_DIR, "rule_thresholds.joblib"))
    encoders = joblib.load(os.path.join(LOCAL_ARTIFACT_DIR, "label_encoders.joblib"))
    try:
        base_alpha = float(open(os.path.join(LOCAL_ARTIFACT_DIR, "blend_alpha.txt")).read().strip())
    except Exception:
        base_alpha = 0.5

    df = ensure_datetime(df)
    df = safe_fillna(df)
    df = encode_categoricals(df, encoders, CAT_COLS)
    df = build_rule_columns(df, thresholds)
    merged_weights = merge_rule_weights(weights, CRITICAL_RULE_WEIGHTS, df)
    df = compute_manual_risk(df, merged_weights)

    try:
        iso_feats = [
            f for f in features
            if f not in ["manual_risk_score", "normalized_iforest", LABEL_COL, DATE_COL]
        ]
        df_iso = df.reindex(columns=iso_feats, fill_value=0).select_dtypes(include=np.number)
        df["iforest_score"] = -iso.score_samples(df_iso)
        df["normalized_iforest"] = (
            (df["iforest_score"] - df["iforest_score"].mean()) /
            (df["iforest_score"].std() + 1e-9)
        )
    except Exception:
        df["normalized_iforest"] = 0

    dmatrix = xgb.DMatrix(df.reindex(columns=features), missing=np.nan)
    df["fraud_score_raw"] = model.predict(dmatrix)

    conditions = [
        df.get("rule_has_blocked_clone", False),
        df.get("rule_status_pay", False),
        df.get("rule_buyer_seller_shared_clone", False),
    ]
    choices = [
        np.clip(base_alpha - 0.20, 0.7, 1.0),
        np.clip(base_alpha - 0.15, 0.7, 1.0),
        np.clip(base_alpha - 0.20, 0.7, 1.0),
    ]
    df["alpha_dynamic"] = np.select(conditions, choices, default=base_alpha)
    df["final_score"] = (
        df["alpha_dynamic"] * df["fraud_score_raw"] +
        (1 - df["alpha_dynamic"]) * df["manual_risk_score"]
    )

    df = explain_transaction_scores(df, model, features, merged_weights, thresholds)
    generate_summary_report(df)
    return df, base_alpha, encoders


# =================== EXPORT & UPLOAD ===================
if __name__ == "__main__":
    # Query BigQuery
    print(f"\n[info] Running BigQuery query for {yesterday}...")
    df = client.query(q).to_dataframe()
    print(f"[info] Loaded {len(df)} rows.")

    if df.empty:
        raise RuntimeError("No data returned from BigQuery.")

    results, alpha, cat_maps = run_daily_pipeline(df)

    # Fix timezone for Excel export
    for col in results.select_dtypes(include=["datetime64[ns, UTC]"]).columns:
        results[col] = results[col].dt.tz_localize(None)

    cols_to_keep = [
        "token", "order_id", "user_id", "seller_id", "payment_amount", "date_hour",
        "total_orders_buyer", "days_since_signup", "is_team", "is_new_user_7d", "is_fake_location",
        "is_country_mismatch", "unique_ips_last_24h", "user_txns_24h", "buyer_seller_shared_clone",
        "is_paypal_after_other_decline", "country_change_rate_24h", "buyer_payer_seen_in_seller",
        "has_blocked_clone", "buyer_count_clone", "seller_count_clone", "total_orders_buyer_seller",
        "order_status", "total_orders_seller", "seller_txns_30d", "seller_level",
        "seller_level_label", "is_fts", "valid_seller_country", "valid_seller_country_label",
        "seller_fraud_14d", "seller_fraud_30d", "seller_avg_order_amount_to_date",
        "seller_service_14d", "seller_message_count", "buyer_message_count",
        "messages_in_closest_order", "all_messages_in_all_orders",
        "total_messages_sent_by_buyer_per_conversation", "affiliates",
        "fraud_score_raw", "manual_risk_score", "final_score", "alpha_dynamic",
    ]
    explanation_cols = [
        "risk_explanation", "dominant_factor", "manual_risk_contribution",
        "ml_detailed_explanation", "ml_positive_signals", "ml_negative_signals",
        "triggered_rules", "rule_details", "threshold_violations",
    ]
    cols_to_use = [c for c in cols_to_keep + explanation_cols if c in results.columns]

    results = results.sort_values("final_score", ascending=False)
    deduped = (
        results.drop_duplicates(subset=["seller_id", "user_id"], keep="first")
        if {"seller_id", "user_id"} <= set(results.columns) else results.copy()
    )

    top_10 = deduped.head(10).copy()
    next_40 = deduped.iloc[10:50].copy()
    buyer_payer = deduped[deduped.get("buyer_payer_seen_in_seller", pd.Series(0)) == 1].copy()
    high_value = deduped[deduped.get("payment_amount", pd.Series(0)) >= 1000].copy()

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    excel_path = f"/tmp/risky_transactions_explained_{timestamp}.xlsx"

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        top_10[cols_to_use].to_excel(writer, sheet_name="Top_10_Explained", index=False)
        next_40[cols_to_use].to_excel(writer, sheet_name="Next_40_Explained", index=False)
        high_value[cols_to_use].to_excel(writer, sheet_name="High_Value_Explained", index=False)

    print(f"\n[info] Excel saved locally: {excel_path}")

    # Generate HTML email summary
    generate_email_html(results, yesterday)

    print("[✓] Daily pipeline completed successfully.")
