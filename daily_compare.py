# -*- coding: utf-8 -*-
"""
Local comparison runner for the DAILY pipeline.
Trains weekly once (fixed_v3 + tuned) and uses its artifacts for both
baseline and fixed daily scorers, then compares predictions on cached daily data.

Fixes applied in FIXED daily scorer:
  D1. manual_risk_score normalized with TRAIN min/max (not the day's own min/max).
      Makes scores comparable across days; the 0.8 CRITICAL threshold becomes meaningful.
  D2. normalized_iforest uses TRAIN mean/std (not the day's own).
      Same cross-day consistency story.
  D3. alpha_dynamic bug fixed.
      Old code clipped to [0.7, 1.0] which INCREASED ML weight when critical rules fired
      (backwards from intent). New design drops dynamic-alpha entirely and uses a
      separate critical_score that boosts the final score via noisy-OR.
  D4. merge_rule_weights no longer tries to estimate lift on live data.
      Daily labels are ~all-zero (chargebacks haven't matured), so the old lift was
      ~0 -> weights clipped to base_w*0.5 -> critical rules under-weighted by half.
      New: use CRITICAL_RULE_WEIGHTS as static constants.
  D5. Critical rules combined via separate critical_score, not folded into manual_risk_score.
      final = noisy_or(blended, critical_score) where:
        blended = alpha * ML + (1-alpha) * manual_risk_score
        critical_score = sum(fired critical weights) / sum(all critical weights)
        noisy_or(a, b) = 1 - (1-a)(1-b) = a + b - a*b
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score, average_precision_score

from weekly_compare import (
    ensure_datetime, safe_fillna, train_for_daily,
    LABEL_COL, DATE_COL, CAT_COLS, HIGH_RISK_COUNTRIES,
)

WEEKLY_CSV = "/tmp/fraud_data/weekly_data.csv"
DAILY_HISTORY_CSV = "/tmp/fraud_data/daily_data.csv"        # full 111-day history (for audit)
DAILY_TODAY_CSV = "/tmp/fraud_data/daily_today.csv"          # production-query slice (for scoring)

# Hand-picked weights (preserved from daily.py)
CRITICAL_RULE_WEIGHTS = {
    "rule_buyer_payer_seen_in_seller": 8.0,
    "rule_buyer_seller_shared_clone": 25.0,
    "rule_has_blocked_clone": 25.0,
    "rule_status_pay": 20.0,
    "rule_seller_count_clone": 8.0,
    "rule_massage_activity_all": 20.0,
    "rule_massage_activity": 20.0,
}

# Rules that are daily-only (their underlying columns don't exist in weekly data,
# so the weekly model never saw them).
DAILY_ONLY_CRITICAL = {
    "rule_buyer_seller_shared_clone",
    "rule_has_blocked_clone",
    "rule_status_pay",
    "rule_seller_count_clone",
    "rule_massage_activity",
    "rule_massage_activity_all",
}


# =================== SHARED HELPERS ===================
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


def build_rule_columns_daily(df, thresholds):
    """Mirrors daily.py.build_rule_columns. Includes daily-only critical rules."""
    df = df.copy()

    def num(c):
        return pd.to_numeric(df.get(c, 0), errors="coerce")

    df["rule_has_blocked_clone"] = (num("has_blocked_clone") == 1)
    df["rule_status_pay"] = (num("payment_amount") >= 600) & (df.get("order_status", "") == "delivered")
    df["rule_buyer_payer_seen_in_seller"] = (num("buyer_payer_seen_in_seller") == 1)
    df["rule_buyer_seller_shared_clone"] = (num("buyer_seller_shared_clone") == 1)
    df["rule_seller_count_clone"] = (num("seller_count_clone") > 0)
    df["rule_massage_activity"] = (num("messages_in_closest_order") < 7) & (df.get("messages_in_closest_order", pd.Series([np.nan]*len(df))).notna())

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


def _accumulate_weighted(df, weights):
    """Sum of (rule_value * weight) per row, treating bool/0-1 numeric as boolean fired."""
    raw = pd.Series(0.0, index=df.index)
    for c, w in weights.items():
        if c not in df.columns:
            continue
        col = df[c]
        if col.isna().all():
            continue
        col_filled = col.fillna(False) if pd.api.types.is_bool_dtype(col) else col.fillna(0)
        if pd.api.types.is_bool_dtype(col_filled):
            contrib = col_filled.astype(bool).astype(int) * w
        elif col_filled.dropna().isin([0, 1, True, False]).all():
            contrib = col_filled.astype(bool).astype(int) * w
        elif col_filled.dropna().between(0, 1).all():
            contrib = col_filled.astype(float) * w
        else:
            contrib = col_filled.astype(float).clip(0, 1) * w
        raw = raw + contrib
    return raw


# =================== BASELINE DAILY (mirrors current daily.py) ===================
def merge_rule_weights_baseline(weekly_weights, critical_rules, df):
    """Faithful reproduction of daily.py.merge_rule_weights (lift-on-live bug included)."""
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
            merged[rule] = float(np.clip((lift ** 1.5) * base_w, base_w * 0.5, base_w * 2.5))
        else:
            merged[rule] = base_w
    return merged


def compute_manual_risk_baseline(df, weights):
    """Daily.py behavior: min/max per-day (LEAKY across days)."""
    df = df.copy()
    df["manual_risk_raw"] = _accumulate_weighted(df, weights)
    mn, mx = df["manual_risk_raw"].min(), df["manual_risk_raw"].max()
    if mx != mn:
        df["manual_risk_score"] = (df["manual_risk_raw"] - mn) / (mx - mn + 1e-9)
    else:
        df["manual_risk_score"] = 0.0
    df.drop(columns=["manual_risk_raw"], inplace=True)
    return df


def score_baseline_daily(df_day, artifacts):
    """Reproduces daily.py.run_daily_pipeline as-is."""
    df = df_day.copy()
    df = ensure_datetime(df)
    df = safe_fillna(df)
    df = encode_categoricals(df, artifacts["encoders"], CAT_COLS)
    df = build_rule_columns_daily(df, artifacts["thresholds"])

    merged_weights = merge_rule_weights_baseline(
        artifacts["weights"], CRITICAL_RULE_WEIGHTS, df
    )
    df = compute_manual_risk_baseline(df, merged_weights)

    # iforest: original code re-normalizes with per-day mean/std
    iso = artifacts["iso"]
    num_cols = artifacts["iso_num_cols"]
    df_iso = (
        df.reindex(columns=num_cols, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    df["iforest_score"] = -iso.score_samples(df_iso)
    df["normalized_iforest"] = (
        (df["iforest_score"] - df["iforest_score"].mean()) /
        (df["iforest_score"].std() + 1e-9)
    )

    features = artifacts["feature_columns"]
    dmat = xgb.DMatrix(df.reindex(columns=features), missing=np.nan)
    df["fraud_score_raw"] = artifacts["model"].predict(dmat)

    # The notorious alpha_dynamic. Reproduces the bug: clip to [0.7, 1.0] which floors at 0.7.
    base_alpha = artifacts["base_alpha"]
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
    return df


# =================== FIXED DAILY ===================
def compute_manual_risk_fixed(df, weights, train_mn, train_mx):
    """D1: normalize with weekly TRAIN min/max for cross-day consistency."""
    df = df.copy()
    df["manual_risk_raw"] = _accumulate_weighted(df, weights)
    denom = (train_mx - train_mn) + 1e-9
    df["manual_risk_score"] = ((df["manual_risk_raw"] - train_mn) / denom).clip(0, 1)
    df.drop(columns=["manual_risk_raw"], inplace=True)
    return df


def compute_critical_score(df, critical_weights):
    """D5: critical_score is a separate signal in [0, 1].
    Sum of (fired weight) / sum of all weights, clipped to [0, 1].
    All critical rules firing -> 1.0. None firing -> 0.0."""
    df = df.copy()
    max_w = sum(critical_weights.values())
    raw = _accumulate_weighted(df, critical_weights)
    df["critical_score"] = (raw / max(max_w, 1e-9)).clip(0, 1)
    df["critical_score_raw"] = raw
    return df


def noisy_or(a, b):
    """final = 1 - (1-a)(1-b) = a + b - a*b. Stays in [0,1] if a,b in [0,1]."""
    return a + b - a * b


def soft_noisy_or(a, b, strength=1.0):
    """Attenuated noisy-OR. strength=1 -> full noisy-OR. strength=0 -> no boost.
    Useful when b's specificity is uncertain; reduces over-boost from weak signals.
    """
    return a + strength * b * (1 - a)


def score_fixed_daily(df_day, artifacts, critical_weights=None, combine_strength=1.0):
    """All D1-D5 fixes applied.

    critical_weights: dict of {rule: weight}. If None, uses CRITICAL_RULE_WEIGHTS as-is.
    combine_strength: noisy-OR strength in [0, 1]. 1.0 = full noisy-OR boost.
    """
    df = df_day.copy()
    df = ensure_datetime(df)
    df = safe_fillna(df)
    df = encode_categoricals(df, artifacts["encoders"], CAT_COLS)
    df = build_rule_columns_daily(df, artifacts["thresholds"])

    # D4: regular weights stay as trained on weekly; critical weights stay static.
    regular_weights = artifacts["weights"]  # weekly-trained, doesn't include daily-only rules
    if critical_weights is None:
        critical_weights = CRITICAL_RULE_WEIGHTS

    df = compute_manual_risk_fixed(
        df, regular_weights,
        artifacts["manual_risk_mn"], artifacts["manual_risk_mx"]
    )
    df = compute_critical_score(df, critical_weights)

    # D2: iforest normalization with TRAIN mean/std
    iso = artifacts["iso"]
    num_cols = artifacts["iso_num_cols"]
    df_iso = (
        df.reindex(columns=num_cols, fill_value=0)
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    df["iforest_score"] = -iso.score_samples(df_iso)
    df["normalized_iforest"] = (
        (df["iforest_score"] - artifacts["iforest_mean"]) /
        (artifacts["iforest_std"] + 1e-9)
    )

    features = artifacts["feature_columns"]
    dmat = xgb.DMatrix(df.reindex(columns=features), missing=np.nan)
    df["fraud_score_raw"] = artifacts["model"].predict(dmat)

    # D3: no dynamic alpha. Static blended = base_alpha * ML + (1-base_alpha) * manual_risk
    base_alpha = artifacts["base_alpha"]
    blended = base_alpha * df["fraud_score_raw"] + (1 - base_alpha) * df["manual_risk_score"]

    # Combine with critical via (soft) noisy-OR
    df["blended_score"] = blended
    df["final_score"] = soft_noisy_or(blended, df["critical_score"], strength=combine_strength)
    df["alpha_dynamic"] = base_alpha  # for compatibility / inspection
    return df


# =================== CRITICAL RULE AUDIT ===================
def audit_critical_rules(df, critical_rule_names, label_col=LABEL_COL, prior=100):
    """Per critical rule: fire rate, CB rate when fired, raw lift, Bayesian-smoothed lift.

    Smoothing: assume `prior` pseudo-observations at base rate. This stabilizes lift
    estimates for rules that fire on few rows.
    """
    if label_col not in df.columns:
        return pd.DataFrame()
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    base = y.mean()
    rows = []
    for rule in critical_rule_names:
        if rule not in df.columns:
            rows.append({
                "rule": rule, "fire_rate": np.nan, "n_fires": 0, "cb_fires": 0,
                "rate_fires": np.nan, "lift_raw": np.nan, "lift_smoothed": np.nan,
            })
            continue
        col = df[rule]
        fires = col.fillna(False).astype(bool) if pd.api.types.is_bool_dtype(col) else (col.fillna(0) > 0)
        n_fires = int(fires.sum())
        cb_fires = int(y[fires].sum())
        if n_fires == 0:
            rate_fires, lift_raw = np.nan, np.nan
        else:
            rate_fires = cb_fires / n_fires
            lift_raw = rate_fires / base if base > 0 else np.nan
        rate_smoothed = (cb_fires + base * prior) / (n_fires + prior)
        lift_smoothed = rate_smoothed / base if base > 0 else np.nan
        rows.append({
            "rule": rule,
            "fire_rate": fires.mean(),
            "n_fires": n_fires,
            "cb_fires": cb_fires,
            "rate_fires": rate_fires,
            "lift_raw": lift_raw,
            "lift_smoothed": lift_smoothed,
        })
    return pd.DataFrame(rows).sort_values("lift_smoothed", ascending=False, na_position="last")


def recalibrated_critical_weights(audit_df, hand_weights, fire_rate_cap=0.20):
    """Nuanced recalibration that respects sample size.

    Logic per rule:
      - No fires observed -> keep hand weight (no info to update from).
      - Few fires (< 30) -> stay close to hand weight (insufficient evidence either way).
      - Strong evidence (lift >= 3, n >= 30) -> hand_weight * sqrt(lift), capped at 2.5x.
      - Strong negative evidence (lift < 0.5, n >= 500) -> hand_weight * 0.1 (rule looks useless).
      - Mild range -> clipped sqrt(lift) keeps weight within [0.5x, 1.5x] of hand.
      - Always attenuate by fire_rate_cap/fire_rate if fire_rate > cap (over-firing penalty).
    """
    new_weights = {}
    for _, row in audit_df.iterrows():
        rule = row["rule"]
        hand_w = hand_weights.get(rule, 1.0)
        fire = row["fire_rate"]
        lift = row["lift_smoothed"]
        n_fires = row["n_fires"]

        if n_fires == 0 or pd.isna(lift):
            new_weights[rule] = hand_w
            continue

        if lift < 0.5 and n_fires >= 500:
            mult = 0.10
        elif lift >= 3.0 and n_fires >= 30:
            mult = float(min(np.sqrt(lift), 2.5))
        else:
            mult = float(np.clip(np.sqrt(max(lift, 0.5)), 0.5, 1.5))

        new_w = hand_w * mult
        if pd.notna(fire) and fire > fire_rate_cap:
            new_w *= fire_rate_cap / fire

        new_weights[rule] = float(new_w)
    return new_weights


# =================== EVALUATION ===================
def evaluate(df_b, df_f, label_name="has_chargeback"):
    """Print comparison of baseline vs fixed daily scoring on the same data."""
    y = df_b[label_name].astype(int).to_numpy() if label_name in df_b.columns else None

    print("\n" + "=" * 80)
    print("  SCORE DISTRIBUTION COMPARISON")
    print("=" * 80)
    print(f"{'metric':<30}  {'baseline':>10}  {'fixed':>10}")
    for name, ser_b, ser_f in [
        ("final_score mean", df_b["final_score"].mean(), df_f["final_score"].mean()),
        ("final_score std", df_b["final_score"].std(), df_f["final_score"].std()),
        ("final_score p50", df_b["final_score"].median(), df_f["final_score"].median()),
        ("final_score p90", df_b["final_score"].quantile(0.9), df_f["final_score"].quantile(0.9)),
        ("final_score p99", df_b["final_score"].quantile(0.99), df_f["final_score"].quantile(0.99)),
        ("final_score max", df_b["final_score"].max(), df_f["final_score"].max()),
        ("frac >= 0.6", (df_b["final_score"] >= 0.6).mean(), (df_f["final_score"] >= 0.6).mean()),
        ("frac >= 0.8", (df_b["final_score"] >= 0.8).mean(), (df_f["final_score"] >= 0.8).mean()),
    ]:
        print(f"{name:<30}  {ser_b:>10.4f}  {ser_f:>10.4f}")

    if y is None or y.sum() == 0:
        print("\n[info] No chargebacks in this slice — skipping AUPRC/ranking analysis.")
        return

    print(f"\n  Chargebacks in slice: {int(y.sum())} / {len(y)} ({y.mean():.4%})")

    print("\n" + "=" * 80)
    print("  LABEL-BASED METRICS (AUPRC + Top-K)")
    print("=" * 80)
    print(f"{'metric':<30}  {'baseline':>10}  {'fixed':>10}  {'delta':>10}")

    auprc_b = average_precision_score(y, df_b["final_score"])
    auprc_f = average_precision_score(y, df_f["final_score"])
    rocauc_b = roc_auc_score(y, df_b["final_score"])
    rocauc_f = roc_auc_score(y, df_f["final_score"])
    print(f"{'AUPRC (final_score)':<30}  {auprc_b:>10.4f}  {auprc_f:>10.4f}  {auprc_f-auprc_b:>+10.4f}")
    print(f"{'ROC-AUC (final_score)':<30}  {rocauc_b:>10.4f}  {rocauc_f:>10.4f}  {rocauc_f-rocauc_b:>+10.4f}")

    # Top-K hit rate: of the top-K highest-scored rows, how many are true chargebacks?
    for k in [50, 100, 200, 500]:
        if k > len(y):
            continue
        idx_b = np.argsort(-df_b["final_score"].to_numpy())[:k]
        idx_f = np.argsort(-df_f["final_score"].to_numpy())[:k]
        hits_b = y[idx_b].sum()
        hits_f = y[idx_f].sum()
        recall_b = hits_b / y.sum()
        recall_f = hits_f / y.sum()
        print(f"{'top-' + str(k) + ' recall':<30}  {recall_b:>10.4f}  {recall_f:>10.4f}  {recall_f-recall_b:>+10.4f}")

    print("\n" + "=" * 80)
    print("  RANK OF KNOWN CHARGEBACKS (lower = better)")
    print("=" * 80)
    pos_idx = np.where(y == 1)[0]
    ranks_b = (-df_b["final_score"].to_numpy()).argsort().argsort()[pos_idx]
    ranks_f = (-df_f["final_score"].to_numpy()).argsort().argsort()[pos_idx]
    print(f"{'metric':<30}  {'baseline':>10}  {'fixed':>10}  {'delta':>10}")
    print(f"{'mean rank':<30}  {ranks_b.mean():>10.1f}  {ranks_f.mean():>10.1f}  {ranks_f.mean()-ranks_b.mean():>+10.1f}")
    print(f"{'median rank':<30}  {np.median(ranks_b):>10.1f}  {np.median(ranks_f):>10.1f}  {np.median(ranks_f)-np.median(ranks_b):>+10.1f}")
    print(f"{'max rank (worst miss)':<30}  {ranks_b.max():>10.1f}  {ranks_f.max():>10.1f}  {ranks_f.max()-ranks_b.max():>+10.1f}")

    print("\n" + "=" * 80)
    print("  CRITICAL RULE TRIGGER COVERAGE")
    print("=" * 80)
    if "critical_score" in df_f.columns:
        triggered = (df_f["critical_score"] > 0)
        print(f"  Rows with any critical rule firing: {int(triggered.sum())} ({triggered.mean():.4%})")
        if triggered.sum() > 0:
            cb_among_triggered = y[triggered].sum()
            print(f"  Chargebacks among those:            {int(cb_among_triggered)} "
                  f"({cb_among_triggered / max(triggered.sum(), 1):.4%} fraud rate)")
            print(f"  Chargebacks NOT covered by critical: {int(y.sum() - cb_among_triggered)}")


# =================== MAIN ===================
def _short_metrics(df, y, label):
    auprc = average_precision_score(y, df["final_score"])
    rocauc = roc_auc_score(y, df["final_score"])
    out = {"variant": label, "auprc": auprc, "roc_auc": rocauc}
    for k in [50, 100, 200, 500]:
        idx = np.argsort(-df["final_score"].to_numpy())[:k]
        out[f"top{k}_hits"] = int(y[idx].sum())
        out[f"top{k}_recall"] = float(y[idx].sum() / max(y.sum(), 1))
    ranks = (-df["final_score"].to_numpy()).argsort().argsort()[np.where(y == 1)[0]]
    out["mean_rank"] = float(ranks.mean())
    out["median_rank"] = float(np.median(ranks))
    return out


def save_excel_comparison(df_d, df_b, df_f1, df_f2, df_f3, df_f4, out_path):
    """Write an Excel file in the EXACT layout of the current daily.py output,
    once per variant (baseline + recommended fixed_v2), so the user can compare row-by-row.

    Sheets:
      Current_Top_10        Top 10 by the *current* daily.py score (what gets emailed today)
      Current_Next_40       Rows 11-50 by current score
      Current_High_Value    Rows with payment_amount >= 1000, by current score
      Recommended_Top_10    Top 10 by the recommended fixed_v2 score
      Recommended_Next_40   Rows 11-50 by recommended score
      Recommended_High_Value Rows with payment_amount >= 1000, by recommended score
      All_Chargebacks       All 31 chargebacks in this slice, both scores, both ranks
      Summary               Variant-level metrics in one table
    """
    n = len(df_d)
    label = df_b[LABEL_COL].astype(int).to_numpy() if LABEL_COL in df_b.columns else np.zeros(n, int)

    # Exact column list from daily.py.cols_to_keep
    base_cols = [
        "token", "order_id", "user_id", "seller_id", "payment_amount", "date_hour",
        "total_orders_buyer", "days_since_signup", "is_team", "is_new_user_7d", "is_fake_location",
        "is_country_mismatch", "unique_ips_last_24h", "user_txns_24h", "buyer_seller_shared_clone",
        "is_paypal_after_other_decline", "country_change_rate_24h", "buyer_payer_seen_in_seller",
        "has_blocked_clone", "buyer_count_clone", "seller_count_clone", "total_orders_buyer_seller",
        "order_status", "total_orders_seller", "seller_txns_30d", "seller_level",
        "is_fts", "valid_seller_country",
        "seller_fraud_14d", "seller_fraud_30d", "seller_avg_order_amount_to_date",
        "seller_service_14d",
        "messages_in_closest_order", "all_messages_in_all_orders",
    ]
    base_cols = [c for c in base_cols if c in df_d.columns]

    crit_cols = [r for r in CRITICAL_RULE_WEIGHTS if r in df_f2.columns]
    triggered_text = df_f2[crit_cols].fillna(False).astype(bool).apply(
        lambda r: ", ".join([c.replace("rule_", "") for c in crit_cols if r[c]]) or "none",
        axis=1,
    )

    def make_view(df_score, score_label, include_both_scores=True):
        view = df_d[base_cols].copy()
        view["has_chargeback"] = label

        # Score columns - sort key first for visibility
        view[f"{score_label}_final_score"] = df_score["final_score"].values
        if include_both_scores:
            view["baseline_score"] = df_b["final_score"].values
            view["recommended_score_v2"] = df_f2["final_score"].values

        # Subscore breakdown for the variant being shown
        if "fraud_score_raw" in df_score.columns:
            view[f"{score_label}_ml_score"] = df_score["fraud_score_raw"].values
        if "manual_risk_score" in df_score.columns:
            view[f"{score_label}_manual_risk"] = df_score["manual_risk_score"].values
        if "critical_score" in df_score.columns:
            view[f"{score_label}_critical_score"] = df_score["critical_score"].values
        if "blended_score" in df_score.columns:
            view[f"{score_label}_blended_pre_critical"] = df_score["blended_score"].values
        if "alpha_dynamic" in df_score.columns:
            view[f"{score_label}_alpha"] = df_score["alpha_dynamic"].values

        view["triggered_critical_rules"] = triggered_text.values

        # tz-strip for excel
        for c in view.select_dtypes(include=["datetime64[ns, UTC]"]).columns:
            view[c] = view[c].dt.tz_localize(None)
        return view

    # Build the two main views
    view_current = make_view(df_b, "current")
    view_recommended = make_view(df_f2, "recommended")

    # Dedup by (seller_id, user_id), keep highest score per pair — matches daily.py behavior
    def dedup_sort(view, sort_col):
        v = view.sort_values(sort_col, ascending=False)
        if {"seller_id", "user_id"} <= set(v.columns):
            v = v.drop_duplicates(subset=["seller_id", "user_id"], keep="first")
        return v

    current_sorted = dedup_sort(view_current, "current_final_score")
    recommended_sorted = dedup_sort(view_recommended, "recommended_final_score")

    current_top10 = current_sorted.head(10)
    current_next40 = current_sorted.iloc[10:50]
    current_highval = current_sorted[current_sorted["payment_amount"] >= 1000].head(100)

    recommended_top10 = recommended_sorted.head(10)
    recommended_next40 = recommended_sorted.iloc[10:50]
    recommended_highval = recommended_sorted[recommended_sorted["payment_amount"] >= 1000].head(100)

    # Chargebacks sheet — both scores, both ranks
    cb_idx = np.where(label == 1)[0]
    if len(cb_idx) > 0:
        rank_baseline = (-df_b["final_score"].to_numpy()).argsort().argsort()  # 0-based
        rank_v2 = (-df_f2["final_score"].to_numpy()).argsort().argsort()
        cb = df_d.iloc[cb_idx][base_cols].copy()
        cb["has_chargeback"] = 1
        cb["baseline_score"] = df_b["final_score"].iloc[cb_idx].values
        cb["recommended_score_v2"] = df_f2["final_score"].iloc[cb_idx].values
        cb["rank_baseline"] = (rank_baseline[cb_idx] + 1).astype(int)
        cb["rank_recommended"] = (rank_v2[cb_idx] + 1).astype(int)
        cb["rank_change"] = cb["rank_recommended"] - cb["rank_baseline"]
        cb["triggered_critical_rules"] = triggered_text.iloc[cb_idx].values
        for c in cb.select_dtypes(include=["datetime64[ns, UTC]"]).columns:
            cb[c] = cb[c].dt.tz_localize(None)
        cb = cb.sort_values("rank_recommended")
    else:
        cb = pd.DataFrame()

    # Summary sheet
    variants = [
        ("baseline (current daily.py)", df_b),
        ("fixed_v1 (D1-D5, hand weights)", df_f1),
        ("fixed_v2 (D1-D5 + calibrated) ← RECOMMENDED", df_f2),
        ("fixed_v3 (calibrated, OR 0.7)", df_f3),
        ("fixed_v4 (calibrated, OR 0.5)", df_f4),
    ]
    summary_rows = []
    for name, df in variants:
        row = {"variant": name}
        s = df["final_score"]
        row["mean"] = float(s.mean())
        row["p99"] = float(s.quantile(0.99))
        row["max"] = float(s.max())
        row["frac_ge_0.6"] = float((s >= 0.6).mean())
        row["frac_ge_0.8"] = float((s >= 0.8).mean())
        if label.sum() > 0:
            row["auprc"] = float(average_precision_score(label, s))
            row["roc_auc"] = float(roc_auc_score(label, s))
            for k in [50, 100, 200, 500]:
                idx = np.argsort(-s.to_numpy())[:k]
                row[f"top{k}_chargebacks_caught"] = int(label[idx].sum())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        current_top10.to_excel(writer, sheet_name="Current_Top_10", index=False)
        current_next40.to_excel(writer, sheet_name="Current_Next_40", index=False)
        current_highval.to_excel(writer, sheet_name="Current_High_Value", index=False)
        recommended_top10.to_excel(writer, sheet_name="Recommended_Top_10", index=False)
        recommended_next40.to_excel(writer, sheet_name="Recommended_Next_40", index=False)
        recommended_highval.to_excel(writer, sheet_name="Recommended_High_Value", index=False)
        if not cb.empty:
            cb.to_excel(writer, sheet_name="All_Chargebacks", index=False)

    return out_path


def main():
    print(f"[load] weekly:        {WEEKLY_CSV}")
    df_w = pd.read_csv(WEEKLY_CSV, low_memory=False)
    print(f"  shape: {df_w.shape}")

    print(f"[load] daily history: {DAILY_HISTORY_CSV}  (for audit / weight calibration)")
    df_d_all = pd.read_csv(DAILY_HISTORY_CSV, low_memory=False)
    print(f"  shape: {df_d_all.shape}")

    print(f"[load] daily TODAY:   {DAILY_TODAY_CSV}  (production SQL slice, what daily.py scores today)")
    df_d = pd.read_csv(DAILY_TODAY_CSV, low_memory=False)
    print(f"  shape: {df_d.shape}")
    df_d["date_hour"] = pd.to_datetime(df_d["date_hour"], errors="coerce", utc=True)
    latest_day = df_d["date_hour"].dt.date.max()

    print("\n[train] running weekly fixed_v3 + tuned to produce artifacts...")
    artifacts = train_for_daily(df_w, mode="fixed_v3")
    print(f"  base_alpha          = {artifacts['base_alpha']:.3f}")
    print(f"  manual_risk_mn / mx = {artifacts['manual_risk_mn']:.4f} / {artifacts['manual_risk_mx']:.4f}")
    print(f"  iforest_mean / std  = {artifacts['iforest_mean']:.4f} / {artifacts['iforest_std']:.4f}")
    print(f"  feature_columns: {len(artifacts['feature_columns'])}")
    print(f"  (weekly test AUPRC blend: {artifacts['test_auprc_blend']:.4f})")

    # Audit + recalibration of critical weights on FULL daily history (good signal).
    # This step is what a weekly retraining job would do; we save the resulting weights
    # and use them as static constants during daily scoring.
    print("\n[audit] critical rule fire rate + lift on FULL history "
          f"(scoring all {len(df_d_all)} rows for audit only)...")
    df_for_audit = score_fixed_daily(
        df_d_all, artifacts, critical_weights=CRITICAL_RULE_WEIGHTS, combine_strength=1.0
    )
    audit = audit_critical_rules(df_for_audit, list(CRITICAL_RULE_WEIGHTS.keys()))
    print(audit.to_string(index=False, float_format="%.4f"))

    new_w = recalibrated_critical_weights(audit, CRITICAL_RULE_WEIGHTS)
    print("\n[recalibrate] critical weights (hand → empirical, from full history):")
    for r in CRITICAL_RULE_WEIGHTS:
        old, new = CRITICAL_RULE_WEIGHTS[r], new_w.get(r, 0.0)
        arrow = "↑" if new > old + 0.5 else "↓" if new < old - 0.5 else "="
        print(f"  {r:<40} {old:>6.1f} → {new:>6.2f}  {arrow}")

    # ---- Score today's slice with all variants ----
    print(f"\n[score] scoring {len(df_d)} rows for date={latest_day} ...")
    df_b = score_baseline_daily(df_d, artifacts)
    df_f1 = score_fixed_daily(df_d, artifacts, critical_weights=CRITICAL_RULE_WEIGHTS, combine_strength=1.0)
    df_f2 = score_fixed_daily(df_d, artifacts, critical_weights=new_w, combine_strength=1.0)
    df_f3 = score_fixed_daily(df_d, artifacts, critical_weights=new_w, combine_strength=0.7)
    df_f4 = score_fixed_daily(df_d, artifacts, critical_weights=new_w, combine_strength=0.5)

    # Side-by-side comparison
    y = df_b[LABEL_COL].astype(int).to_numpy() if LABEL_COL in df_b.columns else np.zeros(len(df_b), int)
    print("\n" + "=" * 100)
    print(f"  SIDE-BY-SIDE COMPARISON ({int(y.sum())} chargebacks / {len(df_b)} rows in today's slice)")
    print("=" * 100)
    variants = [
        ("baseline", df_b),
        ("fixed_v1 (hand, full OR)", df_f1),
        ("fixed_v2 (calib, full OR)", df_f2),
        ("fixed_v3 (calib, OR×0.7)", df_f3),
        ("fixed_v4 (calib, OR×0.5)", df_f4),
    ]

    if y.sum() == 0:
        print("  No matured labels in today's slice (expected — chargebacks mature ~30+ days later).")
        print("  Skipping label-based metrics; score distributions and Excel still produced.")
    else:
        rows = [_short_metrics(df, y, name) for name, df in variants]
        df_out = pd.DataFrame(rows)
        cols = ["variant", "auprc", "roc_auc", "top50_hits", "top100_hits", "top200_hits", "top500_hits", "mean_rank", "median_rank"]
        print(df_out[cols].to_string(index=False, float_format="%.4f"))

    # Score distribution summary
    print("\n  SCORE DISTRIBUTION (max / p99 / mean):")
    for name, df in variants:
        s = df["final_score"]
        print(f"    {name:<32}  max={s.max():.3f}  p99={s.quantile(0.99):.3f}  mean={s.mean():.3f}  frac>=0.6={(s>=0.6).mean():.4%}")

    # Write Excel comparison file
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y-%m-%d_%H-%M")
    out_path = f"/tmp/daily_compare_results_{timestamp}.xlsx"
    print(f"\n[export] writing Excel comparison to {out_path} ...")
    save_excel_comparison(df_d, df_b, df_f1, df_f2, df_f3, df_f4, out_path)
    print(f"[export] done. open it with: open '{out_path}'")


if __name__ == "__main__":
    main()
