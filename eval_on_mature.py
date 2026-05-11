# -*- coding: utf-8 -*-
"""
Re-evaluates daily baseline vs fixed_v2 on a MATURE dataset:
36K rows from 2026-01-20 to 2026-03-31, with 73 chargebacks pulled fresh from
`agg_chargebacks` (the daily table itself was stale, only had 5 labels for this period).

To avoid label leakage between weight calibration and evaluation, this script does a
clean TIME SPLIT:
  - calibration: 2026-01-20 to 2026-02-28 (older half)
  - evaluation:  2026-03-01 to 2026-03-31 (newer half)
"""
import warnings
warnings.filterwarnings("ignore")

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from weekly_compare import train_for_daily, LABEL_COL
from daily_compare import (
    score_baseline_daily, score_fixed_daily,
    audit_critical_rules, recalibrated_critical_weights,
    CRITICAL_RULE_WEIGHTS,
)

WEEKLY_CSV = "/tmp/fraud_data/weekly_data.csv"
MATURE_CSV = "/tmp/fraud_data/daily_mature.csv"


def top_k_summary(df, y, label):
    out = {"variant": label}
    out["auprc"] = float(average_precision_score(y, df["final_score"]))
    out["roc_auc"] = float(roc_auc_score(y, df["final_score"]))
    for k in [50, 100, 200, 500, 1000]:
        idx = np.argsort(-df["final_score"].to_numpy())[:k]
        out[f"top{k}_hits"] = int(y[idx].sum())
        out[f"top{k}_recall"] = float(y[idx].sum() / max(y.sum(), 1))
    ranks = (-df["final_score"].to_numpy()).argsort().argsort()[np.where(y == 1)[0]]
    out["mean_rank"] = float(ranks.mean())
    out["median_rank"] = float(np.median(ranks))
    return out


def main():
    print(f"[load] mature daily: {MATURE_CSV}")
    df_mature = pd.read_csv(MATURE_CSV, low_memory=False)
    df_mature["date_hour"] = pd.to_datetime(df_mature["date_hour"], errors="coerce", utc=True)
    print(f"  total mature rows: {len(df_mature)}")
    print(f"  total chargebacks: {int(df_mature[LABEL_COL].sum())}")

    cutoff = pd.Timestamp("2026-03-01", tz="UTC")
    df_calib = df_mature[df_mature["date_hour"] < cutoff].copy()
    df_eval = df_mature[df_mature["date_hour"] >= cutoff].copy()
    print(f"  calibration slice  (Jan 20 -> Feb 28): {len(df_calib)} rows, "
          f"{int(df_calib[LABEL_COL].sum())} chargebacks")
    print(f"  evaluation slice   (Mar  1 -> Mar 31): {len(df_eval)} rows, "
          f"{int(df_eval[LABEL_COL].sum())} chargebacks")

    print(f"\n[train] weekly fixed_v3 + tuned ...")
    df_w = pd.read_csv(WEEKLY_CSV, low_memory=False)
    artifacts = train_for_daily(df_w, mode="fixed_v3")
    print(f"  base_alpha = {artifacts['base_alpha']:.3f}")
    print(f"  weekly test AUPRC blend = {artifacts['test_auprc_blend']:.4f}")

    # Audit + recalibrate critical weights on the CALIBRATION slice only
    print(f"\n[audit] critical rules on calibration slice (Jan-Feb)...")
    df_calib_scored = score_fixed_daily(
        df_calib, artifacts, critical_weights=CRITICAL_RULE_WEIGHTS, combine_strength=1.0
    )
    audit = audit_critical_rules(df_calib_scored, list(CRITICAL_RULE_WEIGHTS.keys()))
    print(audit.to_string(index=False, float_format="%.4f"))

    new_w = recalibrated_critical_weights(audit, CRITICAL_RULE_WEIGHTS)
    print(f"\n[recalibrate] critical weights (hand -> calibrated on Jan-Feb mature labels):")
    for r in CRITICAL_RULE_WEIGHTS:
        old, new = CRITICAL_RULE_WEIGHTS[r], new_w.get(r, 0.0)
        arrow = "↑" if new > old + 0.5 else "↓" if new < old - 0.5 else "="
        print(f"  {r:<40} {old:>6.1f} -> {new:>6.2f}  {arrow}")

    # Score evaluation slice with all variants
    print(f"\n[score] scoring {len(df_eval)} eval rows...")
    df_b = score_baseline_daily(df_eval, artifacts)
    df_f1 = score_fixed_daily(df_eval, artifacts, critical_weights=CRITICAL_RULE_WEIGHTS, combine_strength=1.0)
    df_f2 = score_fixed_daily(df_eval, artifacts, critical_weights=new_w, combine_strength=1.0)
    df_f3 = score_fixed_daily(df_eval, artifacts, critical_weights=new_w, combine_strength=0.7)
    df_f4 = score_fixed_daily(df_eval, artifacts, critical_weights=new_w, combine_strength=0.5)

    y = df_b[LABEL_COL].astype(int).to_numpy()
    print(f"\n  eval set: {len(y)} rows, {int(y.sum())} chargebacks ({y.mean():.4%} fraud rate)")

    if y.sum() == 0:
        print("  No labels — can't evaluate.")
        return

    variants = [
        ("baseline (current daily.py)", df_b),
        ("fixed_v1 (hand weights, OR=1)", df_f1),
        ("fixed_v2 (calibrated, OR=1)  <- RECOMMENDED", df_f2),
        ("fixed_v3 (calibrated, OR=0.7)", df_f3),
        ("fixed_v4 (calibrated, OR=0.5)", df_f4),
    ]
    rows = [top_k_summary(df, y, name) for name, df in variants]
    df_summary = pd.DataFrame(rows)

    print(f"\n{'=' * 110}")
    print(f"  HELD-OUT EVALUATION (calibration on Jan-Feb labels, test on Mar labels)")
    print(f"{'=' * 110}")
    cols = ["variant", "auprc", "roc_auc", "top50_hits", "top100_hits",
            "top200_hits", "top500_hits", "top1000_hits", "mean_rank", "median_rank"]
    print(df_summary[cols].to_string(index=False, float_format="%.4f"))

    print(f"\n  TOP-K RECALL (fraction of {int(y.sum())} chargebacks caught in top K):")
    cols_recall = ["variant", "top50_recall", "top100_recall", "top200_recall", "top500_recall", "top1000_recall"]
    print(df_summary[cols_recall].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
