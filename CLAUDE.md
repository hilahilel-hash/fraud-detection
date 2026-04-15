# Fraud Detection Pipeline — Project Summary

## Overview
Automated fraud detection pipeline that trains weekly and scores daily, built from Fiverr's Colab notebooks.

## Repository
- GitHub: https://github.com/hilahilel-hash/fraud-detection
- Local: /Users/hila.hilel/my-first-repo

## Files
- `weekly.py` — Weekly training pipeline (XGBoost + IsolationForest + adaptive thresholds)
- `daily.py` — Daily scoring pipeline (loads artifacts, scores yesterday's data, exports Excel)
- `drive_utils.py` — Google Drive utilities (currently not used — Drive access blocked by Fiverr IT)
- `requirements.txt` — Python dependencies
- `.github/workflows/weekly.yml` — Weekly schedule + retry logic
- `.github/workflows/daily.yml` — Daily schedule + retry logic
- `.github/workflows/test_run.yml` — Manual test: runs weekly then daily

## Schedule
| Pipeline | First run | Retry (if failed) |
|---|---|---|
| Weekly | Sunday 08:00 Israel time | Sunday 13:00 |
| Daily | Every day 11:00 Israel time | Every day 14:00 |

## GitHub Secrets (configured)
- `GCP_CREDENTIALS` — Google user credentials (refresh token) for BigQuery
- `GMAIL_USER` — hila17hilel@gmail.com (sending account)
- `GMAIL_APP_PASSWORD` — App password for Gmail
- `SLACK_WEBHOOK_URL` — Slack webhook (configured but not active — IT blocks it)

## Notifications
- Email sent to hila.hilel@fiverr.com after every run
- Excel report attached to email
- Excel also saved as GitHub Actions artifact (30 days retention)

## BigQuery
- Project: fiverr-bq-payments-adhoc-prod
- Weekly table: fiverr-dwh-data-prod.dwh.dm_paypal_fraud_weekly
- Daily table: fiverr-dwh-data-prod.dwh.dm_paypal_fraud_daily
- Auth: authorized_user (hila.hilel@fiverr.com refresh token)

## Known Limitations
- Google Drive upload blocked (Fiverr IT blocks Drive API scope and IAM)
- Slack notifications blocked (Fiverr IT blocks incoming webhooks from external apps)
- Service Account creation blocked (no IAM permissions)

## Model
- XGBoost v11.1 with adaptive thresholds
- IsolationForest for anomaly detection
- Blended scoring: alpha * ML_score + (1-alpha) * rules_score
- Artifacts saved to /tmp/fraud_artifacts/ and uploaded via GitHub Actions
