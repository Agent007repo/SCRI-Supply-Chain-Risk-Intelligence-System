# SCRI Model Report

## Project Type

Applied ML risk analytics notebook with a production roadmap.

## Reviewer Summary

SCRI is designed to detect and forecast global supply-chain stress using public macroeconomic and market signals. The project combines a composite stress index, anomaly detection, and supervised multi-horizon forecasting.

## Core Modeling Components

- Supply Chain Stress Index built from market and macro signals.
- LSTM autoencoder for anomaly detection against calm-period behavior.
- LightGBM classifiers for 7-day, 14-day, and 30-day risk horizons.
- Ensemble alert tiering for operational interpretation.

## Recruiter Signal

This project is strong evidence for applied machine learning, risk analytics, time-series reasoning, and business translation. It is especially relevant for data science, risk analytics, supply-chain intelligence, and AI/data product roles.

## Technical Reviewer Signal

The notebook demonstrates model design, feature engineering, backtesting against historical shocks, and business-facing interpretation. The next step is to move reusable logic into Python modules and add a CLI or dashboard.

## Known Limitations

- Public macro and market signals are proxies, not direct logistics telemetry.
- Forecast labels depend on the constructed stress index.
- Operational deployment would require stronger data validation, monitoring, retraining, and alert calibration.
- Results should be treated as analytical research, not live operational advice.

## Next Improvements

- Move data loading and feature engineering into `src/` modules.
- Add saved plots under an `outputs/` directory.
- Add a quick-run script for reviewers.
- Add model monitoring and calibration notes.
