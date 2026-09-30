# SCRI: Supply Chain Risk Intelligence System

<div align="center">

**Independent research notebook for macro-risk forecasting and anomaly detection**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen?style=flat-square)](https://lightgbm.readthedocs.io/)
[![FRED API](https://img.shields.io/badge/Data-FRED%20API-blue?style=flat-square)](https://fred.stlouisfed.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

*Historical analysis with a reported January 2023–April 2026 test period; results are not independently rerun or validated for operational use.*

</div>

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [The Solution](#2-the-solution)
3. [System Architecture](#3-system-architecture)
4. [Data Sources](#4-data-sources)
5. [The Four-Layer Pipeline](#5-the-four-layer-pipeline)
   - [Layer 1: Supply Chain Stress Index (SCSI)](#layer-1-supply-chain-stress-index-scsi)
   - [Layer 2: LSTM Autoencoder - Anomaly Detection](#layer-2-lstm-autoencoder--anomaly-detection)
   - [Layer 3: LightGBM Multi-Horizon Forecaster](#layer-3-lightgbm-multi-horizon-forecaster)
   - [Layer 4: Ensemble Alert Engine](#layer-4-ensemble-alert-engine)
6. [Model Results](#6-model-results)
7. [Visualizations Explained](#7-visualizations-explained)
8. [Backtesting: Event Detection](#8-backtesting-event-detection)
9. [Archived Notebook Output](#9-archived-notebook-output)
10. [Honest Limitations](#10-honest-limitations)
11. [Setup and Usage](#11-setup-and-usage)
12. [Production Roadmap](#12-production-roadmap)
13. [References](#13-references)

---

## 1. The Problem

Global trade represents approximately **$56 trillion in annual flows**. When supply chains break, the damage propagates across industries in days. The largest disruptions in recent history:

| Event | Date | Impact |
|---|---|---|
| US-China Tariff War | Jul 2018 | $360B in goods affected, container rates spiked 40% |
| COVID-19 Demand Collapse | Mar 2020 | Global trade fell 5.3% in a single quarter |
| Ever Given / Suez Canal Blockage | Mar 2021 | $9.6B/day in trade disrupted for 6 days |
| Russia-Ukraine Invasion | Feb 2022 | Global energy and grain supply chains severed |
| Shanghai Lockdowns | Mar-Jun 2022 | World's busiest port offline for 8 weeks |
| Red Sea / Houthi Attacks | Oct 2023 | 12% of world trade rerouted, adding 10-14 days per voyage |
| US Liberation Day Tariff Shock | Apr 2025 | Broadest tariff regime since 1930, immediate supply chain repricing |

**The core problem is reaction lag.** Most enterprise risk systems run batch processes once per day. By the time an alert reaches a logistics team, the window for mitigation has often already closed. These systems also rely on static rule-based thresholds set during calm periods, which fail precisely when stress is novel and unprecedented.

**What is needed** is a system that:
1. Reads multiple real-time macro and market signals simultaneously
2. Distinguishes between normal volatility and genuine disruption
3. Quantifies risk probability at multiple operational time horizons (7 days, 14 days, 30 days)
4. Flags both *known* risk patterns (historical regime matching) and *unknown* risk patterns (novel shocks never seen before)
5. Explains which signal is driving each alert so analysts can verify and act

---

## 2. The Solution

SCRI is a **two-model, four-layer ML pipeline** that ingests seven publicly available macro and market signals in a notebook run and produces model risk probability estimates with a tiered alert classification.

The architecture deliberately separates two fundamentally different tasks:

**LSTM Autoencoder** is trained only on normal market conditions from 2015-2018. It learns what calm looks like across a 30-day window of multivariate signals. When current conditions deviate sharply from that learned normal state space, reconstruction error spikes. This catches *unknown unknowns* including novel geopolitical events the model has never seen.

**LightGBM Classifiers** are trained on 8 years of labeled historical data covering every major supply chain shock from 2015-2022. They recognize the *known patterns* that precede stress events: yield curve inversions, VIX spikes alongside falling consumer sentiment, shipping equity drawdowns. This catches *known risk regimes*.

The outputs of both models feed into a weighted ensemble that produces a single calibrated risk score for each trading day.

---

## 3. System Architecture

```
RAW DATA LAYER
  FRED API (6 series) + yfinance (4 tickers) + Synthetic Fallback
              |
              v
  Master DataFrame: 2,947 business days | 2015-01-01 to April 2026 (reported run)
              |
              v
  Feature Engineering: 64 features | lagged 1 day | point-in-time audit still required
              |
        _____|______
       |            |
       v            v
  SCSI Index    LSTM Autoencoder
  (6 signals,   (128,953 params,
   composite)    trained pre-2019)
       |            |
       |  labels    |  AE score
       v            v
  LightGBM Classifiers x3
  7d | 14d | 30d horizons
  65 features | TimeSeriesSplit CV
  Isotonic regression calibration
              |
              v
  Ensemble Alert Engine
  7d (35%) + 14d (30%) + 30d (20%) + AE (15%)
              |
              v
  Risk Tier: CLEAR / WATCH / ELEVATED / HIGH / CRITICAL
  + Calibrated probabilities at 3 horizons
  + SHAP feature attribution per alert
```

---

## 4. Data Sources

All data is **free and publicly available**. No proprietary data feeds required.

| Signal | FRED Series ID | Frequency | Role in SCSI |
|--------|---------------|-----------|--------------|
| VIX (CBOE Volatility Index) | `VIXCLS` | Daily | Primary fear/uncertainty gauge. Weight: 25% |
| WTI Crude Oil | `DCOILWTICO` | Daily | Logistics cost driver. Weight: 20% |
| PPI All Commodities | `PPIACO` | Monthly | Input cost pressure index |
| Yield Curve (10Y-2Y Spread) | `T10Y2Y` | Daily | Recession precursor. Weight: 10% |
| University of Michigan Consumer Sentiment | `UMCSENT` | Monthly | Demand proxy. Weight: 10% |
| NY Fed Recession Probability | `RECPROUSM156N` | Monthly | Macro regime indicator. Weight: 10% |
| Shipping Equity Basket | yfinance: ZIM, FDX, UPS, BDRY | Daily | Market-priced disruption signal. Weight: 20% |

**Reported data coverage:** 2,947 business days from January 2015 through the April 2026 notebook run. Coverage and real-versus-synthetic mode must be checked for each rerun; completeness alone does not establish point-in-time validity.

---

## 5. The Four-Layer Pipeline

### Layer 1: Supply Chain Stress Index (SCSI)

The SCSI is a **composite stress index** combining six normalized signals into a single standardized score. It serves as both the primary visualization and the source of supervised learning labels.

**Construction:**
```
SCSI = 0.25 * VIX_norm
     + 0.20 * Oil_volatility_norm
     + 0.20 * Shipping_drawdown_norm
     + 0.10 * Yield_curve_inverse_norm
     + 0.10 * Recession_prob_norm
     + 0.10 * Sentiment_inverse_norm
```

All components are z-score normalized and the final SCSI is re-standardized to mean 0, standard deviation 1 across full history.

**Why a composite?** No single signal is sufficient. Oil can spike from supply constraints with no demand collapse. VIX can spike from equity events unrelated to trade. The SCSI detects stress only when multiple signals diverge simultaneously, the signature of genuine supply chain disruption rather than isolated market noise.

**Binary labels for supervised learning:**
- `stress_event_7d = 1` if SCSI exceeds its 75th percentile in the next 7 business days
- `stress_event_14d = 1` if SCSI exceeds its 75th percentile in the next 14 business days
- `stress_event_30d = 1` if SCSI exceeds its 75th percentile in the next 30 business days

**SCSI statistics from this run:**
```
SCSI range           : [-1.61, 8.91]
75th pctile threshold: 0.322
Positive label rates:
  7d horizon  - 33.6% of days precede a stress event
  14d horizon - 40.2% of days precede a stress event
  30d horizon - 51.6% of days precede a stress event
```

The SCSI peak of 8.91 standard deviations corresponds to the COVID-19 demand collapse in March 2020, validating the index's sensitivity to extreme events.

---

### Layer 2: LSTM Autoencoder - Anomaly Detection

**What it is:** A sequence autoencoder built with Long Short-Term Memory networks. It compresses a 30-day window of 25 multivariate signals into a 32-dimensional latent representation and then reconstructs the original sequence from that compressed encoding.

**Key design choice:** Trained exclusively on pre-2019 data (887 sequences from the 2015-2018 calm period before any significant tariff escalation). It learns what normal market conditions look like and never sees COVID, the Suez blockage, Ukraine, or the Red Sea attacks during training.

**How it detects anomalies:** At inference time, it processes every window in the full 2015-2026 dataset. For normal periods, the reconstruction is accurate and MSE is low. When current market patterns deviate sharply from the learned normal state space, reconstruction fails and MSE spikes. This per-sample reconstruction error is the anomaly score.

**Architecture:**
```
Input: (batch, 30 timesteps, 25 features)
  LSTM Encoder (2 layers, hidden_dim=64)
  Linear projection to Latent vector (dim=32)
  Linear projection back to (batch, 30 timesteps, 64)
  LSTM Decoder (2 layers, hidden_dim=64)
  Linear output: Reconstructed sequence (batch, 30, 25)
Total parameters: 128,953
```

**Training results:**
```
Training regime  : Pre-2019 normal conditions (887 sequences)
Epochs           : 80 with early stopping and LR scheduling
Final train MSE  : 0.286
Final val MSE    : 0.464 (best checkpoint: 0.464)
Training time    : 6.9 seconds on T4 GPU
Anomaly threshold: 90th percentile of all reconstruction errors
Flagged days     : 279 out of 2,791 total sequences (10%)
```

**Why not train on all data?** If the autoencoder sees COVID during training, it learns to reconstruct COVID-level signals accurately, defeating the purpose entirely. Training only on calm periods means high-stress market patterns always produce high reconstruction error regardless of whether those patterns appeared in historical data.

**The information cascade:** The normalized AE anomaly score is fed as a feature into LightGBM and also carries explicit 15% weight in the final ensemble, ensuring the anomaly signal propagates even for novel shock patterns that gradient boosting trees cannot anticipate from historical patterns.

---

### Layer 3: LightGBM Multi-Horizon Forecaster

Three independent binary classifiers, one per forecast horizon, each answering a different operational question:

| Horizon | Operational Question | Lead Time Use Case |
|---------|---------------------|-------------------|
| **7-day** | Is a stress event likely this week? | Port rerouting, spot rate locking |
| **14-day** | Is a stress event likely in the next two weeks? | Buffer stock adjustments, carrier negotiations |
| **30-day** | Is a stress event likely this month? | Supplier diversification, budget approvals |

**Why LightGBM over neural alternatives?**
- Handles missing values natively, critical for monthly macro series on daily frequency
- No stationarity assumptions, unlike ARIMA-family models
- Captures non-linear signal interactions (simultaneous VIX spike and PMI drop is more informative than either alone)
- Fast enough to retrain daily on a single CPU
- Natively SHAP-explainable for audit and compliance requirements

**Feature space (65 features):**
- 1-day, 5-day, and 21-day returns for price-based signals
- Rolling z-scores at 21, 63, and 126-day windows for each signal
- Realized volatility (annualized) at 5, 10, and 21-day windows
- Rolling max-drawdown at 21 and 63-day windows
- Regime flags (VIX above 30, VIX above 40, PMI contraction, yield curve inversion)
- Cross-asset divergence features
- LSTM AE anomaly score
- Calendar features (quarter-end effects, month, day of week)

**Train/Test split:**
```
Training set : 2015-08-07 to 2022-12-30  (1,931 days)
               Covers all major shocks: US-China tariffs, COVID,
               Suez Canal, Russia-Ukraine, Shanghai lockdowns
Test set     : 2023-01-02 to 2026-04-17  (860 days) - GENUINELY OUT-OF-SAMPLE
               Covers: Red Sea/Houthi attacks, 2025 Liberation Day tariff shock
```

This split is the key fix from the original project version. The prior 2018 split gave the model only 627 days of calm pre-tariff data, so it never learned what a stress event looks like, producing near-random predictions. The 2023 split trains on every major historical shock and reserves the two most recent events for truly out-of-sample evaluation.

**Probability calibration:** Raw gradient boosting probabilities are good at ranking (captured by AUC-ROC) but often poorly calibrated. After training, an isotonic regression calibrator is fitted on the validation set and applied to all test predictions, ensuring a 70% predicted probability means approximately 70% of similar historical conditions actually preceded a stress event within that horizon.

---

### Layer 4: Ensemble Alert Engine

The final risk score combines all model outputs:

```
Ensemble = (0.35 x 7d_calibrated) + (0.30 x 14d_calibrated)
         + (0.20 x 30d_calibrated) + (0.15 x AE_anomaly_norm)
```

The 7-day model receives the highest weight due to its strongest measured performance. The AE score receives an explicit 15% direct weight, not relying solely on its LightGBM feature importance, to guarantee anomaly detection reaches the final output for novel shock patterns.

**Alert tiers:**

| Tier | Ensemble Score | Meaning |
|------|---------------|---------|
| CRITICAL | 0.80 or above | Extreme stress imminent. Immediate escalation. |
| HIGH | 0.60 to 0.79 | Significant disruption likely. Alert logistics leadership. |
| ELEVATED | 0.45 to 0.59 | Elevated stress probability. Increase monitoring cadence. |
| WATCH | 0.30 to 0.44 | Conditions warrant attention. Prepare alternatives. |
| CLEAR | Below 0.30 | Normal operating conditions. Routine monitoring. |

---

## 6. Model Results

The metrics below are reported from the notebook's January 2023–April 2026 test period. They are retained as historical project results, not newly reproduced measurements. Operational validity requires a review of the preprocessing, label construction, calibration splits, and point-in-time data availability.

### LightGBM Performance (Calibrated, Out-of-Sample Test Set)

| Horizon | AUC-ROC | AUPRC | Interpretation |
|---------|---------|-------|----------------|
| **7-day** | **0.711** | **0.643** | Some reported ranking ability; further validation required. |
| 14-day | 0.523 | 0.602 | Near-random ranking; operational usefulness not established. |
| 30-day | 0.479 | 0.755 | Below 0.5 ranking; operational usefulness not established. |

**What AUC-ROC measures:** Area Under the Receiver Operating Characteristic curve. A score of 1.0 is a perfect model; 0.5 is a coin flip. An AUC of 0.711 means the 7-day model correctly ranks a positive day above a negative day 71% of the time. This is evidence of reported ranking ability, not proof of deployment readiness or reliable early warning.

**What AUPRC measures:** Area Under the Precision-Recall Curve. Unlike AUC-ROC, AUPRC is sensitive to class imbalance and directly measures how well the model identifies positive events without producing false alarms. The 7-day AUPRC of 0.643 versus a baseline positive rate of 0.477 represents meaningful lift above random.

**30-day interpretation:** ROC-AUC of 0.479 does not demonstrate useful ranking on this test set. The reported positive-label prevalence is 76.3%; prevalence does not by itself invalidate ROC-AUC or excuse weak discrimination. Compare the model against prevalence-based and simple time-series baselines, inspect uncertainty, and validate calibration on a separate chronological split. Do not use this output as an operational forecast without that work.

### Cross-Validation (7-Day Model, Expanding Window, 5 Folds)

| Fold | Train Size | Val Size | AUC-ROC | AP |
|------|-----------|---------|---------|-----|
| 1 | 459 | 465 | 0.710 | 0.184 |
| 2 | 924 | 465 | 0.769 | 0.804 |
| 3 | 1,389 | 465 | 0.642 | 0.602 |
| 4 | 1,854 | 465 | 0.630 | 0.526 |
| 5 | 2,319 | 465 | 0.657 | 0.643 |
| **Mean** | | | **0.682 +/- 0.051** | **0.552 +/- 0.205** |

Reported fold AUC ranges from 0.63 to 0.77. AP varies substantially across folds. These summaries warrant a chronological split audit, including a gap for overlapping forward labels, before claiming robust performance across regimes.

### SHAP Feature Importance (7-Day Model, Top 5 Drivers)

| Rank | Feature | Mean SHAP | Economic Interpretation |
|------|---------|-----------|------------------------|
| 1 | `sent` (Consumer Sentiment) | 0.257 | Low consumer confidence precedes reduced order volumes and demand-driven supply disruptions |
| 2 | `vix` (Volatility Index) | 0.239 | Market fear is the strongest real-time signal of perceived disruption risk |
| 3 | `wti_oil` (WTI Crude) | 0.172 | Oil price level directly determines freight rates and logistics input costs |
| 4 | `ship_stock` (Shipping Equity Basket) | 0.139 | Shipping company valuations price in future capacity and demand changes months ahead |
| 5 | `sent_zscore_126d` (Sentiment 6-month Z-score) | 0.086 | Persistent sentiment weakness signals structural demand collapse, not just a cyclical dip |

This feature ranking is economically coherent and tells a clear story. Consumer sentiment and market fear lead, followed by direct cost signals (oil), then market-priced forward indicators (shipping equities), then regime-level sentiment persistence. The AE anomaly score contributes primarily through its explicit 15% ensemble weight rather than via LightGBM feature importance, confirming the anomaly detector provides complementary information that gradient boosting trees cannot capture from historical patterns alone.

---

## 7. Visualizations Explained

The notebook generates six output plots. Here is what each shows and how to interpret it.

---

### scri_eda_signals.png - Signal Overview

**What it shows:** Four time-series panels from 2015 to present.

Panel 1 (SCSI): The composite stress index over time. Red shading marks periods when SCSI exceeded zero (elevated stress); green marks subdued periods. Dashed vertical lines are in-sample shock events; dotted lines are out-of-sample events. The COVID spike to approximately 9 standard deviations dominates the chart visually.

Panel 2 (VIX): The CBOE Volatility Index. The dashed line at 30 marks the fear threshold. Readings above 30 historically correlate with meaningful financial market stress that feeds into supply chain uncertainty.

Panel 3 (WTI Crude Oil vs ISM Manufacturing PMI): Dual-axis chart. Oil price on the left axis (orange), PMI on the right axis (purple). The dotted line at PMI = 50 is the expansion/contraction boundary. When oil is high and PMI is simultaneously below 50, that is the stagflationary combination historically most damaging to supply chains.

Panel 4 (Yield Curve): The 10Y-2Y Treasury spread. Red shading marks yield curve inversion (spread below zero), which has preceded every US recession in the modern era. The 2022-2024 inversion is clearly visible.

**How to read it:** Look for convergence across panels. A single-signal anomaly such as an oil spike without a VIX move is often isolated noise. Multiple panels moving simultaneously is the signature of genuine supply chain stress.

---

### scri_eda_dist.png - Correlation and SCSI Distribution

**What it shows:** Two panels.

Left panel (Full signal correlation heatmap): Lower-triangular correlation matrix of all available signals. Negative correlations (blue) indicate opposing movements, which are valuable because they confirm the features carry independent information rather than just measuring the same underlying stress factor.

Right panel (SCSI distribution by regime): Histogram comparing all SCSI values (blue) versus SCSI values in the 7 days before a labeled stress event (red). The dashed vertical line is the 75th percentile threshold (0.322) used to generate binary labels.

**How to read it:** The key insight in the right panel is that the pre-event distribution (red) is shifted significantly to the right of the full distribution (blue). This confirms that the SCSI has real predictive content. The degree of separation between the two histograms is a visual measure of how informative the SCSI is as a binary label source.

---

### ae_anomaly_detection.png - LSTM Autoencoder Results

**What it shows:** Two panels sharing the same time axis from 2015 to present.

Top panel (Reconstruction Error): The purple line is the daily anomaly score on a 0-100 scale. Red-shaded regions are days where the score exceeded the 90th percentile threshold (flagged as anomalous). The red dashed horizontal line marks that 90th percentile. Dotted vertical lines mark known events.

Bottom panel (SCSI vs Anomaly Detections): The SCSI index (dark line) with red shading for periods above the 75th percentile, confirming that anomaly detections in the top panel correspond to genuine SCSI elevation in the bottom panel.

**How to read it:** The most important visual check is alignment between flagged periods in the top panel and high-SCSI periods in the bottom panel. The COVID spike is the most dramatic validation: the AE reconstruction error peaks near 100 in March-April 2020, precisely when SCSI peaks at 8.91 standard deviations. The autoencoder, trained only on 2015-2018 calm data and never shown COVID, correctly identifies it as the most extreme anomaly in the full dataset.

---

### lgb_calibration.png - Probability Calibration Curves

**What it shows:** Three panels, one per horizon, comparing raw LightGBM probabilities (red circles) to isotonic-calibrated probabilities (green squares) against a perfect calibration diagonal (black dashes).

**How to read it:** On a perfectly calibrated model, every point falls exactly on the diagonal, meaning when the model predicts 40% probability, approximately 40% of those predictions are followed by a stress event. Points above the diagonal mean the model is underconfident. Points below mean it is overconfident. The green calibrated series should sit closer to the diagonal than the red raw series. The 7-day model calibrates cleanly (raw AUC 0.710 to calibrated AUC 0.711, essentially unchanged, confirming the model was already well-ranked and calibration only refined the probability scale). The 30-day output is nearly constant in the reported run. That is a calibration and usefulness concern to investigate, rather than proof of correct behavior.

---

### lgb_results.png - Multi-Horizon Model Performance

**What it shows:** A 2x3 grid of six panels.

Top row, left (ROC Curves): True Positive Rate vs False Positive Rate for all three horizons. The 7d curve (red) curves furthest toward the upper-left, confirming strongest ranking performance (AUC = 0.711). The 30d curve lies near the diagonal, confirming near-random ranking.

Top row, middle (Precision-Recall Curves): Unlike ROC curves, PR curves are not fooled by class imbalance. The 7-day AUPRC of 0.643 represents meaningful lift over the naive baseline of 0.477.

Top row, right (AUC-ROC vs AUPRC bar chart): Side-by-side comparison by horizon. The reported comparison shows weaker ranking at longer horizons in this run. It does not establish why performance deteriorates.

Bottom row (Time-series overlays, one per horizon): The colored fill shows calibrated risk probability over the 2023-2026 test period. The dark line is the actual SCSI. Black dots at the top mark days where the true label was positive. Vertical dotted lines mark out-of-sample events. The key check is whether the colored fill rises before the event lines, not after.

---

### scri_dashboard.png - Risk Dashboard (Test Period)

**What it shows:** Three stacked panels covering the full out-of-sample test period, January 2023 to April 2026.

Top panel (Ensemble Risk Score and Tier): The primary operational view. Background color bands show the current risk tier (green = CLEAR, yellow = WATCH, orange = ELEVATED, red = HIGH, dark red = CRITICAL). The black line is the ensemble risk score on a 0-1 scale. This is what a risk analyst monitors daily.

Middle panel (Risk Probability by Horizon plus AE Score): Three colored lines for 7d, 14d, and 30d calibrated probabilities, plus the purple dashed line for the normalized AE anomaly score. When the 30-day line rises before the 7-day line, the model is detecting a building regime shift. When the AE score spikes without a corresponding LightGBM move, the autoencoder has detected structural novelty that historical pattern-matching has not yet processed.

Bottom panel (SCSI Actual vs 7-Day Forecast): Dual-axis overlay of actual SCSI (dark line) against the 7-day calibrated risk probability (red fill). The right axis should track the left axis with the forecast anticipating SCSI movements rather than following them. An apparent lead in an overlay must be checked against dated predictions, point-in-time inputs, and explicit event timing before claiming predictive lead time.

**How to read the dashboard overall:** Periods where all three panels simultaneously show elevated signals are the highest-confidence alerts. Divergences between panels are informative. AE elevated but LightGBM flat suggests a novel pattern the historical training cannot explain. LightGBM elevated but AE flat suggests a familiar historical regime recurring.

---

## 8. Backtesting: Event Detection

The reported event check covers two selected test-period episodes. It does not establish general detection accuracy or predictive lead time.

```
=== AUDIT: Event Detection (Out-of-Sample Test Period, 2023-present) ===

Detection threshold (80th percentile of test scores): 0.6055
Elevated threshold  (70th percentile of test scores): 0.5792

Event                             Date        Max Score  Tier    Status         Set
US-China Tariffs (Round 1)        2018-07-06  N/A        N/A     NO DATA        in-sample
US-China Tariffs (Escalation)     2019-05-10  N/A        N/A     NO DATA        in-sample
COVID-19 Pandemic                 2020-03-11  N/A        N/A     NO DATA        in-sample
Suez Canal Blockage               2021-03-23  N/A        N/A     NO DATA        in-sample
Russia-Ukraine Invasion           2022-02-24  N/A        N/A     NO DATA        in-sample
Shanghai Lockdowns                2022-03-28  N/A        N/A     NO DATA        in-sample

** Red Sea / Houthi Attacks       2023-10-19  0.7514     HIGH    DETECTED       OUT-OF-SAMPLE
** US Tariff Shock (Liberation Day) 2025-04-02 0.7884    HIGH    DETECTED       OUT-OF-SAMPLE

Selected historical episode coverage: 2/2 (small sample)
Data integrity              : 100.00% real-world data density
```

**Why do in-sample events show NO DATA?** This is architecturally correct. The risk report covers only the test period (2023-present). In-sample events (2015-2022) are inside the training window. Including them in the detection audit would be data leakage and would artificially inflate results. They are excluded by design.

**What the two-episode result means:** both selected episodes had elevated scores within a reported 15-day window. The timing direction and first alert must be shown before claiming advance warning. Thresholds were based on test scores, so this is a retrospective diagnostic. A broader preregistered event set, false-alert counts, and lead-time distribution are needed to estimate operational performance.

---

## 9. Archived Notebook Output

Historical output from the notebook run dated April 18, 2026. This is an archived snapshot, not a current or live signal:

```
Date         7d Risk  14d Risk  30d Risk  AE Score  Ensemble  Tier
2026-04-13    0.385    0.313     1.000     0.557     0.448     WATCH
2026-04-14    0.385    0.667     1.000     0.559     0.538     ELEVATED
2026-04-15    0.385    0.313     1.000     0.548     0.446     WATCH
2026-04-16    0.385    0.400     1.000     0.543     0.468     ELEVATED
2026-04-17    0.385    0.500     1.000     0.539     0.493     ELEVATED

Current SCSI: 0.546 to 0.670 (well above the 0.322 stress threshold)
```

These outputs document the reported run. A 30-day estimate of 1.0 requires calibration review and should not be interpreted as certainty. The snapshot does not demonstrate live monitoring or independently confirm a real-world risk assessment.

---

## 10. Honest Limitations

**Evaluation work still required:** fit normalization and thresholds within each training window; check release dates and revised macro data; separate fitting, calibration, and testing chronologically; use a suitable gap for overlapping forward labels; report baseline comparisons, uncertainty, false alerts, and actual advance-warning timing. These are review priorities, not claimed completed fixes. Synthetic fallback results must be clearly separated from real-data results.

**1. The 30-day model has weak reported discrimination (ROC-AUC 0.479).**
High positive prevalence does not establish forecast quality. Compare against simple baselines and review calibration before any operational use.

**2. The 14-day model is near-random (ROC-AUC 0.523).**
The reason for this result has not been established. All horizons require validation; the better reported 7-day result alone is not sufficient for operational use.

**3. The LSTM AE has a training-validation gap (train MSE 0.286, val MSE 0.464).**
The autoencoder overfits slightly to the 2015-2018 calm window. Validation sequences from late 2018 already carry early tariff-era patterns that differ from the training regime. This does not break anomaly detection but the gap could be reduced by extending the normal training window to pre-2020.

**4. No Baltic Dry Index (BDI) in real data mode.**
The original design included BDI as a direct shipping demand signal. BDI is not available via the FRED API and requires Bloomberg or Quandl for high-quality history. The system compensates with the shipping equity basket (ZIM, FDX, UPS, BDRY) in real-data mode.

**5. Shipping equity basket starts in 2021.**
ZIM International only listed in January 2021. The basket has 1,362 observations rather than the full 2,947. Forward-filling is used for pre-2021 dates, meaning the shipping equity feature carries less information before 2021.

---

## 11. Setup and Usage

### Option A: Google Colab (Recommended, Zero Local Setup)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. File > Upload notebook > select `SCRI_SupplyChainRisk.ipynb`
3. Add your FRED API key via Colab Secrets (lock icon in the left sidebar):
   - Name: `FRED_API_KEY`
   - Value: your key from [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) (free registration)
   - Toggle "Notebook access" to ON
4. Set runtime: Runtime > Change runtime type > T4 GPU
5. Uncomment the pip install line in Step 0
6. Runtime > Run all

Expected confirmation in Step 1:
```
FRED API Key: SET  (source: Colab Secrets)
Device     : cuda
```

Expected confirmation in Step 2:
```
Data successfully loaded from FRED and yfinance.
Master DataFrame: (2947, 8) | Mode: REAL
```

**No API key?** The notebook auto-falls back to a synthetic data generator. All architecture, ML logic, and results structure run identically. The synthetic generator uses AR(1) processes with regime-switching and injects known shock events at correct historical dates.

### Option B: Local Jupyter

Clone the actual repository:

```bash
git clone https://github.com/Agent007repo/SCRI-Supply-Chain-Risk-Intelligence-System.git
cd SCRI-Supply-Chain-Risk-Intelligence-System
```

Use the dependency list below and review the notebook's configuration cells for local execution. This repository currently distributes the notebook rather than a packaged local application; a tested environment lockfile and local setup path remain development work. The Colab path above is the documented starting point.

### Requirements

```
Python     3.10+
lightgbm   4.6.0
torch      2.10.0
sklearn    1.6.1
yfinance   0.2.66
fredapi    0.5.2
shap       0.51.0
pandas     2.2.2
numpy      2.0.2
scipy      1.16.3
python-dotenv 1.0+
```

GPU is not required but reduces LSTM training from approximately 2 minutes on CPU to 7 seconds on T4 GPU.

---

## 12. Production Roadmap

| Phase | Component | Technology |
|-------|-----------|-----------|
| 1 | Real-time data pipeline | Apache Kafka + FRED/Bloomberg WebSocket |
| 2 | Feature store | Feast / Tecton |
| 3 | Model serving + calibration | FastAPI + MLflow |
| 4 | Drift detection + retrain trigger | Evidently AI |
| 5 | Alert routing | PagerDuty / Slack webhook |
| 6 | Trade corridor mapping | Marine Traffic API |
| 7 | BDI integration | Bloomberg Data License / Quandl |
| 8 | Geopolitical risk signal | Caldara-Iacoviello GPR index |

---

## 13. References

**Academic:**
- Benigno et al. (2022). *Global Supply Chain Pressure Index.* NY Fed Liberty Street Economics.
- Caldara & Iacoviello (2022). *Measuring Geopolitical Risk.* American Economic Review.
- Lim et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International Journal of Forecasting.
- Weber et al. (2022). *Reconfigurations of global supply chains.* NBER Working Paper 30457.
- Niculescu-Mizil & Caruana (2005). *Predicting Good Probabilities with Supervised Learning.* ICML.
- Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*, 3rd edition.

**Data Sources:**
- FRED API: https://fred.stlouisfed.org/docs/api/fred
- NY Fed GSCPI: https://www.newyorkfed.org/research/policy/gscpi
- Caldara-Iacoviello GPR: https://www.matteoiacoviello.com/gpr.htm

---

<div align="center">

Independent research | Historical results documented | Point-in-time evaluation remains a priority

</div>
