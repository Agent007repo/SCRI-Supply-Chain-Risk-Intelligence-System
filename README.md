# SCRI: Supply Chain Risk Intelligence System

An offline research notebook exploring macro-stress forecasting and anomaly detection. It constructs a Supply Chain Stress Index (SCSI), trains an LSTM autoencoder, fits LightGBM classifiers for 7-, 14-, and 30-business-day outcomes, and combines calibrated forecasts with an anomaly score.

SCSI is a constructed macro/market proxy. It is not an independently observed measure of shipping delays, disruption losses, or real-world event probability.

## Pipeline and evaluation boundaries

| Stage | Current implementation |
|---|---|
| Inputs | FRED macro series and yfinance shipping equities; synthetic fallback can exercise the workflow |
| SCSI | Available components are weighted and standardized using a frozen pre-2019 reference; components without that history are omitted |
| Targets | Threshold exceedance in observations t+1 through t+h; unknown tails stay missing |
| Autoencoder | Pre-2019 features and scaler; disjoint sequence training/validation windows; training-reference anomaly scaling |
| Classifiers | Pre-2023 data, a 30-observation purge before test, and separate purged fitting, early-stopping, and calibration blocks |
| Cross-validation | Training data only, starting in 2019 after the reference regime, with purged 7-day outcomes |
| Ensemble | 35% 7-day, 30% 14-day, 20% 30-day, 15% autoencoder; fixed demonstration alert bands |
| Reporting | Current feature-complete rows can be forecast even before their future labels become observable |

Features are selected using training-period coverage. Macro data are forward-filled without backward-filling future observations. `evaluation.py` contains the forward-window, frozen normalization, and chronological partition helpers. Isotonic calibration is fitted separately from early stopping; a fitting curve is not evidence of test calibration.

## Results status

Earlier reported 7-day ROC-AUC 0.711, 14-/30-day metrics, event detections, anomaly counts, and dated alerts are withdrawn as evidence for the corrected pipeline. Original preprocessing used full-history transformations and thresholds, misaligned forward labels, overlapping evaluation windows, and test-derived event thresholds. The ensemble also scaled its horizon weights twice. Outputs are cleared; archived PNG figures predate these fixes and must not be presented as current results.

No full corrected historical run has been reproduced. Synthetic smoke tests demonstrate execution, not economic predictive value. Report real versus synthetic source provenance, baseline comparisons, calibration, confidence intervals, and event-level false alarms before making an application claim.

## Run

Use Python 3.11+ and install `requirements.txt` in a clean environment. The broad version ranges are not a reproducible lockfile. Copy `.env.example` to `.env`, set `FRED_API_KEY`, and run `SCRI_SupplyChainRisk.ipynb` from the repository root.

```bash
pip install -r requirements.txt
jupyter notebook SCRI_SupplyChainRisk.ipynb
```

Internet access is needed for real data. An unavailable source can trigger synthetic fallback: inspect ingestion messages before interpreting results. Generated checkpoints and local credentials are ignored by Git.

```bash
python -m unittest discover -s tests -p test_regressions.py -v
```

Regression tests cover exact forward horizons, unknown labels, future-invariant transforms, purged partitions, and ensemble weights. They do not verify live APIs, every dependency combination, or operational forecasting.

## Remaining research limitations

Historical FRED/yfinance observations are not a point-in-time vintage archive. Monthly release dates and revisions still require auditing. Calendar alignment, constructed-label validity, the frozen regime choice, feature redundancy, and ensemble design require independent evaluation. Alert scores are demonstration bands; the ensemble has no established event-probability calibration. This is a retrospective prototype, not a live early-warning service. No energy, carbon, or sovereignty benchmark is provided.

## License

MIT. See `LICENSE`.
