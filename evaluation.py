"""Time-series evaluation helpers. Horizons count observations (business days)."""
import numpy as np
import pandas as pd


def forward_window(series, horizon, statistic='max'):
    """Future t+1..t+h only; unknown trailing outcomes remain NaN."""
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('horizon must be a positive integer.')
    if statistic not in {'max', 'mean'}:
        raise ValueError('statistic must be max or mean.')
    window = pd.concat([series.shift(-i) for i in range(1, horizon + 1)], axis=1)
    known = window.notna().all(axis=1)
    values = window.max(axis=1) if statistic == 'max' else window.mean(axis=1)
    return values.where(known)


def frozen_zscore(series, reference_end='2018-12-31'):
    reference = series.loc[:reference_end].dropna()
    if len(reference) < 2:
        raise ValueError('At least two reference observations are required.')
    scale = reference.std()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (series - reference.mean()) / scale


def chronological_partitions(n, gap=30, validation_fraction=.15, calibration_fraction=.15):
    """Return fitting, early-stop, and calibration slices with purged boundaries."""
    if gap < 0 or not 0 < validation_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError('Invalid partition parameters.')
    n_val = max(1, int(n * validation_fraction))
    n_cal = max(1, int(n * calibration_fraction))
    cal_start = n - n_cal
    val_end = cal_start - gap
    val_start = val_end - n_val
    fit_end = val_start - gap
    if fit_end < 2:
        raise ValueError('Not enough observations for three partitions and label gaps.')
    return slice(0, fit_end), slice(val_start, val_end), slice(cal_start, n)
