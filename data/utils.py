from __future__ import annotations

import pandas as pd
import numpy as np


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    minimum_periods = max(10, min(window, 20))
    mean = values.ewm(span=window, min_periods=minimum_periods, adjust=False).mean()
    std = values.ewm(span=window, min_periods=minimum_periods, adjust=False).std(bias=False)
    z = (values - mean) / std.replace(0.0, np.nan)

    if z.isna().all():
        fallback_std = values.std(ddof=0)
        if pd.notna(fallback_std) and fallback_std > 0:
            z = (values - values.mean()) / fallback_std

    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-4.0, 4.0)


def first_available_column(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(index=frame.index, dtype=float)
