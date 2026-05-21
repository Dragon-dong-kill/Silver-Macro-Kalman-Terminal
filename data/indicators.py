from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import CACHE_TTL_SENTIMENT, NEWS_KEYWORDS, RSS_FEEDS
from data.fetcher import FetchResult
from data.utils import first_available_column, rolling_zscore


def calculate_adx(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    if ohlc.empty:
        return pd.Series(dtype=float)

    data = ohlc.copy()
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    tr_components = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    )
    true_range = tr_components.max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_dm_smoothed = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_smoothed / atr.replace(0.0, np.nan)
    minus_di = 100.0 * minus_dm_smoothed / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return adx.replace([np.inf, -np.inf], np.nan)
