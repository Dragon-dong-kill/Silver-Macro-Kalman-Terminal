from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DOLLAR_WEIGHT,
    DXY_TICKER,
    FRED_REAL_RATE,
    RATE_WEIGHT,
    SENTIMENT_WEIGHT,
    SILVER_TICKER,
    TNX_TICKER,
    VIX_TICKER,
    GOLD_TICKER,
    COPPER_TICKER,
    SP500_TICKER,
    FRED_BREAKEVEN,
)
from data.utils import first_available_column, rolling_zscore


def compute_macro_score(
    closes: pd.DataFrame,
    real_rate: pd.DataFrame,
    sentiment: pd.DataFrame,
    extra_data: pd.DataFrame | None,
    z_window: int,
    sentiment_weight: float = SENTIMENT_WEIGHT,
    dollar_weight: float = DOLLAR_WEIGHT,
    rate_weight: float = RATE_WEIGHT,
) -> pd.DataFrame:
    macro = pd.DataFrame(index=closes.index)
    macro["silver"] = first_available_column(closes, [SILVER_TICKER])
    macro["dxy"] = first_available_column(closes, [DXY_TICKER]).ffill()
    macro["tnx"] = first_available_column(closes, [TNX_TICKER]).ffill()

    macro["dxy_delta"] = macro["dxy"].pct_change() * 100.0
    macro["tnx_delta"] = macro["tnx"].diff()

    if not real_rate.empty and FRED_REAL_RATE in real_rate.columns:
        aligned_real_rate = real_rate[[FRED_REAL_RATE]].reindex(macro.index, method="ffill")
        macro["real_rate"] = pd.to_numeric(aligned_real_rate[FRED_REAL_RATE], errors="coerce")
        macro["rate_delta"] = macro["real_rate"].diff().combine_first(macro["tnx_delta"])
    else:
        macro["real_rate"] = np.nan
        macro["rate_delta"] = macro["tnx_delta"]

    if not real_rate.empty and FRED_BREAKEVEN in real_rate.columns:
        aligned_be = real_rate[[FRED_BREAKEVEN]].reindex(macro.index, method="ffill")
        macro["breakeven"] = pd.to_numeric(aligned_be[FRED_BREAKEVEN], errors="coerce")
        macro["breakeven_delta"] = macro["breakeven"].diff()
    else:
        macro["breakeven"] = np.nan
        macro["breakeven_delta"] = 0.0

    if extra_data is not None and not extra_data.empty:
        for ticker, col_name in [
            (VIX_TICKER, "vix"),
            (GOLD_TICKER, "gold"),
            (COPPER_TICKER, "copper"),
            (SP500_TICKER, "sp500"),
        ]:
            macro[col_name] = first_available_column(extra_data, [ticker]).ffill()
        macro["gold_silver_ratio"] = (
            (macro["gold"] / macro["silver"].replace(0.0, np.nan))
            .replace([np.inf, -np.inf], np.nan)
            .ffill()
        )
        macro["copper_ret"] = macro["copper"].pct_change() * 100.0
        macro["sp500_ret"] = macro["sp500"].pct_change() * 100.0
        macro["vix_delta"] = macro["vix"].pct_change() * 100.0
    else:
        for col in ["vix", "gold", "copper", "sp500", "gold_silver_ratio", "copper_ret", "sp500_ret", "vix_delta"]:
            macro[col] = np.nan

    if not sentiment.empty and "sentiment" in sentiment:
        sentiment_map = sentiment["sentiment"].copy()
        sentiment_map.index = pd.to_datetime(sentiment_map.index).normalize()
        macro_dates = pd.Series(pd.to_datetime(macro.index).normalize(), index=macro.index)
        macro["sentiment_raw"] = macro_dates.map(sentiment_map).fillna(0.0)
    else:
        macro["sentiment_raw"] = 0.0

    macro["z_sentiment"] = rolling_zscore(macro["sentiment_raw"], z_window)
    macro["z_dxy"] = rolling_zscore(macro["dxy_delta"], z_window)
    macro["z_rate"] = rolling_zscore(macro["rate_delta"], z_window)
    macro["z_vix"] = rolling_zscore(macro["vix_delta"], z_window)
    macro["z_gold_silver"] = rolling_zscore(macro["gold_silver_ratio"], z_window)
    macro["z_copper"] = rolling_zscore(macro["copper_ret"], z_window)
    macro["z_sp500"] = rolling_zscore(macro["sp500_ret"], z_window)

    macro["sentiment_factor"] = sentiment_weight * macro["z_sentiment"]
    macro["dxy_factor"] = -dollar_weight * macro["z_dxy"]
    macro["rate_factor"] = -rate_weight * macro["z_rate"]

    macro["U_score"] = (
        macro["sentiment_factor"] + macro["dxy_factor"] + macro["rate_factor"]
    )

    extra_enrichment = (
        -0.08 * macro["z_vix"].fillna(0.0)
        + 0.06 * macro["z_gold_silver"].fillna(0.0)
        + 0.06 * macro["z_copper"].fillna(0.0)
        + 0.04 * macro["z_sp500"].fillna(0.0)
    )
    macro["U_score"] = (macro["U_score"] + extra_enrichment.fillna(0.0)).clip(-5.0, 5.0)

    abs_return = macro["silver"].pct_change().abs()
    vol_fast = abs_return.rolling(window=14, min_periods=5).mean()
    vol_slow = abs_return.rolling(window=60, min_periods=20).mean()
    macro["vol_ratio"] = (
        vol_fast.divide(vol_slow.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .clip(0.30, 3.50)
        .fillna(1.0)
    )
    return macro
