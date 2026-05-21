import numpy as np
import pandas as pd

from config import (
    DOLLAR_WEIGHT, DXY_TICKER, FRED_REAL_RATE, RATE_WEIGHT,
    SENTIMENT_WEIGHT, SILVER_TICKER, TNX_TICKER,
    VIX_TICKER, GOLD_TICKER, COPPER_TICKER, SP500_TICKER, FRED_BREAKEVEN,
)
from data.utils import first_available_column, rolling_zscore


def compute_macro_score(closes, real_rate, sentiment, extra_data, z_window):
    macro = pd.DataFrame(index=closes.index)
    macro["silver"] = first_available_column(closes, [SILVER_TICKER])
    macro["dxy"] = first_available_column(closes, [DXY_TICKER]).ffill()
    macro["tnx"] = first_available_column(closes, [TNX_TICKER]).ffill()
    macro["dxy_delta"] = macro["dxy"].pct_change() * 100.0
    macro["tnx_delta"] = macro["tnx"].diff()

    ret = macro["silver"].pct_change()
    macro["silver_momentum_5"] = ret.rolling(5).sum()
    macro["silver_momentum_20"] = ret.rolling(20).sum()

    if not real_rate.empty and FRED_REAL_RATE in real_rate.columns:
        a = real_rate[[FRED_REAL_RATE]].reindex(macro.index, method="ffill")
        macro["real_rate"] = pd.to_numeric(a[FRED_REAL_RATE], errors="coerce")
        macro["rate_delta"] = macro["real_rate"].diff().combine_first(macro["tnx_delta"])
    else:
        macro["real_rate"] = np.nan
        macro["rate_delta"] = macro["tnx_delta"]

    if not real_rate.empty and FRED_BREAKEVEN in real_rate.columns:
        ab = real_rate[[FRED_BREAKEVEN]].reindex(macro.index, method="ffill")
        macro["breakeven"] = pd.to_numeric(ab[FRED_BREAKEVEN], errors="coerce")
        macro["breakeven_delta"] = macro["breakeven"].diff()
    else:
        macro["breakeven"] = np.nan
        macro["breakeven_delta"] = 0.0

    if extra_data is not None and not extra_data.empty:
        for ticker, col_name in [(VIX_TICKER, "vix"), (GOLD_TICKER, "gold"),
                                  (COPPER_TICKER, "copper"), (SP500_TICKER, "sp500")]:
            macro[col_name] = first_available_column(extra_data, [ticker]).ffill()
        gs = (macro["gold"] / macro["silver"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill()
        macro["gold_silver_ratio"] = gs
        macro["copper_momentum"] = macro["copper"].pct_change(20)
        macro["sp500_momentum"] = macro["sp500"].pct_change(20)
        macro["vix_pct"] = macro["vix"].pct_change() * 100.0
    else:
        for col in ["vix", "gold", "copper", "sp500", "gold_silver_ratio",
                     "copper_momentum", "sp500_momentum", "vix_pct"]:
            macro[col] = np.nan

    if not sentiment.empty and "sentiment" in sentiment:
        sm = sentiment["sentiment"].copy()
        sm.index = pd.to_datetime(sm.index).normalize()
        mdates = pd.Series(pd.to_datetime(macro.index).normalize(), index=macro.index)
        macro["sentiment_raw"] = mdates.map(sm).fillna(0.0)
    else:
        macro["sentiment_raw"] = 0.0

    macro["z_sentiment"] = rolling_zscore(macro["sentiment_raw"], z_window)
    macro["z_dxy"] = rolling_zscore(macro["dxy_delta"], z_window)
    macro["z_rate"] = rolling_zscore(macro["rate_delta"], z_window)
    macro["z_vix"] = rolling_zscore(macro["vix_pct"].fillna(0), z_window)
    macro["z_gold_silver"] = rolling_zscore(macro["gold_silver_ratio"].ffill(), z_window)
    macro["z_copper"] = rolling_zscore(macro["copper_momentum"].fillna(0), z_window)
    macro["z_sp500"] = rolling_zscore(macro["sp500_momentum"].fillna(0), z_window)
    macro["z_breakeven"] = rolling_zscore(macro["breakeven_delta"].fillna(0), z_window)

    macro["sentiment_factor"] = SENTIMENT_WEIGHT * macro["z_sentiment"]
    macro["dxy_factor"] = -DOLLAR_WEIGHT * macro["z_dxy"]
    macro["rate_factor"] = -RATE_WEIGHT * macro["z_rate"]

    macro["U_score"] = macro["sentiment_factor"] + macro["dxy_factor"] + macro["rate_factor"]
    extra_boost = (
        -0.12 * macro["z_vix"].fillna(0) + 0.08 * macro["z_gold_silver"].fillna(0)
        + 0.08 * macro["z_copper"].fillna(0) + 0.05 * macro["z_sp500"].fillna(0)
        + 0.05 * macro["z_breakeven"].fillna(0)
    )
    macro["U_score"] = (macro["U_score"] + extra_boost.fillna(0)).clip(-5.0, 5.0)

    abs_return = macro["silver"].pct_change().abs()
    vol_fast = abs_return.ewm(span=14, min_periods=5).mean()
    vol_slow = abs_return.ewm(span=60, min_periods=20).mean()
    macro["vol_ratio"] = (vol_fast / vol_slow.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0.30, 3.50).fillna(1.0)

    macro["fair_value_deviation"] = _compute_fair_value(macro)

    macro["market_regime"] = _detect_regime(macro)

    _compute_bollinger(macro, z_window)

    return macro


def _detect_regime(macro):
    close = macro["silver"]
    ma20 = close.rolling(20, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=20).mean()
    trend_ratio = (ma20 / ma50.replace(0.0, np.nan) - 1.0).abs()

    vol20 = close.pct_change().rolling(20).std()
    vol50 = close.pct_change().rolling(50).std()
    vol_expanding = (vol20 / vol50.replace(0.0, np.nan)).fillna(1.0)

    regime = pd.Series("ranging", index=macro.index)
    regime[(trend_ratio > 0.025) & (vol_expanding > 0.8)] = "trending"
    regime[(trend_ratio > 0.05) & (vol_expanding > 1.1)] = "strong_trending"

    return regime.ffill().fillna("ranging")


def _compute_fair_value(macro):
    factors = macro[["z_sentiment", "z_dxy", "z_rate", "z_vix"]].fillna(0)
    composite = (0.30 * factors["z_sentiment"] - 0.25 * factors["z_dxy"]
                 - 0.25 * factors["z_rate"] - 0.10 * factors["z_vix"]
                 - 0.10 * macro.get("z_gold_silver", pd.Series(0, index=macro.index)).fillna(0))
    z = (composite - composite.rolling(60, min_periods=20).mean()) / composite.rolling(60, min_periods=20).std().replace(0.0, np.nan)
    return z.clip(-4.0, 4.0).fillna(0.0)


def _compute_bollinger(macro, z_window):
    close = macro["silver"]
    ma20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std()
    macro["bb_upper"] = ma20 + std20 * 1.8
    macro["bb_lower"] = ma20 - std20 * 1.8
    macro["bb_mid"] = ma20
    macro["bb_width"] = (std20 * 3.6 / ma20.replace(0.0, np.nan) * 100).fillna(2.0)
    macro["bb_position"] = ((close - ma20) / std20.replace(0.0, np.nan)).clip(-3.0, 3.0).fillna(0.0)
