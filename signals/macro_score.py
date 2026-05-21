
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
    closes,
    real_rate,
    sentiment,
    extra_data,
    z_window,
    sentiment_weight=SENTIMENT_WEIGHT,
    dollar_weight=DOLLAR_WEIGHT,
    rate_weight=RATE_WEIGHT,
):
    macro = pd.DataFrame(index=closes.index)

    macro["silver"] = first_available_column(closes, [SILVER_TICKER])
    macro["dxy"] = first_available_column(closes, [DXY_TICKER]).ffill()
    macro["tnx"] = first_available_column(closes, [TNX_TICKER]).ffill()

    macro["dxy_delta"] = macro["dxy"].pct_change() * 100.0
    macro["tnx_delta"] = macro["tnx"].diff()

    silver_ret = macro["silver"].pct_change()
    macro["silver_momentum_5"] = silver_ret.rolling(5).sum()
    macro["silver_momentum_20"] = silver_ret.rolling(20).sum()

    if not real_rate.empty and FRED_REAL_RATE in real_rate.columns:
        aligned = real_rate[[FRED_REAL_RATE]].reindex(macro.index, method="ffill")
        macro["real_rate"] = pd.to_numeric(aligned[FRED_REAL_RATE], errors="coerce")
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
        macro["gold_silver_ratio"] = (macro["gold"] / macro["silver"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill()
        macro["copper_momentum"] = macro["copper"].pct_change(20)
        macro["sp500_momentum"] = macro["sp500"].pct_change(20)
        macro["vix_pct"] = macro["vix"].pct_change() * 100.0
    else:
        for col in ["vix", "gold", "copper", "sp500", "gold_silver_ratio", "copper_momentum", "sp500_momentum", "vix_pct"]:
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
    macro["z_vix"] = rolling_zscore(macro["vix_pct"].fillna(0), z_window)
    macro["z_gold_silver"] = rolling_zscore(macro["gold_silver_ratio"].ffill(), z_window)
    macro["z_copper"] = rolling_zscore(macro["copper_momentum"].fillna(0), z_window)
    macro["z_sp500"] = rolling_zscore(macro["sp500_momentum"].fillna(0), z_window)
    macro["z_breakeven"] = rolling_zscore(macro["breakeven_delta"].fillna(0), z_window)

    macro["sentiment_factor"] = sentiment_weight * macro["z_sentiment"]
    macro["dxy_factor"] = -dollar_weight * macro["z_dxy"]
    macro["rate_factor"] = -rate_weight * macro["z_rate"]

    macro["U_score"] = macro["sentiment_factor"] + macro["dxy_factor"] + macro["rate_factor"]

    extra_boost = (
        -0.12 * macro["z_vix"].fillna(0)        # VIX涨 → 风险资产受压
        + 0.08 * macro["z_gold_silver"].fillna(0)  # 金银比高 → 白银相对低估
        + 0.08 * macro["z_copper"].fillna(0)      # 铜涨 → 工业需求向好
        + 0.05 * macro["z_sp500"].fillna(0)        # 美股涨 → 风险偏好
        + 0.05 * macro["z_breakeven"].fillna(0)    # 通胀预期升 → 白银保值
    )
    macro["U_score"] = (macro["U_score"] + extra_boost.fillna(0)).clip(-5.0, 5.0)

    macro["market_regime"] = _detect_regime(macro, z_window)

    abs_return = macro["silver"].pct_change().abs()
    vol_fast = abs_return.ewm(span=14, min_periods=5).mean()
    vol_slow = abs_return.ewm(span=60, min_periods=20).mean()
    macro["vol_ratio"] = vol_fast.divide(vol_slow.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).clip(0.30, 3.50).fillna(1.0)

    macro["fair_value_deviation"] = _compute_fair_value_deviation(macro)

    return macro


def _detect_regime(macro, z_window):
    close = macro["silver"]
    if close.dropna().empty:
        return pd.Series("unknown", index=macro.index)

    ret = close.pct_change()
    vol = ret.rolling(20).std()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    trend_strength = (ma20 / ma50 - 1.0).abs()

    vol_percentile = vol.rank(pct=True)

    regime = pd.Series("ranging", index=macro.index)
    regime[(trend_strength > 0.02) & (vol_percentile > 0.4)] = "trending"
    regime[(trend_strength > 0.04) & (vol_percentile > 0.6)] = "strong_trending"
    return regime.ffill().fillna("ranging")


def _compute_fair_value_deviation(macro):
    factors = macro[["z_sentiment", "z_dxy", "z_rate", "z_vix"]].fillna(0)
    composite = (
        0.30 * factors["z_sentiment"]
        - 0.25 * factors["z_dxy"]
        - 0.25 * factors["z_rate"]
        - 0.10 * factors["z_vix"]
        - 0.10 * macro.get("z_gold_silver", pd.Series(0, index=macro.index)).fillna(0)
    )
    z_composite = (composite - composite.rolling(60, min_periods=20).mean()) / composite.rolling(60, min_periods=20).std().replace(0.0, np.nan)
    return z_composite.clip(-3.0, 3.0).fillna(0.0)
