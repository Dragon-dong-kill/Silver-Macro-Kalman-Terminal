from __future__ import annotations

import numpy as np
import pandas as pd


def build_signal_frame(
    filtered: pd.DataFrame,
    macro: pd.DataFrame,
    adx: pd.Series,
    long_threshold: float,
    short_threshold: float,
    min_velocity: float,
    stop_mult: float,
    reward_risk: float,
    adx_threshold: float = 20.0,
) -> pd.DataFrame:
    factor_columns = [
        "sentiment_factor",
        "dxy_factor",
        "rate_factor",
        "sentiment_raw",
        "z_sentiment",
        "dxy",
        "dxy_delta",
        "z_dxy",
        "real_rate",
        "tnx",
        "rate_delta",
        "z_rate",
        "z_vix",
        "z_gold_silver",
        "z_copper",
        "z_sp500",
        "vix",
        "gold",
        "copper",
        "sp500",
        "gold_silver_ratio",
        "breakeven",
    ]
    available_factors = [col for col in factor_columns if col in macro.columns]
    data = filtered.join(macro[available_factors], how="left").copy()
    data["adx"] = pd.to_numeric(adx.reindex(data.index), errors="coerce")
    data["price_bias"] = data["observed"] - data["kalman_price"]
    data["price_bias_pct"] = data["price_bias"] / data["kalman_price"].replace(0.0, np.nan) * 100.0

    close_buffer = data["observed"].diff().abs().rolling(14, min_periods=5).mean()
    pct_buffer = data["observed"] * data["observed"].pct_change().abs().rolling(14, min_periods=5).mean()
    data["vol_buffer"] = (
        close_buffer.combine_first(pct_buffer)
        .replace([np.inf, -np.inf, 0.0], np.nan)
        .ffill()
        .bfill()
        .fillna(data["observed"] * 0.01)
    )
    data["recent_buy_cross"] = data["buy_signal"].rolling(3, min_periods=1).max().astype(bool)
    data["recent_sell_cross"] = data["sell_signal"].rolling(3, min_periods=1).max().astype(bool)

    data["long_vote_count"] = (
        (data["velocity"] > min_velocity).astype(int)
        + (data["U_score"] >= long_threshold).astype(int)
        + (data["price_bias"] >= 0.0).astype(int)
        + (data["sentiment_factor"] >= 0.0).astype(int)
        + (data["dxy_factor"] >= 0.0).astype(int)
        + (data["rate_factor"] >= 0.0).astype(int)
    )
    data["short_vote_count"] = (
        (data["velocity"] < -min_velocity).astype(int)
        + (data["U_score"] <= short_threshold).astype(int)
        + (data["price_bias"] <= 0.0).astype(int)
        + (data["sentiment_factor"] <= 0.0).astype(int)
        + (data["dxy_factor"] <= 0.0).astype(int)
        + (data["rate_factor"] <= 0.0).astype(int)
    )
    data["long_score_pct"] = data["long_vote_count"] / 6.0 * 100.0
    data["short_score_pct"] = data["short_vote_count"] / 6.0 * 100.0

    data["long_setup"] = (
        (data["velocity"] > min_velocity)
        & (data["U_score"] >= long_threshold)
        & (data["price_bias"] >= 0.0)
    )
    data["short_setup"] = (
        (data["velocity"] < -min_velocity)
        & (data["U_score"] <= short_threshold)
        & (data["price_bias"] <= 0.0)
    )
    data["adx_range_filter"] = data["adx"].notna() & (data["adx"] < float(adx_threshold))
    data.loc[data["adx_range_filter"], "long_setup"] = False
    data.loc[data["adx_range_filter"], "short_setup"] = False

    data["entry_side"] = np.select(
        [
            data["adx_range_filter"],
            data["long_setup"],
            data["short_setup"],
            data["long_vote_count"] >= 4,
            data["short_vote_count"] >= 4,
        ],
        ["震荡市观望(ADX<20)", "多头入场窗口", "空头入场窗口", "多头观察", "空头观察"],
        default="观望",
    )
    data["stop_mult"] = stop_mult
    data["reward_risk"] = reward_risk
    return data


def build_opportunity_table(signal_frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = signal_frame[signal_frame["long_setup"] | signal_frame["short_setup"]].tail(limit)

    for timestamp, row in candidates.iterrows():
        is_long = bool(row["long_setup"])
        side = "做多" if is_long else "做空"
        buffer = max(float(row["vol_buffer"]), float(row["observed"]) * 0.002)
        entry_low = float(row["observed"]) - buffer * 0.25
        entry_high = float(row["observed"]) + buffer * 0.25
        stop_distance = buffer * float(row["stop_mult"])
        target_distance = stop_distance * float(row["reward_risk"])
        stop = float(row["observed"]) - stop_distance if is_long else float(row["observed"]) + stop_distance
        target = float(row["observed"]) + target_distance if is_long else float(row["observed"]) - target_distance
        strength = float(row["long_score_pct"] if is_long else row["short_score_pct"])
        trigger = (
            "动量转强 / 宏观顺风 / 价格站上滤波线"
            if is_long
            else "动量转弱 / 宏观逆风 / 价格跌破滤波线"
        )

        rows.append(
            {
                "时间": timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp),
                "方向": side,
                "信号强度": strength,
                "参考入场区间": f"{entry_low:,.2f} - {entry_high:,.2f}",
                "参考止损": stop,
                "第一目标": target,
                "U_score": float(row["U_score"]),
                "动量": float(row["velocity"]),
                "ADX": float(row["adx"]) if pd.notna(row.get("adx")) else np.nan,
                "触发逻辑": trigger,
            }
        )

    return pd.DataFrame(rows)


def current_trade_plan(signal_frame: pd.DataFrame, latest_quote: float) -> dict[str, object]:
    row = signal_frame.iloc[-1]
    buffer = max(float(row["vol_buffer"]), float(latest_quote) * 0.002)
    entry_low = latest_quote - buffer * 0.25
    entry_high = latest_quote + buffer * 0.25

    if bool(row.get("adx_range_filter", False)):
        side = "震荡市观望(ADX<20)"
        action = "趋势强度不足，暂停多空入场"
        stop = np.nan
        target = np.nan
        strength = max(float(row["long_score_pct"]), float(row["short_score_pct"]))
        tone = "flat"
    elif bool(row["long_setup"]):
        side = "多头入场窗口"
        action = "可关注回踩不破后的多头入场"
        stop = latest_quote - buffer * float(row["stop_mult"])
        target = latest_quote + buffer * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["long_score_pct"])
        tone = "long"
    elif bool(row["short_setup"]):
        side = "空头入场窗口"
        action = "可关注反抽不过后的空头入场"
        stop = latest_quote + buffer * float(row["stop_mult"])
        target = latest_quote - buffer * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["short_score_pct"])
        tone = "short"
    elif float(row["long_score_pct"]) > float(row["short_score_pct"]):
        side = "多头观察"
        action = "等待 U_score 或动量进一步确认"
        stop = np.nan
        target = np.nan
        strength = float(row["long_score_pct"])
        tone = "watch"
    elif float(row["short_score_pct"]) > float(row["long_score_pct"]):
        side = "空头观察"
        action = "等待 U_score 或动量进一步确认"
        stop = np.nan
        target = np.nan
        strength = float(row["short_score_pct"])
        tone = "watch"
    else:
        side = "观望"
        action = "多空条件不充分，避免追单"
        stop = np.nan
        target = np.nan
        strength = 0.0
        tone = "flat"

    return {
        "方向": side,
        "动作": action,
        "入场区间": f"{entry_low:,.2f} - {entry_high:,.2f}",
        "参考止损": stop,
        "第一目标": target,
        "信号强度": strength,
        "tone": tone,
    }
