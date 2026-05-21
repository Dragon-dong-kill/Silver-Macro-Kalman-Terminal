from __future__ import annotations

import numpy as np
import pandas as pd


def build_signal_frame(
    filtered,
    macro,
    adx,
    long_threshold,
    short_threshold,
    min_velocity,
    stop_mult,
    reward_risk,
    ai_score=0,
    adx_threshold=20.0,
):
    factor_columns = [
        "sentiment_factor", "dxy_factor", "rate_factor",
        "sentiment_raw", "z_sentiment", "dxy", "dxy_delta", "z_dxy",
        "real_rate", "tnx", "rate_delta", "z_rate",
        "z_vix", "z_gold_silver", "z_copper", "z_sp500",
        "vix", "gold", "copper", "sp500", "gold_silver_ratio",
        "breakeven", "market_regime", "fair_value_deviation",
        "silver_momentum_5", "silver_momentum_20",
    ]
    available = [col for col in factor_columns if col in macro.columns]
    data = filtered.join(macro[available], how="left").copy()

    data["adx"] = pd.to_numeric(adx.reindex(data.index), errors="coerce")
    data["price_bias"] = data["observed"] - data["kalman_price"]
    data["price_bias_pct"] = data["price_bias"] / data["kalman_price"].replace(0.0, np.nan) * 100.0

    close_buffer = data["observed"].diff().abs().rolling(14, min_periods=5).mean()
    pct_buffer = data["observed"] * data["observed"].pct_change().abs().rolling(14, min_periods=5).mean()
    data["vol_buffer"] = close_buffer.combine_first(pct_buffer).replace([np.inf, -np.inf, 0.0], np.nan).ffill().bfill().fillna(data["observed"] * 0.01)

    data["ai_score"] = ai_score
    data["ai_direction"] = 1 if ai_score > 15 else (-1 if ai_score < -15 else 0)

    data["regime_is_trending"] = data.get("market_regime", "ranging").isin(["trending", "strong_trending"])
    data["regime_is_ranging"] = data.get("market_regime", "ranging") == "ranging"
    data["is_strong_trend"] = data.get("market_regime", "ranging") == "strong_trending"

    data["fair_value_signal"] = 0
    fv = data.get("fair_value_deviation", pd.Series(0, index=data.index))
    data.loc[fv > 1.0, "fair_value_signal"] = -1
    data.loc[fv < -1.0, "fair_value_signal"] = 1

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

    data["ai_vote"] = 0
    if data["ai_direction"].iloc[-1] > 0:
        data["long_vote_count"] += 1
        data["ai_vote"] = 1
    elif data["ai_direction"].iloc[-1] < 0:
        data["short_vote_count"] += 1
        data["ai_vote"] = -1

    data["long_score_pct"] = data["long_vote_count"] / 7.0 * 100.0
    data["short_score_pct"] = data["short_vote_count"] / 7.0 * 100.0

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
    data["fair_value_filter"] = fv.abs() > 2.5
    data.loc[data["adx_range_filter"], "long_setup"] = False
    data.loc[data["adx_range_filter"], "short_setup"] = False
    data.loc[data["fair_value_filter"], "long_setup"] = False
    data.loc[data["fair_value_filter"], "short_setup"] = False

    trending_mask = data["regime_is_trending"]
    data.loc[trending_mask & (data["velocity"] > min_velocity * 1.5) & (data["U_score"] >= long_threshold - 0.1), "long_setup"] = True
    data.loc[trending_mask & (data["velocity"] < -min_velocity * 1.5) & (data["U_score"] <= short_threshold + 0.1), "short_setup"] = True

    data["entry_side"] = np.select(
        [
            data["adx_range_filter"] | data["fair_value_filter"],
            data["long_setup"],
            data["short_setup"],
            data["long_vote_count"] >= 5,
            data["short_vote_count"] >= 5,
            data["long_vote_count"] >= 3,
            data["short_vote_count"] >= 3,
        ],
        ["观望(低趋势/极端估值)", "多头入场窗口", "空头入场窗口", "多头强势观察", "空头强势观察", "多头弱观察", "空头弱观察"],
        default="观望",
    )
    data["stop_mult"] = stop_mult
    data["reward_risk"] = reward_risk
    return data


def build_opportunity_table(signal_frame, limit=20):
    rows = []
    candidates = signal_frame[signal_frame["long_setup"] | signal_frame["short_setup"]].tail(limit)

    for timestamp, row in candidates.iterrows():
        is_long = bool(row["long_setup"])
        side = "做多" if is_long else "做空"
        buf = max(float(row["vol_buffer"]), float(row["observed"]) * 0.002)
        entry_low = float(row["observed"]) - buf * 0.25
        entry_high = float(row["observed"]) + buf * 0.25
        stop_dist = buf * float(row["stop_mult"])
        target_dist = stop_dist * float(row["reward_risk"])
        stop_price = float(row["observed"]) - stop_dist if is_long else float(row["observed"]) + stop_dist
        target_price = float(row["observed"]) + target_dist if is_long else float(row["observed"]) - target_dist
        strength = float(row["long_score_pct"] if is_long else row["short_score_pct"])
        regime = str(row.get("market_regime", "N/A"))
        ai_v = int(row.get("ai_vote", 0))
        ai_tag = " AI看多" if ai_v > 0 else (" AI看空" if ai_v < 0 else "")

        rows.append({
            "时间": timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp),
            "方向": side,
            "信号强度": strength,
            "市场状态": regime + ai_tag,
            "参考入场区间": f"{entry_low:,.2f} - {entry_high:,.2f}",
            "参考止损": stop_price,
            "第一目标": target_price,
            "U_score": float(row["U_score"]),
            "动量": float(row["velocity"]),
            "ADX": float(row["adx"]) if pd.notna(row.get("adx")) else np.nan,
        })
    return pd.DataFrame(rows)


def current_trade_plan(signal_frame, latest_quote, ai_direction=""):
    row = signal_frame.iloc[-1]
    buf = max(float(row["vol_buffer"]), float(latest_quote) * 0.002)
    entry_low = latest_quote - buf * 0.25
    entry_high = latest_quote + buf * 0.25
    regime = str(row.get("market_regime", "N/A"))
    fv_dev = float(row.get("fair_value_deviation", 0))

    if bool(row.get("adx_range_filter", False)):
        side = "震荡市观望(ADX<20)"
        action = "趋势强度不足，暂停多空入场"
        stop_val = np.nan
        target_val = np.nan
        strength = max(float(row["long_score_pct"]), float(row["short_score_pct"]))
        tone = "flat"
    elif abs(fv_dev) > 2.5:
        side = "极端估值观望"
        action = f"公允值偏离{fv_dev:+.1f}σ，等待回归"
        stop_val = np.nan
        target_val = np.nan
        strength = max(float(row["long_score_pct"]), float(row["short_score_pct"]))
        tone = "flat"
    elif bool(row["long_setup"]):
        side = f"多头入场窗口 [{regime}]"
        action = "可关注回踩不破后的多头入场"
        stop_val = latest_quote - buf * float(row["stop_mult"])
        target_val = latest_quote + buf * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["long_score_pct"])
        tone = "long"
    elif bool(row["short_setup"]):
        side = f"空头入场窗口 [{regime}]"
        action = "可关注反抽不过后的空头入场"
        stop_val = latest_quote + buf * float(row["stop_mult"])
        target_val = latest_quote - buf * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["short_score_pct"])
        tone = "short"
    elif float(row["long_score_pct"]) > float(row["short_score_pct"]) + 10:
        side = "多头观察"
        action = "等待动量或AI确认"
        stop_val = np.nan
        target_val = np.nan
        strength = float(row["long_score_pct"])
        tone = "watch"
    elif float(row["short_score_pct"]) > float(row["long_score_pct"]) + 10:
        side = "空头观察"
        action = "等待动量或AI确认"
        stop_val = np.nan
        target_val = np.nan
        strength = float(row["short_score_pct"])
        tone = "watch"
    else:
        side = "观望"
        action = "多空条件不充分，避免追单"
        stop_val = np.nan
        target_val = np.nan
        strength = 0.0
        tone = "flat"

    if ai_direction == "bullish" and tone in ("flat", "watch"):
        action += " [AI偏向多头]"
    elif ai_direction == "bearish" and tone in ("flat", "watch"):
        action += " [AI偏向空头]"

    return {
        "方向": side,
        "动作": action,
        "入场区间": f"{entry_low:,.2f} - {entry_high:,.2f}",
        "参考止损": stop_val,
        "第一目标": target_val,
        "信号强度": strength,
        "tone": tone,
        "regime": regime,
        "fv_deviation": fv_dev,
    }
