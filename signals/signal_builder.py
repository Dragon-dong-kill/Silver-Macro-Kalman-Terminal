import numpy as np
import pandas as pd


def build_signal_frame(filtered, macro, adx, long_threshold, short_threshold,
                        min_velocity, stop_mult, reward_risk, ai_score=0, adx_threshold=20.0):
    factor_columns = [
        "sentiment_factor", "dxy_factor", "rate_factor",
        "z_sentiment", "z_dxy", "z_rate", "z_vix", "z_gold_silver", "z_copper", "z_sp500",
        "vix", "gold", "copper", "sp500", "gold_silver_ratio",
        "breakeven", "market_regime", "fair_value_deviation",
        "silver_momentum_5", "silver_momentum_20",
        "bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_position",
        "vol_ratio",
    ]
    available = [c for c in factor_columns if c in macro.columns]
    data = filtered.join(macro[available], how="left").copy()

    data["adx"] = pd.to_numeric(adx.reindex(data.index), errors="coerce")
    data["price_bias"] = data["observed"] - data["kalman_price"]
    data["price_bias_pct"] = data["price_bias"] / data["kalman_price"].replace(0.0, np.nan) * 100.0

    buf1 = data["observed"].diff().abs().rolling(14, min_periods=5).mean()
    buf2 = data["observed"] * data["observed"].pct_change().abs().rolling(14, min_periods=5).mean()
    data["vol_buffer"] = buf1.combine_first(buf2).replace([np.inf, -np.inf, 0.0], np.nan).ffill().bfill().fillna(data["observed"] * 0.01)

    regime = data.get("market_regime", pd.Series("ranging", index=data.index))
    fv = data.get("fair_value_deviation", pd.Series(0, index=data.index))
    bb_pos = data.get("bb_position", pd.Series(0, index=data.index))
    bb_width = data.get("bb_width", pd.Series(2.0, index=data.index))
    vol_ratio = data.get("vol_ratio", pd.Series(1.0, index=data.index))

    data["is_ranging"] = regime == "ranging"
    data["is_trending"] = regime.isin(["trending", "strong_trending"])
    data["is_strong"] = regime == "strong_trending"

    long_votes = (
        (data["velocity"] > min_velocity).astype(int)
        + (data["U_score"] >= long_threshold).astype(int)
        + (data["price_bias"] >= 0.0).astype(int)
        + (data["sentiment_factor"] >= 0.0).astype(int)
        + (data["dxy_factor"] >= 0.0).astype(int)
        + (data["rate_factor"] >= 0.0).astype(int)
    )
    short_votes = (
        (data["velocity"] < -min_velocity).astype(int)
        + (data["U_score"] <= short_threshold).astype(int)
        + (data["price_bias"] <= 0.0).astype(int)
        + (data["sentiment_factor"] <= 0.0).astype(int)
        + (data["dxy_factor"] <= 0.0).astype(int)
        + (data["rate_factor"] <= 0.0).astype(int)
    )

    data["ai_score"] = ai_score
    data["ai_direction"] = 1 if ai_score > 15 else (-1 if ai_score < -15 else 0)
    data["ai_vote"] = 0
    if ai_score > 15:
        long_votes += 1
        data["ai_vote"] = 1
    elif ai_score < -15:
        short_votes += 1
        data["ai_vote"] = -1

    data["long_vote_count"] = long_votes
    data["short_vote_count"] = short_votes
    data["long_score_pct"] = (long_votes / 7.0 * 100.0).clip(0, 100)
    data["short_score_pct"] = (short_votes / 7.0 * 100.0).clip(0, 100)

    data["long_setup"] = False
    data["short_setup"] = False
    data["strategy_type"] = np.nan

    _build_ranging_signals(data, fv, bb_pos, bb_width, vol_ratio)
    _build_trend_signals(data, min_velocity, long_threshold, short_threshold, adx_threshold, vol_ratio)
    _build_strong_trend_signals(data, min_velocity, long_threshold, short_threshold, vol_ratio)

    _deduplicate(data)

    _set_labels(data)

    data["stop_mult"] = stop_mult
    data["reward_risk"] = reward_risk
    return data


def _build_ranging_signals(data, fv, bb_pos, bb_width, vol_ratio):
    valid_range = bb_width > 0.8
    valid_vol = vol_ratio < 3.0

    data["ranging_long"] = (
        data["is_ranging"]
        & valid_range
        & valid_vol
        & (bb_pos < -2.0)
        & (data["velocity"] > -0.08)
    )
    data["ranging_short"] = (
        data["is_ranging"]
        & valid_range
        & valid_vol
        & (bb_pos > 2.0)
        & (data["velocity"] < 0.08)
    )
    data["ranging_long_exit"] = data["is_ranging"] & (bb_pos > -0.3)
    data["ranging_short_exit"] = data["is_ranging"] & (bb_pos < 0.3)
    data["ranging_long_stop_exit"] = data["is_ranging"] & (bb_pos < -3.0)
    data["ranging_short_stop_exit"] = data["is_ranging"] & (bb_pos > 3.0)

    data.loc[data["ranging_long"], "long_setup"] = True
    data.loc[data["ranging_long"], "strategy_type"] = "布林带-均值回归做多"
    data.loc[data["ranging_short"], "short_setup"] = True
    data.loc[data["ranging_short"], "strategy_type"] = "布林带-均值回归做空"


def _build_trend_signals(data, min_vel, lt, st, adx_t, vol_ratio):
    valid_adx = data["adx"].notna() & (data["adx"] >= float(adx_t))
    valid_vol = vol_ratio < 3.5

    data["trending_long"] = (
        data["is_trending"]
        & valid_vol
        & (data["velocity"] > min_vel * 1.2)
        & (data["U_score"] >= lt - 0.1)
        & (valid_adx | (data["velocity"] > min_vel * 1.8))
        & (data["long_score_pct"] >= 57)
    )
    data["trending_short"] = (
        data["is_trending"]
        & valid_vol
        & (data["velocity"] < -min_vel * 1.2)
        & (data["U_score"] <= st + 0.1)
        & (valid_adx | (data["velocity"] < -min_vel * 1.8))
        & (data["short_score_pct"] >= 57)
    )

    data.loc[data["trending_long"], "long_setup"] = True
    data.loc[data["trending_long"], "strategy_type"] = "趋势-动量做多"
    data.loc[data["trending_short"], "short_setup"] = True
    data.loc[data["trending_short"], "strategy_type"] = "趋势-动量做空"


def _build_strong_trend_signals(data, min_vel, lt, st, vol_ratio):
    valid_vol = vol_ratio < 4.0

    data["strong_long"] = (
        data["is_strong"]
        & valid_vol
        & (data["velocity"] > min_vel * 0.6)
        & (data["U_score"] >= lt - 0.3)
        & (data["long_score_pct"] >= 43)
    )
    data["strong_short"] = (
        data["is_strong"]
        & valid_vol
        & (data["velocity"] < -min_vel * 0.6)
        & (data["U_score"] <= st + 0.3)
        & (data["short_score_pct"] >= 43)
    )

    data.loc[data["strong_long"] & ~data["long_setup"], "long_setup"] = True
    data.loc[data["strong_long"] & (data["strategy_type"].isna() | (data["strategy_type"] == "")), "strategy_type"] = "强趋势-做多"
    data.loc[data["strong_short"] & ~data["short_setup"], "short_setup"] = True
    data.loc[data["strong_short"] & (data["strategy_type"].isna() | (data["strategy_type"] == "")), "strategy_type"] = "强趋势-做空"


def _deduplicate(data):
    for col in ["long_setup", "short_setup"]:
        s = data[col].shift(1).fillna(False)
        data.loc[s & data[col], col] = False


def _set_labels(data):
    conds = [
        data["long_setup"], data["short_setup"],
        data["long_score_pct"] >= 71, data["short_score_pct"] >= 71,
        data["long_score_pct"] >= 57, data["short_score_pct"] >= 57,
    ]
    choices = [
        "多头入场", "空头入场",
        "多头强观察", "空头强观察",
        "多头弱观察", "空头弱观察",
    ]
    data["entry_side"] = np.select(conds, choices, default="观望")
    data["strategy_type"] = data["strategy_type"].fillna("")


def build_opportunity_table(signal_frame, limit=20):
    rows = []
    cand = signal_frame[signal_frame["long_setup"] | signal_frame["short_setup"]].tail(limit)
    for ts, row in cand.iterrows():
        is_l = bool(row["long_setup"])
        side = "做多" if is_l else "做空"
        buf = max(float(row["vol_buffer"]), float(row["observed"]) * 0.002)
        el = float(row["observed"]) - buf * 0.25
        eh = float(row["observed"]) + buf * 0.25
        sd = buf * float(row["stop_mult"])
        td = sd * float(row["reward_risk"])
        sp = float(row["observed"]) - sd if is_l else float(row["observed"]) + sd
        tp = float(row["observed"]) + td if is_l else float(row["observed"]) - td
        st = float(row["long_score_pct"] if is_l else row["short_score_pct"])
        sty = str(row.get("strategy_type", ""))

        rows.append({
            "时间": ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
            "方向": side,
            "信号强度": st,
            "策略": sty,
            "参考入场": "{:,.2f}-{:,.2f}".format(el, eh),
            "止损": sp,
            "目标": tp,
            "U_score": float(row["U_score"]),
            "动量": float(row["velocity"]),
            "ADX": float(row["adx"]) if pd.notna(row.get("adx")) else np.nan,
            "布林带位置": float(row.get("bb_position", 0)),
        })
    return pd.DataFrame(rows)


def current_trade_plan(signal_frame, latest_quote, ai_direction=""):
    row = signal_frame.iloc[-1]
    buf = max(float(row["vol_buffer"]), float(latest_quote) * 0.002)
    el = latest_quote - buf * 0.25
    eh = latest_quote + buf * 0.25
    regime = str(row.get("market_regime", "N/A"))
    fv_dev = float(row.get("fair_value_deviation", 0))
    sty = str(row.get("strategy_type", ""))
    bb_p = float(row.get("bb_position", 0))

    if bool(row.get("long_setup", False)):
        side = "多头窗口 [{}]".format(regime)
        action = sty
        sv = latest_quote - buf * float(row["stop_mult"])
        tv = latest_quote + buf * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["long_score_pct"])
        tone = "long"
    elif bool(row.get("short_setup", False)):
        side = "空头窗口 [{}]".format(regime)
        action = sty
        sv = latest_quote + buf * float(row["stop_mult"])
        tv = latest_quote - buf * float(row["stop_mult"]) * float(row["reward_risk"])
        strength = float(row["short_score_pct"])
        tone = "short"
    elif float(row["long_score_pct"]) >= 57:
        side = "多头观察"
        action = "布林带位置={:+.1f}σ".format(bb_p)
        sv = np.nan
        tv = np.nan
        strength = float(row["long_score_pct"])
        tone = "watch"
    elif float(row["short_score_pct"]) >= 57:
        side = "空头观察"
        action = "布林带位置={:+.1f}σ".format(bb_p)
        sv = np.nan
        tv = np.nan
        strength = float(row["short_score_pct"])
        tone = "watch"
    else:
        side = "观望"
        action = "信号强度不足 | 布林带={:+.1f}σ".format(bb_p)
        sv = np.nan
        tv = np.nan
        strength = 0.0
        tone = "flat"

    return {
        "方向": side, "动作": action,
        "入场区间": "{:,.2f} - {:,.2f}".format(el, eh),
        "参考止损": sv, "第一目标": tv,
        "信号强度": strength, "tone": tone,
        "regime": regime, "fv_deviation": fv_dev, "bb_position": bb_p,
    }
