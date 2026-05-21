from __future__ import annotations

from datetime import timedelta

import numpy as np
import streamlit as st

from config import APP_NAME, FRED_BREAKEVEN, FRED_REAL_RATE
from data.fetcher import (
    fetch_extra_data,
    fetch_fred_data,
    fetch_latest_silver_price,
    fetch_market_data,
    fetch_silver_ohlc,
    resolve_data_range,
)
from data.indicators import calculate_adx
from data.sentiment import daily_sentiment, fetch_news_sentiment, format_news_table
from kalman.filter import run_kalman
from signals.macro_score import compute_macro_score
from signals.signal_builder import build_opportunity_table, build_signal_frame, current_trade_plan
from ui.backtest_tab import render_backtest_tab
from ui.sidebar import render_warnings, sidebar_controls
from ui.tabs import (
    render_composite_tab,
    render_dollar_tab,
    render_extra_tab,
    render_kalman_tab,
    render_rate_tab,
    render_raw_data_tab,
    render_sentiment_tab,
    render_trade_tab,
)


def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} | 白银宏观卡尔曼交易终端", page_icon="Ag", layout="wide")
    controls = sidebar_controls()

    if controls["auto_refresh"]:
        st.sidebar.caption("刷新浏览器页面即可重新运行；实时报价缓存 60 秒。")

    st.title("白银宏观卡尔曼交易终端")
    st.caption(
        "以 RSS 情绪、美元指数变动、实际利率或美债收益率变动作为宏观控制输入，"
        "对 SI=F 白银期货价格状态进行二维卡尔曼滤波，并生成多空入场窗口。"
    )

    with st.spinner("正在加载行情数据、FRED 利率、增强品种和 RSS 情绪数据..."):
        market = fetch_market_data(str(controls["period"]), str(controls["interval"]))
        silver_ohlc = fetch_silver_ohlc(str(controls["period"]), str(controls["interval"]))
        latest_price, latest_warning = fetch_latest_silver_price()

        if market.data.empty:
            render_warnings(market.warnings)
            st.stop()

        start, end = resolve_data_range(market.data)
        fred_symbols = [FRED_REAL_RATE, FRED_BREAKEVEN]
        real_rate = fetch_fred_data(start, end, fred_symbols)
        news = fetch_news_sentiment()

        extra_data = fetch_extra_data(str(controls["period"]), str(controls["interval"]))

    all_warnings = (
        market.warnings
        + silver_ohlc.warnings
        + real_rate.warnings
        + news.warnings
        + extra_data.warnings
    )
    if latest_warning:
        all_warnings.append(latest_warning)

    sentiment_daily = daily_sentiment(news.data)
    macro = compute_macro_score(
        closes=market.data,
        real_rate=real_rate.data,
        sentiment=sentiment_daily,
        extra_data=extra_data.data if not extra_data.data.empty else None,
        z_window=int(controls["z_window"]),
    )
    filtered = run_kalman(
        price=macro["silver"],
        u_score=macro["U_score"],
        vol_ratio=macro["vol_ratio"],
        rho=float(controls["rho"]),
        alpha=float(controls["alpha"]),
        q=float(controls["q"]),
        r=float(controls["r"]),
    )
    adx_series = calculate_adx(silver_ohlc.data, period=14)

    if filtered.empty:
        st.error("数据清洗后没有可用的 SI=F 价格序列。")
        render_warnings(all_warnings)
        st.stop()

    latest_row = filtered.iloc[-1]
    latest_quote = latest_price if latest_price is not None else latest_row["observed"]
    signal_frame = build_signal_frame(
        filtered=filtered,
        macro=macro,
        adx=adx_series,
        long_threshold=float(controls["long_threshold"]),
        short_threshold=float(controls["short_threshold"]),
        min_velocity=float(controls["min_velocity"]),
        stop_mult=float(controls["stop_mult"]),
        reward_risk=float(controls["reward_risk"]),
    )
    opportunities = build_opportunity_table(signal_frame)
    current_plan = current_trade_plan(signal_frame, float(latest_quote))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("SI=F 最新报价", f"{latest_quote:,.2f}")
    c2.metric("当前信号", str(current_plan["方向"]))
    c3.metric("信号强度", f"{float(current_plan['信号强度']):.0f}%")
    c4.metric("卡尔曼价格", f"{latest_row['kalman_price']:,.2f}")
    c5.metric("隐藏动量", f"{latest_row['velocity']:,.3f}")
    c6.metric("宏观总分 U_score", f"{latest_row['U_score']:,.2f}")

    if all_warnings:
        with st.expander("数据源告警", expanded=False):
            render_warnings(all_warnings)

    st.subheader("相关新闻")
    home_news = format_news_table(news.data, limit=12)
    if home_news.empty:
        st.info("当前没有可展示的相关新闻。")
    else:
        st.dataframe(
            home_news.style.format({"情绪分": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

    tabs = st.tabs(
        [
            "交易机会",
            "情绪因子",
            "美元指数因子",
            "利率因子",
            "综合分拆解",
            "增强数据",
            "卡尔曼动量",
            "回测分析",
            "原始数据",
        ]
    )
    with tabs[0]:
        render_trade_tab(filtered, macro, signal_frame, opportunities, float(latest_quote))
    with tabs[1]:
        render_sentiment_tab(news.data, sentiment_daily, macro)
    with tabs[2]:
        render_dollar_tab(macro)
    with tabs[3]:
        render_rate_tab(macro)
    with tabs[4]:
        render_composite_tab(macro)
    with tabs[5]:
        render_extra_tab(macro)
    with tabs[6]:
        render_kalman_tab(filtered, signal_frame, controls)
    with tabs[7]:
        render_backtest_tab(signal_frame, controls)
    with tabs[8]:
        render_raw_data_tab(market.data, real_rate.data, extra_data.data if not extra_data.data.empty else None, macro, news.data)


if __name__ == "__main__":
    main()
