
from typing import Iterable

import streamlit as st

from config import APP_NAME


def sidebar_controls() -> dict[str, bool | float | int | str]:
    st.sidebar.title(APP_NAME)
    st.sidebar.caption("本地白银宏观情绪卡尔曼交易面板。")

    period_labels = {"6mo": "6 个月", "1y": "1 年", "2y": "2 年", "5y": "5 年", "10y": "10 年"}
    interval_labels = {"1d": "日线", "1h": "小时线"}
    period = st.sidebar.selectbox(
        "历史窗口",
        ["6mo", "1y", "2y", "5y", "10y"],
        index=2,
        format_func=lambda value: period_labels[value],
    )
    interval = st.sidebar.selectbox(
        "行情周期",
        ["1d", "1h"],
        index=0,
        format_func=lambda value: interval_labels[value],
    )
    z_window = st.sidebar.slider("Z-Score 滚动窗口", 20, 180, 60, 5)

    st.sidebar.divider()
    rho = st.sidebar.slider("Rho：动量衰减", 0.00, 1.00, 0.86, 0.01)
    alpha = st.sidebar.slider("Alpha：宏观敏感度", 0.00, 2.00, 0.25, 0.01)
    q = st.sidebar.slider("Q：过程噪声", 0.0001, 5.0000, 0.0800, 0.0001, format="%.4f")
    r = st.sidebar.slider("R：测量噪声", 0.0100, 50.0000, 2.5000, 0.0100, format="%.4f")

    st.sidebar.divider()
    st.sidebar.caption("交易机会阈值")
    long_threshold = st.sidebar.slider("多头 U_score 阈值", -2.00, 2.00, 0.25, 0.05)
    short_threshold = st.sidebar.slider("空头 U_score 阈值", -2.00, 2.00, -0.25, 0.05)
    min_velocity = st.sidebar.slider("最小动量强度", 0.00, 3.00, 0.05, 0.01)
    stop_mult = st.sidebar.slider("止损波动倍数", 0.50, 5.00, 1.50, 0.10)
    reward_risk = st.sidebar.slider("目标 / 风险比", 0.50, 5.00, 2.00, 0.10)

    st.sidebar.divider()
    st.sidebar.caption("🤖 AI 基本面分析")
    deepseek_key = st.sidebar.text_input("DeepSeek API Key", type="password", placeholder="sk-...")
    run_ai = st.sidebar.button("运行 AI 分析", type="secondary")

    st.sidebar.divider()
    auto_refresh = st.sidebar.toggle("自动刷新最新数据", value=False)
    if st.sidebar.button("刷新缓存"):
        st.cache_data.clear()
        st.rerun()

    return {
        "period": period,
        "interval": interval,
        "z_window": z_window,
        "rho": rho,
        "alpha": alpha,
        "q": q,
        "r": r,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "min_velocity": min_velocity,
        "stop_mult": stop_mult,
        "reward_risk": reward_risk,
        "auto_refresh": auto_refresh,
        "deepseek_key": deepseek_key,
        "run_ai": run_ai,
    }


def render_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        st.warning(warning)
