from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import DOLLAR_WEIGHT, DXY_TICKER, FRED_REAL_RATE, RATE_WEIGHT, SENTIMENT_WEIGHT, TNX_TICKER, APP_NAME
from data.sentiment import format_news_table
from signals.signal_builder import current_trade_plan
from ui.charts import (
    build_contribution_chart,
    build_extra_charts,
    build_factor_bar_line_chart,
    build_macro_chart,
    build_price_chart,
    build_velocity_chart,
    build_regime_chart,
)


def render_trade_tab(filtered, macro, signal_frame, opportunities, latest_quote, ai_result=None):
    st.warning("本页面只输出量化策略信号，不构成投资建议，也不会自动下单。")

    ai_dir = ai_result.get("direction", "") if ai_result else ""
    plan = current_trade_plan(signal_frame, latest_quote, ai_direction=ai_dir)
    latest = signal_frame.iloc[-1]

    if plan["tone"] == "long":
        st.success(f"当前信号：{plan['方向']}。{plan['动作']}。")
    elif plan["tone"] == "short":
        st.error(f"当前信号：{plan['方向']}。{plan['动作']}。")
    elif plan["tone"] == "watch":
        st.info(f"当前信号：{plan['方向']}。{plan['动作']}。")
    else:
        st.info(f"当前信号：{plan['方向']}。{plan['动作']}。")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("建议状态", str(plan["方向"]))
    c2.metric("信号强度", f"{float(plan['信号强度']):.0f}%")
    c3.metric("市场状态", str(plan.get("regime", "N/A")))
    stop_text = "等待确认" if pd.isna(plan["参考止损"]) else f"{float(plan['参考止损']):,.2f}"
    target_text = "等待确认" if pd.isna(plan["第一目标"]) else f"{float(plan['第一目标']):,.2f}"
    c4.metric("参考止损", stop_text)
    c5.metric("第一目标", target_text)

    fv = float(plan.get("fv_deviation", 0))
    st.markdown(
        f"""
        **触发规则**

        多头入场窗口：动量 > 阈值 + U_score >= 多头阈值 + 价格站上滤波线。

        空头入场窗口：动量 < -阈值 + U_score <= 空头阈值 + 价格跌破滤波线。

        震荡过滤：ADX < 20 或公允值偏离 > 2.5σ 时强制观望。

        当前：价格偏离滤波 **{latest['price_bias_pct']:.2f}%** | 波动缓冲 **{latest['vol_buffer']:.3f}** | ADX **{latest.get('adx', np.nan):.2f}** | 公允值偏离 **{fv:+.2f}σ**
        """
    )

    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="trade_price_chart")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="trade_velocity_chart")

    st.subheader("最近多空入场机会")
    if opportunities.empty:
        st.info("当前历史窗口内没有满足完整条件的多空入场机会。")
    else:
        st.dataframe(
            opportunities.style.format({
                "信号强度": "{:.0f}%", "参考止损": "{:.2f}", "第一目标": "{:.2f}",
                "U_score": "{:.2f}", "动量": "{:.3f}", "ADX": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.plotly_chart(build_macro_chart(macro.loc[filtered.index]), use_container_width=True, key="trade_macro_chart")


def render_sentiment_tab(news, sentiment_daily, macro):
    st.subheader("情绪因子：新闻标题 → VADER → 日均值 → Z-Score")
    chart = macro.tail(180).copy()
    fig = build_factor_bar_line_chart(chart, bar_col="sentiment_factor", line_col="sentiment_raw",
                                       title="情绪贡献与原始新闻情绪", bar_name="情绪贡献", line_name="原始情绪分")
    st.plotly_chart(fig, use_container_width=True, key="sentiment_factor_chart")

    table = chart[["sentiment_raw", "z_sentiment", "sentiment_factor"]].tail(20).rename(
        columns={"sentiment_raw": "原始情绪分", "z_sentiment": "情绪 Z-Score", "sentiment_factor": "情绪贡献"})
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

    st.subheader("相关新闻")
    news_table = format_news_table(news, limit=30)
    if news_table.empty:
        st.info("当前 RSS 源没有返回匹配新闻。")
    else:
        st.dataframe(news_table.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)


def render_dollar_tab(macro):
    st.subheader("美元指数因子：美元走强通常压制白银")
    fig = build_factor_bar_line_chart(macro, bar_col="dxy_factor", line_col="dxy",
                                       title="美元指数贡献与 DXY 水平", bar_name="美元指数贡献", line_name="DXY")
    st.plotly_chart(fig, use_container_width=True, key="dollar_factor_chart")
    table = macro[["dxy", "dxy_delta", "z_dxy", "dxy_factor"]].tail(30).rename(
        columns={"dxy": "美元指数", "dxy_delta": "美元指数变动%", "z_dxy": "美元指数 Z-Score", "dxy_factor": "美元指数贡献"})
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_rate_tab(macro):
    st.subheader("利率因子：实际利率或收益率上行通常压制白银")
    fig = build_factor_bar_line_chart(macro, bar_col="rate_factor", line_col="real_rate",
                                       title="利率贡献与十年期实际利率", bar_name="利率贡献", line_name="实际利率")
    st.plotly_chart(fig, use_container_width=True, key="rate_factor_chart")
    table = macro[["real_rate", "tnx", "rate_delta", "z_rate", "rate_factor"]].tail(30).rename(
        columns={"real_rate": "十年期实际利率", "tnx": "十年期美债收益率", "rate_delta": "利率变动", "z_rate": "利率 Z-Score", "rate_factor": "利率贡献"})
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_composite_tab(macro):
    st.subheader("宏观综合分 U_score + 市场状态检测")
    st.markdown("`U_score = 情绪贡献 + 美元贡献 + 利率贡献 + 增强因子`，同时检测市场状态(震荡/趋势/强趋势)和公允值偏离。")
    st.plotly_chart(build_contribution_chart(macro), use_container_width=True, key="composite_contribution_chart")
    st.plotly_chart(build_regime_chart(macro), use_container_width=True, key="regime_chart")

    cols = ["sentiment_factor", "dxy_factor", "rate_factor", "U_score", "sentiment_raw", "dxy_delta", "rate_delta", "market_regime"]
    table = macro[cols].tail(30).rename(columns={
        "sentiment_factor": "情绪贡献", "dxy_factor": "美元贡献", "rate_factor": "利率贡献",
        "U_score": "宏观总分", "sentiment_raw": "原始情绪", "dxy_delta": "美元变动%", "rate_delta": "利率变动", "market_regime": "市场状态"})
    st.dataframe(table.style.format({k: "{:.3f}" for k in table.columns if k != "市场状态"}), use_container_width=True)


def render_ai_tab(ai_result, macro, filtered, api_key=""):
    st.subheader("🤖 AI 基本面实时分析")

    if not api_key:
        st.info("💡 配置 DeepSeek API Key 可获得 AI 驱动的实时基本面分析。")
        with st.expander("如何获取 API Key？"):
            st.markdown("""
            1. 访问 [platform.deepseek.com](https://platform.deepseek.com)
            2. 注册并登录
            3. 在 API Keys 页面创建新 Key
            4. 复制 Key 填入侧边栏即可
            """)
    if ai_result is None:
        st.warning("AI 分析暂不可用（未配置 API Key）。当前使用本地多因子模型。")
        return

    source = ai_result.get("source", "N/A")
    st.caption(f"分析来源：{source}")

    direction = ai_result.get("direction", "neutral")
    if direction == "bullish":
        st.success("📈 AI 判断：**偏多**")
    elif direction == "bearish":
        st.error("📉 AI 判断：**偏空**")
    else:
        st.info("➡️ AI 判断：**中性**")

    c1, c2, c3 = st.columns(3)
    confidence = ai_result.get("confidence", 0.5)
    c1.metric("置信度", f"{confidence:.0%}")
    c2.metric("基本面分数", f"{ai_result.get('score', 0):+d}")
    c3.metric("技术面参考", f"{float(filtered.iloc[-1]['velocity']):.3f}")

    st.subheader("关键驱动因素")
    st.info(ai_result.get("key_drivers", "N/A"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("⚠️ 风险因素")
        st.warning(ai_result.get("risk_factors", "N/A"))
    with c2:
        st.subheader("🔮 短期展望")
        st.info(ai_result.get("short_term_view", "N/A"))

    st.subheader("🏛️ 宏观评估")
    st.markdown(ai_result.get("macro_assessment", "N/A"))


def render_extra_tab(macro):
    st.subheader("增强数据因子")
    extra_charts = build_extra_charts(macro)
    for col_name, fig in extra_charts.items():
        st.plotly_chart(fig, use_container_width=True, key=f"extra_{col_name}_chart")

    table_cols = [c for c in ["vix", "gold", "copper", "sp500", "gold_silver_ratio", "breakeven",
                                "z_vix", "z_gold_silver", "z_copper", "z_sp500"] if c in macro.columns]
    if table_cols:
        rename_map = {"vix": "VIX", "gold": "黄金", "copper": "铜", "sp500": "标普500",
                      "gold_silver_ratio": "金银比", "breakeven": "T10YIE",
                      "z_vix": "VIX Z", "z_gold_silver": "金银比 Z", "z_copper": "铜 Z", "z_sp500": "标普 Z"}
        table = macro[table_cols].tail(30).rename(columns=rename_map)
        st.dataframe(table.style.format("{:.2f}"), use_container_width=True)


def render_kalman_tab(filtered, signal_frame, controls):
    st.subheader("卡尔曼动量因子")
    st.markdown(f"Rho={float(controls['rho']):.2f} Alpha={float(controls['alpha']):.2f} Q={float(controls['q']):.4f} R={float(controls['r']):.4f}")

    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="kalman_price_chart")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="kalman_velocity_chart")

    table_cols = ["observed", "kalman_price", "price_bias_pct", "velocity", "adx", "U_score", "entry_side", "long_score_pct", "short_score_pct"]
    table = signal_frame[table_cols].tail(30).rename(columns={
        "observed": "实际价格", "kalman_price": "卡尔曼价格", "price_bias_pct": "偏离%",
        "velocity": "隐藏动量", "adx": "ADX", "U_score": "宏观总分", "entry_side": "信号状态",
        "long_score_pct": "多头强度", "short_score_pct": "空头强度"})
    st.dataframe(table.style.format({"实际价格": "{:.2f}", "卡尔曼价格": "{:.2f}", "偏离%": "{:.2f}",
                                      "隐藏动量": "{:.3f}", "ADX": "{:.2f}", "宏观总分": "{:.2f}",
                                      "多头强度": "{:.0f}%", "空头强度": "{:.0f}%"}), use_container_width=True)


def render_raw_data_tab(market, real_rate, extra_data, macro, news):
    st.subheader("行情与宏观原始数据")
    st.dataframe(market.tail(80), use_container_width=True)
    st.subheader("FRED 利率数据")
    if real_rate.empty:
        st.info("FRED 数据为空。")
    else:
        st.dataframe(real_rate.tail(80), use_container_width=True)
    if extra_data is not None and not extra_data.empty:
        st.subheader("增强品种数据")
        st.dataframe(extra_data.tail(80), use_container_width=True)
    st.subheader("宏观计算结果")
    st.dataframe(macro.tail(80), use_container_width=True)
    st.subheader("新闻原始抓取结果")
    news_table = format_news_table(news, limit=80)
    if news_table.empty:
        st.info("新闻数据为空。")
    else:
        st.dataframe(news_table.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)
