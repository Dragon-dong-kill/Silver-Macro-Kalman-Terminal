from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DOLLAR_WEIGHT,
    DXY_TICKER,
    FRED_REAL_RATE,
    RATE_WEIGHT,
    SENTIMENT_WEIGHT,
    TNX_TICKER,
)
from data.sentiment import format_news_table
from signals.signal_builder import current_trade_plan
from ui.charts import (
    build_contribution_chart,
    build_extra_charts,
    build_factor_bar_line_chart,
    build_macro_chart,
    build_price_chart,
    build_velocity_chart,
)


def render_trade_tab(
    filtered: pd.DataFrame,
    macro: pd.DataFrame,
    signal_frame: pd.DataFrame,
    opportunities: pd.DataFrame,
    latest_quote: float,
) -> None:
    st.warning("本页面只输出量化策略信号，不构成投资建议，也不会自动下单。期货杠杆风险高，入场前需要结合保证金、滑点、合约乘数和账户风险限额。")

    plan = current_trade_plan(signal_frame, latest_quote)
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
    c3.metric("参考入场区间", str(plan["入场区间"]))
    stop_text = "等待确认" if pd.isna(plan["参考止损"]) else f"{float(plan['参考止损']):,.2f}"
    target_text = "等待确认" if pd.isna(plan["第一目标"]) else f"{float(plan['第一目标']):,.2f}"
    c4.metric("参考止损", stop_text)
    c5.metric("第一目标", target_text)

    st.markdown(
        f"""
        **触发规则**

        多头入场窗口：隐藏动量 > 阈值、U_score >= 多头阈值、价格在卡尔曼滤波线上方。

        空头入场窗口：隐藏动量 < -阈值、U_score <= 空头阈值、价格在卡尔曼滤波线下方。

        震荡过滤：`ADX < 20` 时强制观望，不给出开多或开空。

        当前价格偏离滤波线：`{latest['price_bias_pct']:.2f}%`；当前波动缓冲：`{latest['vol_buffer']:.3f}`；当前 ADX：`{latest.get('adx', np.nan):.2f}`。
        """
    )

    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="trade_price_chart")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="trade_velocity_chart")

    st.subheader("最近多空入场机会")
    if opportunities.empty:
        st.info("当前历史窗口内没有满足完整条件的多空入场机会。可以降低阈值，或等待动量与宏观分数共振。")
    else:
        st.dataframe(
            opportunities.style.format(
                {
                    "信号强度": "{:.0f}%",
                    "参考止损": "{:.2f}",
                    "第一目标": "{:.2f}",
                    "U_score": "{:.2f}",
                    "动量": "{:.3f}",
                    "ADX": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.plotly_chart(build_macro_chart(macro.loc[filtered.index]), use_container_width=True, key="trade_macro_chart")


def render_sentiment_tab(news: pd.DataFrame, sentiment_daily: pd.DataFrame, macro: pd.DataFrame) -> None:
    st.subheader("情绪因子：新闻标题 -> VADER -> 日均值 -> Z-Score")
    st.markdown(
        f"""
        1. 从 Reuters、CNBC、Yahoo Finance、Google News RSS 抓取包含 `Silver / Fed / War` 的标题。
        2. 使用 VADER `compound` 得到每条标题的情绪分，范围为 `-1` 到 `1`。
        3. 按日期求平均，得到 `sentiment_raw`。
        4. 对 `sentiment_raw` 做 EWM Z-Score，得到 `z_sentiment`（近期权重更高）。
        5. 情绪贡献 = `{SENTIMENT_WEIGHT:.2f} * z_sentiment`。
        """
    )

    chart = macro.tail(180).copy()
    fig = build_factor_bar_line_chart(
        chart,
        bar_col="sentiment_factor",
        line_col="sentiment_raw",
        title="情绪贡献与原始新闻情绪",
        bar_name="情绪贡献",
        line_name="原始情绪分",
    )
    st.plotly_chart(fig, use_container_width=True, key="sentiment_factor_chart")

    daily_table = chart[["sentiment_raw", "z_sentiment", "sentiment_factor"]].tail(20).rename(
        columns={"sentiment_raw": "原始情绪分", "z_sentiment": "情绪 Z-Score", "sentiment_factor": "情绪贡献"}
    )
    st.dataframe(daily_table.style.format("{:.3f}"), use_container_width=True)

    st.subheader("相关新闻")
    news_table = format_news_table(news, limit=30)
    if news_table.empty:
        st.info("当前 RSS 源没有返回匹配新闻。")
    else:
        st.dataframe(news_table.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)


def render_dollar_tab(macro: pd.DataFrame) -> None:
    st.subheader("美元指数因子：美元走强通常压制白银")
    st.markdown(
        f"""
        1. 使用 yfinance 获取美元指数 `{DXY_TICKER}`。
        2. 计算日变动：`dxy_delta = pct_change(DXY) * 100`。
        3. 对 `dxy_delta` 做 EWM Z-Score，得到 `z_dxy`。
        4. 美元因子采用反向符号：`dxy_factor = -{DOLLAR_WEIGHT:.2f} * z_dxy`。
        """
    )

    fig = build_factor_bar_line_chart(
        macro,
        bar_col="dxy_factor",
        line_col="dxy",
        title="美元指数贡献与 DXY 水平",
        bar_name="美元指数贡献",
        line_name="DXY",
    )
    st.plotly_chart(fig, use_container_width=True, key="dollar_factor_chart")

    table = macro[["dxy", "dxy_delta", "z_dxy", "dxy_factor"]].tail(30).rename(
        columns={"dxy": "美元指数", "dxy_delta": "美元指数变动%", "z_dxy": "美元指数 Z-Score", "dxy_factor": "美元指数贡献"}
    )
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_rate_tab(macro: pd.DataFrame) -> None:
    st.subheader("利率因子：实际利率或收益率上行通常压制白银")
    st.markdown(
        f"""
        1. 优先使用 FRED `{FRED_REAL_RATE}` 十年期实际利率。
        2. 如果 FRED 数据不可用，则回退使用 yfinance `{TNX_TICKER}` 的十年期美债收益率变动。
        3. 计算 `rate_delta`，再做 EWM Z-Score 得到 `z_rate`。
        4. 利率因子采用反向符号：`rate_factor = -{RATE_WEIGHT:.2f} * z_rate`。
        """
    )

    fig = build_factor_bar_line_chart(
        macro,
        bar_col="rate_factor",
        line_col="real_rate",
        title="利率贡献与十年期实际利率",
        bar_name="利率贡献",
        line_name="实际利率",
    )
    st.plotly_chart(fig, use_container_width=True, key="rate_factor_chart")

    table = macro[["real_rate", "tnx", "rate_delta", "z_rate", "rate_factor"]].tail(30).rename(
        columns={
            "real_rate": "十年期实际利率",
            "tnx": "十年期美债收益率",
            "rate_delta": "利率变动",
            "z_rate": "利率 Z-Score",
            "rate_factor": "利率贡献",
        }
    )
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_composite_tab(macro: pd.DataFrame) -> None:
    st.subheader("宏观综合分 U_score")
    st.markdown(
        f"""
        `U_score = {SENTIMENT_WEIGHT:.2f} * z_sentiment - {DOLLAR_WEIGHT:.2f} * z_dxy - {RATE_WEIGHT:.2f} * z_rate`

        额外增强：`-0.08 * z_vix + 0.06 * z_gold_silver + 0.06 * z_copper + 0.04 * z_sp500`

        解释：情绪越正面越利多；美元走强、利率上行利空白银；VIX 上升利空风险资产；金银比/铜价/美股上升利好白银需求预期。
        """
    )
    st.plotly_chart(build_contribution_chart(macro), use_container_width=True, key="composite_contribution_chart")

    columns = ["sentiment_factor", "dxy_factor", "rate_factor", "U_score", "sentiment_raw", "dxy_delta", "rate_delta"]
    table = macro[columns].tail(30).rename(
        columns={
            "sentiment_factor": "情绪贡献",
            "dxy_factor": "美元贡献",
            "rate_factor": "利率贡献",
            "U_score": "宏观总分 U_score",
            "sentiment_raw": "原始情绪分",
            "dxy_delta": "美元指数变动%",
            "rate_delta": "利率变动",
        }
    )
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_extra_tab(macro: pd.DataFrame) -> None:
    st.subheader("增强数据因子：VIX、金银比、铜价、标普500、盈亏平衡通胀率")
    st.markdown(
        """
        这些额外数据源会对 U_score 产生微小偏置：
        - **VIX 上升** 意味着市场恐慌，利空白银（-0.08 * z_vix）
        - **金银比上升** 通常意味着白银相对被低估（+0.06 * z_gold_silver）
        - **铜价上涨** 代表工业需求旺盛，利好白银（+0.06 * z_copper）
        - **标普500 上涨** 代表风险偏好回暖（+0.04 * z_sp500）
        - **T10YIE 盈亏平衡通胀率** 反映通胀预期
        """
    )

    extra_charts = build_extra_charts(macro)
    for col_name, fig in extra_charts.items():
        st.plotly_chart(fig, use_container_width=True, key=f"extra_{col_name}_chart")

    table_cols = []
    for col in ["vix", "gold", "copper", "sp500", "gold_silver_ratio", "breakeven"]:
        if col in macro.columns:
            table_cols.append(col)
    for col in ["z_vix", "z_gold_silver", "z_copper", "z_sp500"]:
        if col in macro.columns:
            table_cols.append(col)

    if table_cols:
        rename_map = {
            "vix": "VIX",
            "gold": "黄金",
            "copper": "铜",
            "sp500": "标普500",
            "gold_silver_ratio": "金银比",
            "breakeven": "T10YIE",
            "z_vix": "VIX Z-Score",
            "z_gold_silver": "金银比 Z-Score",
            "z_copper": "铜 Z-Score",
            "z_sp500": "标普500 Z-Score",
        }
        table = macro[table_cols].tail(30).rename(columns=rename_map)
        st.dataframe(table.style.format("{:.2f}"), use_container_width=True)


def render_kalman_tab(filtered: pd.DataFrame, signal_frame: pd.DataFrame, controls: dict[str, object]) -> None:
    st.subheader("卡尔曼动量因子")
    st.markdown(
        f"""
        状态向量：`x = [价格, 动量]^T`

        状态转移矩阵：`F = [[1, 1], [0, Rho]]`，当前 `Rho = {float(controls['rho']):.2f}`。

        控制矩阵：`B = [[0.5 * Alpha], [Alpha]]`，当前 `Alpha = {float(controls['alpha']):.2f}`。

        动态测量噪声：`R_dynamic = R_base * vol_ratio`，其中 `vol_ratio` 来自 14/60 期绝对波动率比值。

        宏观总分 `U_score` 会作为外生冲击映射到动量加速度；动量转正且宏观顺风时偏多，动量转负且宏观逆风时偏空。
        """
    )
    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="kalman_price_chart")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="kalman_velocity_chart")

    table = signal_frame[
        [
            "observed",
            "kalman_price",
            "price_bias_pct",
            "velocity",
            "adx",
            "U_score",
            "entry_side",
            "long_score_pct",
            "short_score_pct",
        ]
    ].tail(30).rename(
        columns={
            "observed": "实际价格",
            "kalman_price": "卡尔曼价格",
            "price_bias_pct": "偏离滤波线%",
            "velocity": "隐藏动量",
            "adx": "ADX",
            "U_score": "宏观总分 U_score",
            "entry_side": "信号状态",
            "long_score_pct": "多头强度",
            "short_score_pct": "空头强度",
        }
    )
    st.dataframe(
        table.style.format(
            {
                "实际价格": "{:.2f}",
                "卡尔曼价格": "{:.2f}",
                "偏离滤波线%": "{:.2f}",
                "隐藏动量": "{:.3f}",
                "ADX": "{:.2f}",
                "宏观总分 U_score": "{:.2f}",
                "多头强度": "{:.0f}%",
                "空头强度": "{:.0f}%",
            }
        ),
        use_container_width=True,
    )


def render_raw_data_tab(
    market: pd.DataFrame,
    real_rate: pd.DataFrame,
    extra_data: pd.DataFrame | None,
    macro: pd.DataFrame,
    news: pd.DataFrame,
) -> None:
    st.subheader("行情与宏观原始数据")
    st.dataframe(market.tail(80), use_container_width=True)

    st.subheader("FRED 实际利率 + T10YIE 数据")
    if real_rate.empty:
        st.info("FRED 数据为空。")
    else:
        st.dataframe(real_rate.tail(80), use_container_width=True)

    if extra_data is not None and not extra_data.empty:
        st.subheader("增强品种数据 (VIX/黄金/铜/标普500)")
        st.dataframe(extra_data.tail(80), use_container_width=True)

    st.subheader("宏观计算结果")
    st.dataframe(macro.tail(80), use_container_width=True)

    st.subheader("新闻原始抓取结果")
    news_table = format_news_table(news, limit=80)
    if news_table.empty:
        st.info("新闻数据为空。")
    else:
        st.dataframe(news_table.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)
