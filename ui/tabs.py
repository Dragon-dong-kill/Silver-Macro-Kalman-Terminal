import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data.sentiment import format_news_table
from signals.signal_builder import current_trade_plan


def build_price_chart(filtered):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered["observed"], mode="lines",
                              name="SI=F 实际价格", line=dict(color="#2F4858", width=1.6)))
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered["kalman_price"], mode="lines",
                              name="卡尔曼滤波价格", line=dict(color="#F28E2B", width=2.2)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=35, b=10),
                      title="白银期货实际价格 vs 卡尔曼滤波均线",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_title=None, yaxis_title="美元")
    return fig


def build_macro_chart(macro):
    chart = macro.dropna(subset=["U_score"]).copy()
    colors = np.where(chart["U_score"] >= 0.0, "#2E7D32", "#B23A48")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart.index, y=chart["U_score"], marker_color=colors, name="U_score",
                          hovertemplate="%{x|%Y-%m-%d}<br>U_score=%{y:.2f}<extra></extra>"))
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10),
                      title="每日宏观综合打分", xaxis_title=None, yaxis_title="加权 Z-Score", showlegend=False)
    return fig


def build_velocity_chart(filtered):
    buy = filtered[filtered["buy_signal"]]
    sell = filtered[filtered["sell_signal"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered["velocity"], mode="lines",
                              name="隐藏动量", line=dict(color="#4E79A7", width=2.0)))
    fig.add_trace(go.Scatter(x=buy.index, y=buy["velocity"], mode="markers", name="买入",
                              marker=dict(color="#2E7D32", size=10, symbol="triangle-up"),
                              hovertemplate="%{x|%Y-%m-%d}<br>买入动量=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=sell.index, y=sell["velocity"], mode="markers", name="卖出",
                              marker=dict(color="#B23A48", size=10, symbol="triangle-down"),
                              hovertemplate="%{x|%Y-%m-%d}<br>卖出动量=%{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title="隐藏动量状态切换",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_title=None, yaxis_title="动量")
    return fig


def build_contribution_chart(macro):
    chart = macro.tail(180).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart.index, y=chart["sentiment_factor"], name="情绪贡献", marker_color="#59A14F"))
    fig.add_trace(go.Bar(x=chart.index, y=chart["dxy_factor"], name="美元指数贡献", marker_color="#4E79A7"))
    fig.add_trace(go.Bar(x=chart.index, y=chart["rate_factor"], name="利率贡献", marker_color="#B07AA1"))
    fig.add_trace(go.Scatter(x=chart.index, y=chart["U_score"], name="U_score", mode="lines",
                              line=dict(color="#F28E2B", width=2.4)))
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(barmode="relative", height=380, margin=dict(l=10, r=10, t=35, b=10),
                      title="U_score 拆解", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_title=None, yaxis_title="分数贡献")
    return fig


def build_factor_bar_line_chart(frame, bar_col, line_col, title, bar_name, line_name):
    chart = frame.tail(180).copy()
    colors = np.where(chart[bar_col] >= 0.0, "#2E7D32", "#B23A48")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart.index, y=chart[bar_col], name=bar_name, marker_color=colors, yaxis="y"))
    fig.add_trace(go.Scatter(x=chart.index, y=chart[line_col], name=line_name, mode="lines",
                              yaxis="y2", line=dict(color="#4E79A7", width=2.0)))
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title=title,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      xaxis_title=None, yaxis=dict(title=bar_name),
                      yaxis2=dict(title=line_name, overlaying="y", side="right", showgrid=False))
    return fig


def build_extra_charts(macro):
    charts = {}
    available = {"vix": "VIX", "gold_silver_ratio": "金银比", "copper": "铜", "sp500": "标普500"}
    for col, label in available.items():
        z_col = "z_" + col
        if col in macro.columns and z_col in macro.columns:
            charts[col] = build_factor_bar_line_chart(macro, bar_col=z_col, line_col=col,
                                                       title=label + " Z-Score 贡献",
                                                       bar_name=label + " 贡献", line_name=label)
    if "breakeven" in macro.columns:
        fig = go.Figure()
        chart = macro.tail(180).copy()
        colors = np.where(chart["breakeven_delta"].fillna(0) >= 0.0, "#2E7D32", "#B23A48")
        fig.add_trace(go.Bar(x=chart.index, y=chart["breakeven_delta"], name="通胀变动", marker_color=colors, yaxis="y"))
        fig.add_trace(go.Scatter(x=chart.index, y=chart["breakeven"], name="T10YIE", mode="lines",
                                  yaxis="y2", line=dict(color="#4E79A7", width=2.0)))
        fig.add_hline(y=0.0, line_color="#777777", line_width=1)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=35, b=10), title="T10YIE 盈亏平衡通胀率",
                          xaxis_title=None, yaxis=dict(title="变动 (bp)"),
                          yaxis2=dict(title="T10YIE (%)", overlaying="y", side="right", showgrid=False))
        charts["breakeven"] = fig
    return charts


def build_regime_chart(macro):
    chart = macro.tail(250).copy()
    if "market_regime" not in chart.columns:
        return go.Figure()
    regime_map = {"ranging": 0, "trending": 1, "strong_trending": 2}
    color_map = {"ranging": "#90A4AE", "trending": "#FFA726", "strong_trending": "#EF5350"}
    regime_num = chart["market_regime"].map(regime_map).fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart.index, y=regime_num, mode="lines", name="市场状态",
                              line=dict(color="#78909C", width=1.5),
                              fill="tozeroy", fillcolor="rgba(120,144,156,0.15)"))
    for regime, num in regime_map.items():
        mask = chart["market_regime"] == regime
        if mask.any():
            fig.add_trace(go.Scatter(x=chart.index[mask], y=[num] * mask.sum(),
                                      mode="markers", name=regime,
                                      marker=dict(color=color_map[regime], size=8, symbol="square"),
                                      showlegend=True))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10),
                      title="市场状态检测 (0震荡 1趋势 2强趋势)",
                      xaxis_title=None, yaxis=dict(tickvals=[0, 1, 2], ticktext=["震荡", "趋势", "强趋势"]),
                      showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def render_trade_tab(filtered, macro, signal_frame, opportunities, latest_quote, ai_result=None):
    st.warning("本页面只输出量化策略信号，不构成投资建议。")
    ai_dir = ai_result.get("direction", "") if ai_result else ""
    plan = current_trade_plan(signal_frame, latest_quote, ai_direction=ai_dir)
    latest = signal_frame.iloc[-1]
    tone = plan["tone"]
    if tone == "long":
        st.success("当前信号：" + str(plan['方向']) + "。" + str(plan['动作']))
    elif tone == "short":
        st.error("当前信号：" + str(plan['方向']) + "。" + str(plan['动作']))
    elif tone == "watch":
        st.info("当前信号：" + str(plan['方向']) + "。" + str(plan['动作']))
    else:
        st.info("当前信号：" + str(plan['方向']) + "。" + str(plan['动作']))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("建议状态", str(plan["方向"]))
    c2.metric("信号强度", "{:.0f}%".format(float(plan['信号强度'])))
    c3.metric("市场状态", str(plan.get("regime", "N/A")))
    stop_text = "等待确认" if pd.isna(plan["参考止损"]) else "{:,.2f}".format(float(plan["参考止损"]))
    target_text = "等待确认" if pd.isna(plan["第一目标"]) else "{:,.2f}".format(float(plan["第一目标"]))
    c4.metric("参考止损", stop_text)
    c5.metric("第一目标", target_text)

    fv = float(plan.get("fv_deviation", 0))
    bb_p = float(plan.get("bb_position", 0))
    st.markdown(
        "价格偏离滤波 **{:.2f}%** | ADX **{:.2f}** | 布林带位置 **{:+.1f}σ** | 公允值偏离 **{:+.2f}σ**".format(
            latest['price_bias_pct'], float(latest.get('adx', np.nan)), bb_p, fv))

    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="tpc")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="tvc")

    st.subheader("最近多空入场机会")
    if opportunities.empty:
        st.info("当前无满足条件的入场机会。")
    else:
        st.dataframe(
            opportunities.style.format({"信号强度": "{:.0f}%", "参考止损": "{:.2f}", "第一目标": "{:.2f}",
                                         "U_score": "{:.2f}", "动量": "{:.3f}", "ADX": "{:.2f}"}),
            use_container_width=True, hide_index=True)

    st.plotly_chart(build_macro_chart(macro.loc[filtered.index]), use_container_width=True, key="tmc")


def render_sentiment_tab(news, sentiment_daily, macro):
    st.subheader("情绪因子：VADER → Z-Score")
    chart = macro.tail(180).copy()
    st.plotly_chart(build_factor_bar_line_chart(chart, bar_col="sentiment_factor", line_col="sentiment_raw",
                                                  title="情绪贡献与原始新闻情绪", bar_name="情绪贡献", line_name="原始情绪分"),
                    use_container_width=True, key="sfc")
    table = chart[["sentiment_raw", "z_sentiment", "sentiment_factor"]].tail(20).rename(
        columns={"sentiment_raw": "原始情绪分", "z_sentiment": "情绪 Z-Score", "sentiment_factor": "情绪贡献"})
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    st.subheader("相关新闻")
    nt = format_news_table(news, limit=30)
    if nt.empty:
        st.info("无匹配新闻。")
    else:
        st.dataframe(nt.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)


def render_dollar_tab(macro):
    st.subheader("美元指数因子")
    st.plotly_chart(build_factor_bar_line_chart(macro, bar_col="dxy_factor", line_col="dxy",
                                                  title="美元指数贡献与 DXY", bar_name="美元指数贡献", line_name="DXY"),
                    use_container_width=True, key="dfc")
    table = macro[["dxy", "dxy_delta", "z_dxy", "dxy_factor"]].tail(30).rename(
        columns={"dxy": "美元指数", "dxy_delta": "变动%", "z_dxy": "Z-Score", "dxy_factor": "贡献"})
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_rate_tab(macro):
    st.subheader("利率因子")
    st.plotly_chart(build_factor_bar_line_chart(macro, bar_col="rate_factor", line_col="real_rate",
                                                  title="利率贡献与十年期实际利率", bar_name="利率贡献", line_name="实际利率"),
                    use_container_width=True, key="rfc")
    table = macro[["real_rate", "tnx", "rate_delta", "z_rate", "rate_factor"]].tail(30).rename(
        columns={"real_rate": "实际利率", "tnx": "十年期收益", "rate_delta": "利率变动", "z_rate": "Z-Score", "rate_factor": "贡献"})
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)


def render_composite_tab(macro):
    st.subheader("宏观综合分 U_score + 市场状态")
    st.plotly_chart(build_contribution_chart(macro), use_container_width=True, key="cc")
    st.plotly_chart(build_regime_chart(macro), use_container_width=True, key="rc")
    cols = ["sentiment_factor", "dxy_factor", "rate_factor", "U_score", "sentiment_raw", "dxy_delta", "rate_delta", "market_regime"]
    table = macro[cols].tail(30).rename(columns={
        "sentiment_factor": "情绪贡献", "dxy_factor": "美元贡献", "rate_factor": "利率贡献",
        "U_score": "宏观总分", "sentiment_raw": "原始情绪", "dxy_delta": "美元变动%", "rate_delta": "利率变动", "market_regime": "市场状态"})
    st.dataframe(table.style.format({k: "{:.3f}" for k in table.columns if k != "市场状态"}), use_container_width=True)


def render_ai_tab(ai_result, macro, filtered, api_key=""):
    st.subheader("AI 基本面实时分析")
    if not api_key:
        st.info("配置 DeepSeek API Key 可获得 AI 驱动的实时基本面分析。"
                "访问 platform.deepseek.com 注册获取。")
    if ai_result is None:
        st.warning("AI 分析暂不可用（未配置 API Key）。当前使用本地多因子模型。")
        return
    st.caption("分析来源：" + str(ai_result.get("source", "N/A")))
    direction = ai_result.get("direction", "neutral")
    if direction == "bullish":
        st.success("AI 判断：偏多")
    elif direction == "bearish":
        st.error("AI 判断：偏空")
    else:
        st.info("AI 判断：中性")
    c1, c2, c3 = st.columns(3)
    c1.metric("置信度", "{:.0%}".format(ai_result.get("confidence", 0.5)))
    c2.metric("基本面分数", "{:+d}".format(ai_result.get("score", 0)))
    c3.metric("技术面动量", "{:.3f}".format(float(filtered.iloc[-1]["velocity"])))
    st.subheader("关键驱动因素")
    st.info(str(ai_result.get("key_drivers", "N/A")))
    c1, c2 = st.columns(2)
    c1.subheader("风险因素")
    c1.warning(str(ai_result.get("risk_factors", "N/A")))
    c2.subheader("短期展望")
    c2.info(str(ai_result.get("short_term_view", "N/A")))
    st.subheader("宏观评估")
    st.markdown(str(ai_result.get("macro_assessment", "N/A")))


def render_extra_tab(macro):
    st.subheader("增强数据因子")
    extra_charts = build_extra_charts(macro)
    for col_name, fig in extra_charts.items():
        st.plotly_chart(fig, use_container_width=True, key="ex_" + col_name)
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
    st.markdown("Rho={:.2f} Alpha={:.2f} Q={:.4f} R={:.4f}".format(
        float(controls['rho']), float(controls['alpha']), float(controls['q']), float(controls['r'])))
    st.plotly_chart(build_price_chart(filtered), use_container_width=True, key="kpc")
    st.plotly_chart(build_velocity_chart(filtered), use_container_width=True, key="kvc")
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
    nt = format_news_table(news, limit=80)
    if nt.empty:
        st.info("新闻数据为空。")
    else:
        st.dataframe(nt.style.format({"情绪分": "{:.3f}"}), use_container_width=True, hide_index=True)
