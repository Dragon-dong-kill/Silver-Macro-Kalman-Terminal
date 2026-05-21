from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def build_price_chart(filtered: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered.index,
            y=filtered["observed"],
            mode="lines",
            name="SI=F 实际价格",
            line=dict(color="#2F4858", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered.index,
            y=filtered["kalman_price"],
            mode="lines",
            name="卡尔曼滤波价格",
            line=dict(color="#F28E2B", width=2.2),
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        title="白银期货实际价格 vs 卡尔曼滤波均线",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="美元 / 盎司",
    )
    return fig


def build_macro_chart(macro: pd.DataFrame) -> go.Figure:
    chart = macro.dropna(subset=["U_score"]).copy()
    colors = np.where(chart["U_score"] >= 0.0, "#2E7D32", "#B23A48")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart.index,
            y=chart["U_score"],
            marker_color=colors,
            name="U_score",
            hovertemplate="%{x|%Y-%m-%d}<br>宏观总分 U_score=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=35, b=10),
        title="每日宏观综合打分",
        xaxis_title=None,
        yaxis_title="加权 Z-Score",
        showlegend=False,
    )
    return fig


def build_velocity_chart(filtered: pd.DataFrame) -> go.Figure:
    buy = filtered[filtered["buy_signal"]]
    sell = filtered[filtered["sell_signal"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered.index,
            y=filtered["velocity"],
            mode="lines",
            name="隐藏动量",
            line=dict(color="#4E79A7", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=buy.index,
            y=buy["velocity"],
            mode="markers",
            name="买入",
            marker=dict(color="#2E7D32", size=10, symbol="triangle-up"),
            hovertemplate="%{x|%Y-%m-%d}<br>买入动量=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell.index,
            y=sell["velocity"],
            mode="markers",
            name="卖出",
            marker=dict(color="#B23A48", size=10, symbol="triangle-down"),
            hovertemplate="%{x|%Y-%m-%d}<br>卖出动量=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title="隐藏动量状态切换",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="动量",
    )
    return fig


def build_contribution_chart(macro: pd.DataFrame) -> go.Figure:
    chart = macro.tail(180).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=chart.index, y=chart["sentiment_factor"], name="情绪贡献", marker_color="#59A14F"))
    fig.add_trace(go.Bar(x=chart.index, y=chart["dxy_factor"], name="美元指数贡献", marker_color="#4E79A7"))
    fig.add_trace(go.Bar(x=chart.index, y=chart["rate_factor"], name="利率贡献", marker_color="#B07AA1"))
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart["U_score"],
            name="U_score",
            mode="lines",
            line=dict(color="#F28E2B", width=2.4),
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        barmode="relative",
        height=380,
        margin=dict(l=10, r=10, t=35, b=10),
        title="U_score 拆解：情绪 + 美元 + 利率",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="分数贡献",
    )
    return fig


def build_factor_bar_line_chart(
    frame: pd.DataFrame,
    bar_col: str,
    line_col: str,
    title: str,
    bar_name: str,
    line_name: str,
) -> go.Figure:
    chart = frame.tail(180).copy()
    colors = np.where(chart[bar_col] >= 0.0, "#2E7D32", "#B23A48")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart.index,
            y=chart[bar_col],
            name=bar_name,
            marker_color=colors,
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart.index,
            y=chart[line_col],
            name=line_name,
            mode="lines",
            yaxis="y2",
            line=dict(color="#4E79A7", width=2.0),
        )
    )
    fig.add_hline(y=0.0, line_color="#777777", line_width=1)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis=dict(title=bar_name),
        yaxis2=dict(title=line_name, overlaying="y", side="right", showgrid=False),
    )
    return fig


def build_extra_charts(macro: pd.DataFrame) -> dict[str, go.Figure]:
    charts: dict[str, go.Figure] = {}

    available = {
        "vix": "VIX 恐慌指数",
        "gold_silver_ratio": "金银比",
        "copper": "铜价水平",
        "sp500": "标普500水平",
    }
    for col, label in available.items():
        z_col = f"z_{col}"
        if col in macro.columns and z_col in macro.columns:
            fig = build_factor_bar_line_chart(
                macro,
                bar_col=z_col,
                line_col=col,
                title=f"{label} Z-Score 贡献",
                bar_name=f"{label} 贡献",
                line_name=label,
            )
            charts[col] = fig

    if "breakeven" in macro.columns:
        fig = go.Figure()
        chart = macro.tail(180).copy()
        colors = np.where(chart["breakeven_delta"].fillna(0) >= 0.0, "#2E7D32", "#B23A48")
        fig.add_trace(
            go.Bar(
                x=chart.index,
                y=chart["breakeven_delta"],
                name="盈亏平衡通胀变动",
                marker_color=colors,
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart.index,
                y=chart["breakeven"],
                name="T10YIE",
                mode="lines",
                yaxis="y2",
                line=dict(color="#4E79A7", width=2.0),
            )
        )
        fig.add_hline(y=0.0, line_color="#777777", line_width=1)
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=35, b=10),
            title="十年期盈亏平衡通胀率 (T10YIE)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_title=None,
            yaxis=dict(title="变动 (bp)"),
            yaxis2=dict(title="T10YIE (%)", overlaying="y", side="right", showgrid=False),
        )
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
    fig.add_trace(go.Scatter(
        x=chart.index, y=regime_num,
        mode="lines",
        name="市场状态",
        line=dict(color="#78909C", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(120,144,156,0.15)",
    ))

    for regime, num in regime_map.items():
        mask = chart["market_regime"] == regime
        if mask.any():
            fig.add_trace(go.Scatter(
                x=chart.index[mask],
                y=[num] * mask.sum(),
                mode="markers",
                name=regime,
                marker=dict(color=color_map[regime], size=8, symbol="square"),
                showlegend=True,
            ))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=35, b=10),
        title="市场状态检测 (0=震荡 1=趋势 2=强趋势)",
        xaxis_title=None,
        yaxis=dict(tickvals=[0, 1, 2], ticktext=["震荡", "趋势", "强趋势"]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
