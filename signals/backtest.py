from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    signal_frame: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
    max_hold_bars: int = 60,
) -> BacktestResult:
    data = signal_frame.copy()
    data = data.dropna(subset=["observed"])

    position = 0.0
    capital = float(initial_capital)
    entry_price = 0.0
    hold_bars = 0

    equity_series: list[float] = []
    trades: list[dict[str, object]] = []

    mkt_buffer = data["observed"] * 0.005
    long_setup = data["long_setup"]
    short_setup = data["short_setup"]

    prev_long = False
    prev_short = False

    for idx in range(len(data)):
        price = float(data["observed"].iloc[idx])
        timestamp = data.index[idx]
        equity_series.append(capital)

        long_trigger = bool(long_setup.iloc[idx]) and not prev_long
        short_trigger = bool(short_setup.iloc[idx]) and not prev_short
        prev_long = bool(long_setup.iloc[idx])
        prev_short = bool(short_setup.iloc[idx])

        if position == 0.0:
            if long_trigger:
                units = capital * 0.95 / price
                position = units
                entry_price = price
                entry_bar = idx
                hold_bars = 0
            elif short_trigger:
                units = capital * 0.95 / price
                position = -units
                entry_price = price
                entry_bar = idx
                hold_bars = 0
        else:
            hold_bars += 1
            exit_signal = False
            exit_reason = ""

            if hold_bars >= max_hold_bars:
                exit_signal = True
                exit_reason = "持仓超时"
            elif position > 0 and short_trigger:
                exit_signal = True
                exit_reason = "反转信号"
            elif position < 0 and long_trigger:
                exit_signal = True
                exit_reason = "反转信号"

            if exit_signal:
                pnl = position * (price - entry_price)
                pnl -= abs(position * entry_price) * commission
                pnl -= abs(position * price) * commission
                capital += pnl
                ret_pct = (pnl / (abs(position) * entry_price)) * 100.0
                trades.append(
                    {
                        "入场时间": data.index[entry_bar],
                        "出场时间": timestamp,
                        "方向": "做多" if position > 0 else "做空",
                        "入场价": entry_price,
                        "出场价": price,
                        "盈亏$": pnl,
                        "收益率%": ret_pct,
                        "出场原因": exit_reason,
                    }
                )
                position = 0.0
                entry_price = 0.0
                hold_bars = 0

    if position != 0.0:
        final_price = float(data["observed"].iloc[-1])
        pnl = position * (final_price - entry_price)
        pnl -= abs(position * entry_price) * commission
        pnl -= abs(position * final_price) * commission
        capital += pnl
        trades.append(
            {
                "入场时间": data.index[entry_bar],
                "出场时间": data.index[-1],
                "方向": "做多" if position > 0 else "做空",
                "入场价": entry_price,
                "出场价": final_price,
                "盈亏$": pnl,
                "收益率%": (pnl / (abs(position) * entry_price)) * 100.0,
                "出场原因": "回测结束强制平仓",
            }
        )

    equity = pd.Series(equity_series, index=data.index, name="equity")

    metrics = _compute_backtest_metrics(equity, trades, initial_capital)
    return BacktestResult(equity=equity, trades=pd.DataFrame(trades), metrics=metrics)


def _compute_backtest_metrics(
    equity: pd.Series,
    trades: list[dict[str, object]],
    initial_capital: float,
) -> dict[str, float]:
    trade_df = pd.DataFrame(trades)

    total_return = (equity.iloc[-1] / initial_capital - 1.0) * 100.0
    daily_returns = equity.pct_change().dropna()
    trading_days = len(daily_returns)
    annual_factor = 252.0

    cagr = 0.0
    if trading_days > 0 and equity.iloc[0] > 0:
        years = trading_days / annual_factor
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(years, 0.01)) - 1.0) * 100.0

    sharpe = 0.0
    if daily_returns.std() > 0 and len(daily_returns) > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(annual_factor)

    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax * 100.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0

    if not trade_df.empty and "盈亏$" in trade_df.columns:
        wins = trade_df[trade_df["盈亏$"] > 0]["盈亏$"]
        losses = trade_df[trade_df["盈亏$"] <= 0]["盈亏$"]
        win_rate = len(wins) / len(trade_df) * 100.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    total_trades = len(trade_df)

    return {
        "总收益率%": round(total_return, 2),
        "年化收益率%": round(cagr, 2),
        "夏普比率": round(sharpe, 2),
        "最大回撤%": round(max_drawdown, 2),
        "胜率%": round(win_rate, 2),
        "盈亏比": round(profit_factor, 2),
        "总交易次数": total_trades,
        "平均盈利$": round(avg_win, 2),
        "平均亏损$": round(avg_loss, 2),
    }


def build_equity_chart(result: BacktestResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index,
            y=result.equity.values,
            mode="lines",
            name="净值曲线",
            line=dict(color="#2E7D32", width=2.0),
        )
    )

    fig.add_hline(y=result.equity.iloc[0], line_color="#777777", line_width=1, line_dash="dash")

    if not result.trades.empty and "入场时间" in result.trades.columns:
        for _, trade in result.trades.iterrows():
            if pd.notna(trade.get("入场时间")):
                color = "#FF5722" if float(trade["盈亏$"]) > 0 else "#2196F3"
                fig.add_trace(
                    go.Scatter(
                        x=[trade["入场时间"], trade["出场时间"]],
                        y=[
                            result.equity.loc[trade["入场时间"]] if trade["入场时间"] in result.equity.index else None,
                            result.equity.loc[trade["出场时间"]] if trade["出场时间"] in result.equity.index else None,
                        ],
                        mode="lines+markers",
                        marker=dict(size=6, color=color),
                        line=dict(color=color, width=1.2),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=35, b=10),
        title="回测净值曲线",
        xaxis_title=None,
        yaxis_title="账户净值 ($)",
    )
    return fig


def build_monthly_heatmap(equity: pd.Series) -> go.Figure:
    if equity.empty:
        return go.Figure()

    daily_returns = equity.resample("D").last().pct_change().dropna()
    if daily_returns.empty:
        return go.Figure()

    returns_df = pd.DataFrame({"return": daily_returns})
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df["year"] = returns_df.index.year
    returns_df["month"] = returns_df.index.month

    monthly = returns_df.pivot_table(
        values="return", index="year", columns="month", aggfunc=lambda x: (1 + x).prod() - 1
    )

    if monthly.empty:
        return go.Figure()

    monthly_pct = monthly * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=monthly_pct.values,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: monthly_pct.shape[1]],
            y=[str(y) for y in monthly_pct.index],
            colorscale="RdYlGn",
            zmid=0,
            text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly_pct.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=35, b=10),
        title="月度收益率热力图 (%)",
        xaxis_title=None,
        yaxis_title=None,
    )
    return fig
