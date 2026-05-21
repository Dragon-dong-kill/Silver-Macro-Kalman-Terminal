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
    signal_frame,
    initial_capital=100_000.0,
    commission=0.001,
    max_hold_bars=60,
    use_stop_loss=True,
    use_trailing_stop=True,
):
    data = signal_frame.copy().dropna(subset=["observed"])

    position = 0.0
    capital = float(initial_capital)
    entry_price = 0.0
    hold_bars = 0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0

    equity_series = []
    trades = []

    long_setup = data["long_setup"]
    short_setup = data["short_setup"]
    prev_long = False
    prev_short = False

    n = len(data)
    for idx in range(n):
        price = float(data["observed"].iloc[idx])
        timestamp = data.index[idx]
        equity_series.append(capital)

        long_trigger = bool(long_setup.iloc[idx]) and not prev_long
        short_trigger = bool(short_setup.iloc[idx]) and not prev_short
        prev_long = bool(long_setup.iloc[idx])
        prev_short = bool(short_setup.iloc[idx])

        if position == 0.0:
            if long_trigger:
                position = capital * 0.95 / price
                entry_price = price
                entry_bar = idx
                hold_bars = 0
                trailing_high = price
                trailing_low = price
            elif short_trigger:
                position = -capital * 0.95 / price
                entry_price = price
                entry_bar = idx
                hold_bars = 0
                trailing_high = price
                trailing_low = price
        else:
            hold_bars += 1
            exit_signal = False
            exit_reason = ""

            if position > 0:
                trailing_high = max(trailing_high, price)
                atr = float(data["vol_buffer"].iloc[idx]) if idx < len(data) else price * 0.01
            else:
                trailing_low = min(trailing_low, price)
                atr = float(data["vol_buffer"].iloc[idx]) if idx < len(data) else price * 0.01

            if hold_bars >= max_hold_bars:
                exit_signal = True
                exit_reason = "持仓超时"

            elif use_stop_loss and position > 0 and price <= entry_price - atr * 1.5:
                exit_signal = True
                exit_reason = "止损(多头)"
            elif use_stop_loss and position < 0 and price >= entry_price + atr * 1.5:
                exit_signal = True
                exit_reason = "止损(空头)"

            elif use_trailing_stop and position > 0 and trailing_high > entry_price:
                trail_stop = trailing_high - atr * 2.0
                if price <= trail_stop:
                    exit_signal = True
                    exit_reason = "追踪止损(多头)"
            elif use_trailing_stop and position < 0 and trailing_low < entry_price:
                trail_stop = trailing_low + atr * 2.0
                if price >= trail_stop:
                    exit_signal = True
                    exit_reason = "追踪止损(空头)"

            elif position > 0 and short_trigger:
                exit_signal = True
                exit_reason = "反转信号→空头"
            elif position < 0 and long_trigger:
                exit_signal = True
                exit_reason = "反转信号→多头"

            if exit_signal:
                pnl = position * (price - entry_price)
                pnl -= abs(position * entry_price) * commission
                pnl -= abs(position * price) * commission
                capital += pnl
                notional = abs(position) * entry_price
                ret_pct = (pnl / notional) * 100.0 if notional > 0 else 0.0
                trades.append({
                    "入场时间": data.index[entry_bar],
                    "出场时间": timestamp,
                    "方向": "做多" if position > 0 else "做空",
                    "入场价": entry_price,
                    "出场价": price,
                    "盈亏$": pnl,
                    "收益率%": ret_pct,
                    "出场原因": exit_reason,
                })
                position = 0.0
                entry_price = 0.0
                hold_bars = 0

    if position != 0.0:
        final_price = float(data["observed"].iloc[-1])
        pnl = position * (final_price - entry_price)
        pnl -= abs(position * entry_price) * commission
        pnl -= abs(position * final_price) * commission
        capital += pnl
        notional = abs(position) * entry_price
        trades.append({
            "入场时间": data.index[entry_bar],
            "出场时间": data.index[-1],
            "方向": "做多" if position > 0 else "做空",
            "入场价": entry_price,
            "出场价": final_price,
            "盈亏$": pnl,
            "收益率%": (pnl / notional) * 100.0 if notional > 0 else 0.0,
            "出场原因": "回测结束强制平仓",
        })

    equity = pd.Series(equity_series, index=data.index, name="equity")
    metrics = _compute_backtest_metrics(equity, trades, initial_capital)
    return BacktestResult(equity=equity, trades=pd.DataFrame(trades), metrics=metrics)


def _compute_backtest_metrics(equity, trades, initial_capital):
    trade_df = pd.DataFrame(trades)
    total_return = (equity.iloc[-1] / initial_capital - 1.0) * 100.0
    daily_returns = equity.pct_change().dropna()
    trading_days = max(len(daily_returns), 1)
    annual_factor = 252.0

    cagr = 0.0
    if equity.iloc[0] > 0:
        years = trading_days / annual_factor
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(years, 0.01)) - 1.0) * 100.0

    sharpe = 0.0
    if daily_returns.std() > 0 and len(daily_returns) > 1:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(annual_factor)

    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax * 100.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    win_rate = 0.0
    profit_factor = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    max_consecutive_losses = 0
    consecutive_losses = 0

    if not trade_df.empty and "盈亏$" in trade_df.columns:
        wins = trade_df[trade_df["盈亏$"] > 0]
        losses = trade_df[trade_df["盈亏$"] <= 0]

        for _, t in trade_df.iterrows():
            if float(t["盈亏$"]) <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        total_trades = len(trade_df)
        win_rate = len(wins) / total_trades * 100.0 if total_trades > 0 else 0.0
        avg_win = float(wins["盈亏$"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["盈亏$"].mean()) if len(losses) > 0 else 0.0
        gross_profit = wins["盈亏$"].sum()
        gross_loss = abs(losses["盈亏$"].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    return {
        "总收益率%": round(total_return, 2),
        "年化收益率%": round(cagr, 2),
        "夏普比率": round(sharpe, 2),
        "最大回撤%": round(max_drawdown, 2),
        "胜率%": round(win_rate, 2),
        "盈亏比": round(profit_factor, 2),
        "总交易次数": len(trade_df),
        "平均盈利$": round(avg_win, 2),
        "平均亏损$": round(avg_loss, 2),
        "最大连续亏损": int(max_consecutive_losses),
    }


def build_equity_chart(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.equity.index, y=result.equity.values,
        mode="lines", name="净值曲线",
        line=dict(color="#2E7D32", width=2.0),
    ))
    fig.add_hline(y=result.equity.iloc[0], line_color="#777777", line_width=1, line_dash="dash")

    if not result.trades.empty and "入场时间" in result.trades.columns:
        for _, trade in result.trades.iterrows():
            color = "#4CAF50" if float(trade["盈亏$"]) > 0 else "#F44336"
            et = trade["入场时间"]
            xt = trade["出场时间"]
            if et in result.equity.index and xt in result.equity.index:
                fig.add_trace(go.Scatter(
                    x=[et, xt],
                    y=[result.equity.loc[et], result.equity.loc[xt]],
                    mode="lines+markers",
                    marker=dict(size=5, color=color),
                    line=dict(color=color, width=1.2),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    fig.update_layout(height=400, margin=dict(l=10, r=10, t=35, b=10),
                      title="回测净值曲线", xaxis_title=None, yaxis_title="账户净值 ($)")
    return fig


def build_monthly_heatmap(equity):
    if equity.empty:
        return go.Figure()
    daily_returns = equity.resample("D").last().pct_change().dropna()
    if daily_returns.empty:
        return go.Figure()
    returns_df = pd.DataFrame({"return": daily_returns})
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df["year"] = returns_df.index.year
    returns_df["month"] = returns_df.index.month
    monthly = returns_df.pivot_table(values="return", index="year", columns="month", aggfunc=lambda x: (1 + x).prod() - 1)
    if monthly.empty:
        return go.Figure()
    monthly_pct = monthly * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(
        z=monthly_pct.values,
        x=months[:monthly_pct.shape[1]],
        y=[str(y) for y in monthly_pct.index],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly_pct.values],
        texttemplate="%{text}", textfont=dict(size=10), hoverongaps=False,
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10),
                      title="月度收益率热力图 (%)", xaxis_title=None, yaxis_title=None)
    return fig
