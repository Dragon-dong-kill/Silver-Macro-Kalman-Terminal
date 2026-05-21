from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict


def run_backtest(signal_frame, initial_capital=100000.0, commission=0.001,
                  max_hold_bars=60, use_stop_loss=True, use_trailing_stop=True):
    data = signal_frame.copy().dropna(subset=["observed"])
    n = len(data)
    if n == 0:
        return BacktestResult(pd.Series(), pd.DataFrame(), {})

    position = 0.0
    capital = float(initial_capital)
    entry_price = 0.0
    hold_bars = 0
    entry_bar = 0
    entry_strategy = ""
    entry_is_ranging = False
    trailing_high = 0.0
    trailing_low = 0.0

    equity_series = []
    trades = []

    ls = data["long_setup"]
    ss = data["short_setup"]
    prev_long = False
    prev_short = False

    is_ranging = data.get("is_ranging", pd.Series(False, index=data.index))

    for idx in range(n):
        price = float(data["observed"].iloc[idx])
        ts = data.index[idx]
        equity_series.append(capital)

        lt = bool(ls.iloc[idx]) and not prev_long
        st = bool(ss.iloc[idx]) and not prev_short
        prev_long = bool(ls.iloc[idx])
        prev_short = bool(ss.iloc[idx])

        if position == 0.0:
            if lt:
                is_range_entry = bool(is_ranging.iloc[idx]) if idx < n else False
                alloc = 0.50 if is_range_entry else 0.95
                position = capital * alloc / price
                entry_price = price
                entry_bar = idx
                hold_bars = 0
                trailing_high = price
                trailing_low = price
                entry_is_ranging = is_range_entry
                sty_col = data.get("strategy_type", pd.Series("", index=data.index))
                entry_strategy = str(sty_col.iloc[idx])
            elif st:
                is_range_entry = bool(is_ranging.iloc[idx]) if idx < n else False
                alloc = 0.50 if is_range_entry else 0.95
                position = -capital * alloc / price
                entry_price = price
                entry_bar = idx
                hold_bars = 0
                trailing_high = price
                trailing_low = price
                entry_is_ranging = is_range_entry
                sty_col = data.get("strategy_type", pd.Series("", index=data.index))
                entry_strategy = str(sty_col.iloc[idx])
        else:
            hold_bars += 1
            exit_signal = False
            exit_reason = ""

            atr_idx = min(idx, n - 1)
            atr = float(data["vol_buffer"].iloc[atr_idx])
            if pd.isna(atr) or atr <= 0:
                atr = price * 0.008

            if position > 0:
                trailing_high = max(trailing_high, price)
            else:
                trailing_low = min(trailing_low, price)

            # --- Ranging exit: bb_position returns to neutral ---
            if entry_is_ranging and not exit_signal:
                bb_pos = float(data.get("bb_position", pd.Series(0, index=data.index)).iloc[idx])
                if position > 0 and bb_pos > -0.3:
                    exit_signal = True
                    exit_reason = "布林带回归(多头)"
                elif position < 0 and bb_pos < 0.3:
                    exit_signal = True
                    exit_reason = "布林带回归(空头)"
                bb_stop = float(data.get("bb_position", pd.Series(0, index=data.index)).iloc[idx])
                if position > 0 and bb_stop < -3.0:
                    exit_signal = True
                    exit_reason = "布林带突破止损(多头)"
                elif position < 0 and bb_stop > 3.0:
                    exit_signal = True
                    exit_reason = "布林带突破止损(空头)"

            # --- Profit target ---
            if not exit_signal and entry_price > 0.001:
                tp_mult = float(data["reward_risk"].iloc[idx]) if "reward_risk" in data.columns else 2.0
                sm = float(data["stop_mult"].iloc[idx]) if "stop_mult" in data.columns else 1.5
                if entry_is_ranging:
                    tp_mult = 1.5
                    sm = 1.0
                sd = atr * sm
                td = sd * tp_mult
                if position > 0 and price >= entry_price + td:
                    exit_signal = True
                    exit_reason = "止盈"
                elif position < 0 and price <= entry_price - td:
                    exit_signal = True
                    exit_reason = "止盈"

            # --- Time exit ---
            if not exit_signal and hold_bars >= max_hold_bars:
                exit_signal = True
                exit_reason = "超时"

            # --- Stop loss ---
            if not exit_signal and use_stop_loss:
                sm = float(data["stop_mult"].iloc[idx]) if "stop_mult" in data.columns else 1.5
                if entry_is_ranging:
                    sm = 1.0
                if position > 0 and price <= entry_price - atr * sm:
                    exit_signal = True
                    exit_reason = "止损"
                elif position < 0 and price >= entry_price + atr * sm:
                    exit_signal = True
                    exit_reason = "止损"

            # --- Trailing stop ---
            if not exit_signal and use_trailing_stop and not entry_is_ranging:
                if position > 0 and trailing_high > entry_price + atr:
                    ts_stop = trailing_high - atr * 2.0
                    if price <= ts_stop:
                        exit_signal = True
                        exit_reason = "追踪止损"
                elif position < 0 and trailing_low < entry_price - atr:
                    ts_stop = trailing_low + atr * 2.0
                    if price >= ts_stop:
                        exit_signal = True
                        exit_reason = "追踪止损"

            # --- Reverse signal ---
            if not exit_signal:
                if position > 0 and st:
                    exit_signal = True
                    exit_reason = "反转信号→空头"
                elif position < 0 and lt:
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
                    "入场时间": data.index[entry_bar], "出场时间": ts,
                    "方向": "做多" if position > 0 else "做空",
                    "入场价": entry_price, "出场价": price,
                    "策略": entry_strategy,
                    "盈亏$": pnl, "收益率%": ret_pct, "出场原因": exit_reason,
                })
                position = 0.0
                entry_price = 0.0
                hold_bars = 0
                entry_strategy = ""
                entry_is_ranging = False

    if position != 0.0:
        fp = float(data["observed"].iloc[-1])
        pnl = position * (fp - entry_price)
        pnl -= abs(position * entry_price) * commission
        pnl -= abs(position * fp) * commission
        capital += pnl
        notional = abs(position) * entry_price
        trades.append({
            "入场时间": data.index[entry_bar], "出场时间": data.index[-1],
            "方向": "做多" if position > 0 else "做空",
            "入场价": entry_price, "出场价": fp,
            "策略": entry_strategy,
            "盈亏$": pnl, "收益率%": (pnl / notional) * 100.0 if notional > 0 else 0.0,
            "出场原因": "强制平仓",
        })

    equity = pd.Series(equity_series, index=data.index, name="equity")
    metrics = _compute_metrics(equity, trades, initial_capital)
    return BacktestResult(equity=equity, trades=pd.DataFrame(trades), metrics=metrics)


def _compute_metrics(equity, trades, ic):
    tdf = pd.DataFrame(trades)
    if equity.empty or len(equity) < 2:
        return {"总收益率%": 0, "年化收益率%": 0, "夏普比率": 0, "最大回撤%": 0,
                "胜率%": 0, "盈亏比": 0, "总交易次数": 0, "平均盈利$": 0,
                "平均亏损$": 0, "最大连续亏损": 0, "震荡胜率%": 0, "趋势胜率%": 0}

    tr = (equity.iloc[-1] / ic - 1.0) * 100.0
    dr = equity.pct_change().dropna()
    td = max(len(dr), 1)
    af = 252.0
    cagr = 0.0
    if equity.iloc[0] > 0:
        years = td / af
        cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(years, 0.01)) - 1.0) * 100.0
    sharpe = 0.0
    if dr.std() > 0 and len(dr) > 1:
        sharpe = dr.mean() / dr.std() * np.sqrt(af)
    cm = equity.expanding().max()
    dd = (equity - cm) / cm * 100.0
    mdd = float(dd.min()) if not dd.empty else 0.0

    wr = 0.0
    pf = 0.0
    aw = 0.0
    al = 0.0
    mcl = 0
    cl = 0
    rw = 0
    tw = 0
    rc = 0
    tc = 0

    if not tdf.empty and "盈亏$" in tdf.columns:
        for _, t in tdf.iterrows():
            p = float(t["盈亏$"])
            if p <= 0:
                cl += 1
                mcl = max(mcl, cl)
            else:
                cl = 0
            sty = str(t.get("策略", ""))
            if "布林带" in sty or "震荡" in sty or "均值" in sty:
                rc += 1
                if p > 0:
                    rw += 1
            elif "趋势" in sty or "强趋势" in sty:
                tc += 1
                if p > 0:
                    tw += 1

        wins = tdf[tdf["盈亏$"] > 0]
        losses = tdf[tdf["盈亏$"] <= 0]
        tt = len(tdf)
        wr = len(wins) / tt * 100.0 if tt > 0 else 0.0
        aw = float(wins["盈亏$"].mean()) if len(wins) > 0 else 0.0
        al = float(losses["盈亏$"].mean()) if len(losses) > 0 else 0.0
        gp = wins["盈亏$"].sum()
        gl = abs(losses["盈亏$"].sum())
        pf = float(gp / gl) if gl > 0 else float("inf")

    return {
        "总收益率%": round(tr, 2), "年化收益率%": round(cagr, 2),
        "夏普比率": round(sharpe, 2), "最大回撤%": round(mdd, 2),
        "胜率%": round(wr, 2), "盈亏比": round(pf, 2),
        "总交易次数": len(tdf), "平均盈利$": round(aw, 2),
        "平均亏损$": round(al, 2), "最大连续亏损": int(mcl),
        "震荡胜率%": round(rw / max(rc, 1) * 100, 1),
        "趋势胜率%": round(tw / max(tc, 1) * 100, 1),
    }


def build_equity_chart(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                              mode="lines", name="净值", line=dict(color="#2E7D32", width=2.0)))
    fig.add_hline(y=result.equity.iloc[0], line_color="#777777", line_width=1, line_dash="dash")
    if not result.trades.empty and "入场时间" in result.trades.columns:
        for _, t in result.trades.iterrows():
            c = "#4CAF50" if float(t["盈亏$"]) > 0 else "#F44336"
            et = t["入场时间"]
            xt = t["出场时间"]
            if et in result.equity.index and xt in result.equity.index:
                fig.add_trace(go.Scatter(x=[et, xt], y=[result.equity.loc[et], result.equity.loc[xt]],
                                          mode="lines+markers", marker=dict(size=4, color=c),
                                          line=dict(color=c, width=1), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=35, b=10),
                      title="回测净值曲线", xaxis_title=None, yaxis_title="账户净值 ($)")
    return fig


def build_monthly_heatmap(equity):
    if equity.empty:
        return go.Figure()
    dr = equity.resample("D").last().pct_change().dropna()
    if dr.empty:
        return go.Figure()
    rdf = pd.DataFrame({"return": dr})
    rdf.index = pd.to_datetime(rdf.index)
    rdf["year"] = rdf.index.year
    rdf["month"] = rdf.index.month
    monthly = rdf.pivot_table(values="return", index="year", columns="month",
                               aggfunc=lambda x: (1 + x).prod() - 1)
    if monthly.empty:
        return go.Figure()
    mp = monthly * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(
        z=mp.values, x=months[:mp.shape[1]], y=[str(y) for y in mp.index],
        colorscale="RdYlGn", zmid=0,
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in mp.values],
        texttemplate="%{text}", textfont=dict(size=10), hoverongaps=False))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=35, b=10),
                      title="月度收益率热力图 (%)", xaxis_title=None, yaxis_title=None)
    return fig
