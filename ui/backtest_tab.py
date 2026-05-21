
import pandas as pd
import streamlit as st

from signals.backtest import BacktestResult, build_equity_chart, build_monthly_heatmap, run_backtest


def render_backtest_tab(signal_frame: pd.DataFrame, controls: dict[str, object]) -> None:
    st.subheader("历史回测 (基于多空入场窗口信号)")
    st.markdown(
        """
        回测规则：
        - 多头入场窗口出现时以 95% 仓位做多
        - 空头入场窗口出现时以 95% 仓位做空
        - 反向信号触发或持仓超过最大持有天数时平仓
        - 双向收取 0.1% 手续费
        - 每次只持有一个方向的仓位

        ⚠️ 此回测为简化模拟，未考虑滑点、保证金占用、交割展期等因素。
        """
    )

    c1, c2, c3 = st.columns(3)
    initial_capital = c1.number_input("初始资金 ($)", min_value=10_000.0, max_value=10_000_000.0, value=100_000.0, step=10_000.0)
    commission = c2.number_input("手续费 (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100.0
    max_hold_bars = c3.number_input("最大持仓天数", min_value=5, max_value=252, value=60, step=5)

    if st.button("运行回测", type="primary"):
        with st.spinner("正在运行回测..."):
            result = run_backtest(
                signal_frame=signal_frame,
                initial_capital=initial_capital,
                commission=commission,
                max_hold_bars=max_hold_bars,
            )
            _display_backtest_result(result)


def _display_backtest_result(result: BacktestResult) -> None:
    metrics = result.metrics

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总收益率", f"{metrics['总收益率%']:+.2f}%")
    c2.metric("年化收益率", f"{metrics['年化收益率%']:+.2f}%")
    c3.metric("夏普比率", f"{metrics['夏普比率']:.2f}")
    c4.metric("最大回撤", f"{metrics['最大回撤%']:.2f}%")
    c5.metric("胜率", f"{metrics['胜率%']:.1f}%")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("盈亏比", f"{metrics['盈亏比']:.2f}")
    c7.metric("总交易次数", int(metrics["总交易次数"]))
    c8.metric("平均盈利 ($)", f"{metrics['平均盈利$']:,.0f}")
    c9.metric("平均亏损 ($)", f"{metrics['平均亏损$']:,.0f}")

    st.plotly_chart(build_equity_chart(result), use_container_width=True, key="backtest_equity")

    monthly_fig = build_monthly_heatmap(result.equity)
    st.plotly_chart(monthly_fig, use_container_width=True, key="backtest_heatmap")

    st.subheader("交易明细")
    if result.trades.empty:
        st.info("回测期间无交易发生。")
    else:
        st.dataframe(
            result.trades.style.format({"盈亏$": "{:,.0f}", "收益率%": "{:.2f}%", "入场价": "{:.2f}", "出场价": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )
