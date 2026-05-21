
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import StringIO
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from config import (
    CACHE_TTL_LATEST,
    CACHE_TTL_MARKET,
    CACHE_TTL_OHLC,
    CACHE_TTL_REAL_RATE,
    FRED_BREAKEVEN,
    FRED_REAL_RATE,
    SILVER_TICKER,
    YAHOO_TIMEOUT,
    MKT_TICKERS,
    EXTRA_TICKERS,
)
from data.utils import normalize_index


@dataclass
class FetchResult:
    data: pd.DataFrame
    warnings: list[str]


def _parse_yf_close(raw: pd.DataFrame, tickers: list[str]) -> FetchResult:
    raw = raw.copy()
    warnings: list[str] = []
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close" if "Close" in raw.columns.get_level_values(0) else "Adj Close"
            close = raw[field].copy()
        else:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"无法解析 yfinance 收盘价数据：{exc}"])

    close = normalize_index(close)
    for ticker in tickers:
        if ticker not in close.columns or close[ticker].dropna().empty:
            warnings.append(f"yfinance 缺失或返回空数据：{ticker}")
    return FetchResult(close, warnings)


@st.cache_data(ttl=CACHE_TTL_MARKET, show_spinner=False)
def fetch_market_data(period: str, interval: str) -> FetchResult:
    try:
        raw = yf.download(
            tickers=MKT_TICKERS,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=YAHOO_TIMEOUT,
        )
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"yfinance 行情下载失败：{exc}"])

    if raw.empty:
        return FetchResult(pd.DataFrame(), ["yfinance 未返回可用行情数据。"])
    return _parse_yf_close(raw, MKT_TICKERS)


@st.cache_data(ttl=CACHE_TTL_MARKET, show_spinner=False)
def fetch_extra_data(period: str, interval: str) -> FetchResult:
    try:
        raw = yf.download(
            tickers=EXTRA_TICKERS,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=YAHOO_TIMEOUT,
        )
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"额外品种下载失败：{exc}"])

    if raw.empty:
        return FetchResult(pd.DataFrame(), ["额外品种数据为空。"])
    return _parse_yf_close(raw, EXTRA_TICKERS)


@st.cache_data(ttl=CACHE_TTL_OHLC, show_spinner=False)
def fetch_silver_ohlc(period: str, interval: str) -> FetchResult:
    try:
        raw = yf.download(
            tickers=SILVER_TICKER,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=YAHOO_TIMEOUT,
        )
    except Exception as exc:
        return FetchResult(pd.DataFrame(), [f"SI=F OHLC download failed: {exc}"])

    if raw.empty:
        return FetchResult(pd.DataFrame(), ["SI=F OHLC is empty."])

    if isinstance(raw.columns, pd.MultiIndex):
        ohlc = raw.droplevel(1, axis=1).copy()
    else:
        ohlc = raw.copy()

    required = ["Open", "High", "Low", "Close"]
    missing = [name for name in required if name not in ohlc.columns]
    if missing:
        return FetchResult(pd.DataFrame(), [f"SI=F OHLC missing fields: {', '.join(missing)}"])

    cleaned = normalize_index(ohlc[required].copy())
    return FetchResult(cleaned, [])


@st.cache_data(ttl=CACHE_TTL_LATEST, show_spinner=False)
def fetch_latest_silver_price():
    try:
        ticker = yf.Ticker(SILVER_TICKER)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            last_price = fast_info.get("last_price")
            if last_price is not None and not pd.isna(last_price):
                return float(last_price), None

        intraday = ticker.history(period="1d", interval="1m", timeout=10)
        if not intraday.empty and "Close" in intraday:
            return float(intraday["Close"].dropna().iloc[-1]), None
    except Exception as exc:
        return None, f"白银实时报价获取失败：{exc}"

    return None, "白银实时报价暂不可用。"


FRED_CSV_URL = "https://fred.stlouisfed.org/data/{symbol}.txt"


@st.cache_data(ttl=CACHE_TTL_REAL_RATE, show_spinner=False)
def fetch_fred_data(start, end, symbols):
    warnings = []
    all_data = {}

    for symbol in symbols:
        raw = None
        try:
            url = FRED_CSV_URL.format(symbol=symbol)
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            lines = resp.text.splitlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("DATE"):
                    data_start = i
                    break
            csv_text = "\n".join(lines[data_start:])
            raw = pd.read_csv(StringIO(csv_text), parse_dates=["DATE"], index_col="DATE")
            raw = raw.rename(columns={"VALUE": symbol})
        except Exception:
            warnings.append(f"FRED {symbol} 数据获取失败。")
            continue

        if raw is None or raw.empty:
            warnings.append(f"FRED {symbol} 数据为空。")
            continue

        raw = normalize_index(raw)
        mask = (raw.index >= pd.Timestamp(start)) & (raw.index <= pd.Timestamp(end))
        sliced = raw.loc[mask]
        if sliced.empty:
            warnings.append(f"FRED {symbol} 在指定时间范围内无数据。")
            continue
        all_data[symbol] = sliced

    if not all_data:
        return FetchResult(pd.DataFrame(), warnings)

    merged = pd.concat(list(all_data.values()), axis=1)
    merged.columns = list(all_data.keys())
    return FetchResult(merged, warnings)


def resolve_data_range(market_data):
    start = market_data.index.min().to_pydatetime() - timedelta(days=10)
    end = market_data.index.max().to_pydatetime() + timedelta(days=1)
    return start, end
