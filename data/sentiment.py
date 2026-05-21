from __future__ import annotations

from datetime import datetime

import feedparser
import pandas as pd
import requests
import streamlit as st

from config import CACHE_TTL_SENTIMENT, NEWS_KEYWORDS, RSS_FEEDS, RSS_TIMEOUT, RSS_USER_AGENT
from data.fetcher import FetchResult


def parse_entry_date(entry: dict) -> pd.Timestamp:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return pd.Timestamp(datetime(*parsed[:6]))
        except Exception:
            pass
    return pd.Timestamp.now(tz="UTC").tz_convert(None)


@st.cache_data(ttl=CACHE_TTL_SENTIMENT, show_spinner=False)
def fetch_news_sentiment(max_items: int = 80) -> FetchResult:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    warnings: list[str] = []
    analyzer = SentimentIntensityAnalyzer()
    rows: list[dict[str, object]] = []
    headers = {"User-Agent": RSS_USER_AGENT}

    for source, url in RSS_FEEDS.items():
        try:
            response = requests.get(url, headers=headers, timeout=RSS_TIMEOUT)
            response.raise_for_status()
            feed = feedparser.parse(response.content)
        except Exception as exc:
            warnings.append(f"{source} RSS 抓取失败：{exc}")
            continue

        for entry in feed.entries[:max_items]:
            title = str(entry.get("title", "")).strip()
            summary = str(entry.get("summary", "")).strip()
            text = f"{title} {summary}".lower()
            matched = [keyword for keyword in NEWS_KEYWORDS if keyword in text]
            if not title or not matched:
                continue

            score = analyzer.polarity_scores(title)["compound"]
            rows.append(
                {
                    "published": parse_entry_date(entry),
                    "source": source,
                    "title": title,
                    "score": float(score),
                    "keyword": ", ".join(matched),
                    "link": entry.get("link", ""),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return FetchResult(frame, warnings + ["未找到匹配关键词的 RSS 新闻标题。"])

    frame = frame.sort_values("published", ascending=False).reset_index(drop=True)
    return FetchResult(frame, warnings)


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty or "published" not in news:
        return pd.DataFrame(columns=["sentiment", "headline_count"])

    data = news.copy()
    data["date"] = pd.to_datetime(data["published"]).dt.normalize()
    grouped = data.groupby("date").agg(sentiment=("score", "mean"), headline_count=("score", "size"))
    return grouped.sort_index()


def format_news_table(news: pd.DataFrame, limit: int = 20, keyword: str | None = None) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=["发布时间", "来源", "关键词", "情绪分", "标题", "链接"])

    frame = news.copy()
    if keyword:
        frame = frame[frame["keyword"].str.contains(keyword, case=False, na=False)]
    frame = frame.head(limit).copy()
    if frame.empty:
        return pd.DataFrame(columns=["发布时间", "来源", "关键词", "情绪分", "标题", "链接"])

    frame["发布时间"] = pd.to_datetime(frame["published"]).dt.strftime("%Y-%m-%d %H:%M")
    return frame.rename(
        columns={
            "source": "来源",
            "keyword": "关键词",
            "score": "情绪分",
            "title": "标题",
            "link": "链接",
        }
    )[["发布时间", "来源", "关键词", "情绪分", "标题", "链接"]]
