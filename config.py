from __future__ import annotations

APP_NAME = "Silver-Macro-Kalman-Terminal"
SILVER_TICKER = "SI=F"
DXY_TICKER = "DX-Y.NYB"
TNX_TICKER = "^TNX"

MKT_TICKERS = [SILVER_TICKER, DXY_TICKER, TNX_TICKER]

VIX_TICKER = "^VIX"
GOLD_TICKER = "GC=F"
COPPER_TICKER = "HG=F"
SP500_TICKER = "^GSPC"

EXTRA_TICKERS = [VIX_TICKER, GOLD_TICKER, COPPER_TICKER, SP500_TICKER]

FRED_REAL_RATE = "DFII10"
FRED_BREAKEVEN = "T10YIE"

SENTIMENT_WEIGHT = 0.40
DOLLAR_WEIGHT = 0.35
RATE_WEIGHT = 0.25

NEWS_KEYWORDS = ("silver", "fed", "federal reserve", "war")
RSS_FEEDS = {
    "Reuters Business": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "CNBC Markets": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Google News Query": (
        "https://news.google.com/rss/search?"
        "q=silver%20OR%20Fed%20OR%20war%20when:7d&hl=en-US&gl=US&ceid=US:en"
    ),
}

DEFAULT_RHO = 0.86
DEFAULT_ALPHA = 0.25
DEFAULT_Q = 0.08
DEFAULT_R = 2.5

DEFAULT_Z_WINDOW = 60
DEFAULT_LONG_THRESHOLD = 0.25
DEFAULT_SHORT_THRESHOLD = -0.25
DEFAULT_MIN_VELOCITY = 0.05
DEFAULT_STOP_MULT = 1.50
DEFAULT_REWARD_RISK = 2.00
DEFAULT_ADX_THRESHOLD = 20.0

CACHE_TTL_MARKET = 600
CACHE_TTL_REAL_RATE = 3600
CACHE_TTL_SENTIMENT = 900
CACHE_TTL_LATEST = 60
CACHE_TTL_OHLC = 600

YAHOO_TIMEOUT = 20
RSS_TIMEOUT = 8
RSS_USER_AGENT = (
    "Mozilla/5.0 Silver-Macro-Kalman-Terminal "
    "(local research dashboard; contact: local)"
)
