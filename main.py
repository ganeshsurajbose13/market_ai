from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from scipy.stats import norm

app = FastAPI(title="Market AI Engine – Professional")

# =========================================================
# Utility Functions
# =========================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df


def add_ema(df: pd.DataFrame):
    df["EMA_5"] = EMAIndicator(df["Close"], window=5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    return df


def ema_cross_signal(df: pd.DataFrame) -> str:
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev["EMA_5"] < prev["EMA_10"] and curr["EMA_5"] > curr["EMA_10"]:
        return "BULLISH (5 EMA crossed above 10 EMA)"

    if prev["EMA_5"] > prev["EMA_10"] and curr["EMA_5"] < curr["EMA_10"]:
        return "BEARISH (5 EMA crossed below 10 EMA)"

    if prev["EMA_10"] < prev["EMA_20"] and curr["EMA_10"] > curr["EMA_20"]:
        return "BULLISH (10 EMA crossed above 20 EMA)"

    if prev["EMA_10"] > prev["EMA_20"] and curr["EMA_10"] < curr["EMA_20"]:
        return "BEARISH (10 EMA crossed below 20 EMA)"

    return "NO MAJOR CROSS"


def probability_prediction(df: pd.DataFrame) -> float:
    df["returns"] = df["Close"].pct_change()
    mean = df["returns"].mean()
    std = df["returns"].std()

    # Probability that next return > 0
    prob_up = 1 - norm.cdf(0, mean, std)
    return round(prob_up * 100, 2)


def analyze_timeframe(symbol: str, interval: str, period: str):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False
    )

    if df.empty:
        return None

    df = clean_dataframe(df)
    df = add_ema(df)

    return {
        "price": round(float(df.iloc[-1]["Close"]), 2),
        "ema_5": round(float(df.iloc[-1]["EMA_5"]), 2),
        "ema_10": round(float(df.iloc[-1]["EMA_10"]), 2),
        "ema_20": round(float(df.iloc[-1]["EMA_20"]), 2),
        "ema_cross": ema_cross_signal(df)
    }

# =========================================================
# API Endpoints
# =========================================================

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}


@app.get("/health")
def health():
    return {"status": "OK", "service": "Market AI", "version": "1.0"}


@app.get("/analyze")
def analyze(symbol: str = "RELIANCE.NS"):

    # ---------- MULTI TIMEFRAME ----------
    tf_1h = analyze_timeframe(symbol, "1h", "60d")
    tf_4h = analyze_timeframe(symbol, "4h", "6mo")
    tf_1d = analyze_timeframe(symbol, "1d", "1y")
    tf_1w = analyze_timeframe(symbol, "1wk", "5y")
    tf_1m = analyze_timeframe(symbol, "1mo", "10y")

    if not tf_1d:
        return {"error": "No data found"}

    # ---------- DAILY PREDICTION ----------
    df_daily = yf.download(symbol, interval="1d", period="1y", progress=False)
    df_daily = clean_dataframe(df_daily)
    df_daily = add_ema(df_daily)

    probability = probability_prediction(df_daily)
    trend_signal = ema_cross_signal(df_daily)

    prediction = "HOLD"
    if probability > 60:
        prediction = "BUY"
    elif probability < 40:
        prediction = "SELL"

    return {
        "symbol": symbol,

        "timeframes": {
            "1H": tf_1h,
            "4H": tf_4h,
            "1D": tf_1d,
            "1W": tf_1w,
            "1M": tf_1m
        },

        "daily_prediction": {
            "trend": trend_signal,
            "probability_up_percent": probability,
            "final_signal": prediction
        }
    }
