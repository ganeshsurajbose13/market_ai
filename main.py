from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from scipy.stats import binomtest

app = FastAPI(title="Market AI Engine – Pro Level")

# ---------------------------------
# Helper Functions
# ---------------------------------

def clean_dataframe(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df

def add_ema(df):
    df["EMA_5"] = EMAIndicator(df["Close"], window=5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    return df

def ema_crossover_signal(df):
    prev = df.iloc[-2]
    curr = df.iloc[-1]

    if prev["EMA_5"] < prev["EMA_10"] and curr["EMA_5"] > curr["EMA_10"]:
        return "BULLISH CROSSOVER"
    elif prev["EMA_5"] > prev["EMA_10"] and curr["EMA_5"] < curr["EMA_10"]:
        return "BEARISH CROSSOVER"
    return "NO CROSSOVER"

def trend_strength(df):
    last = df.iloc[-1]
    if last["Close"] > last["EMA_20"]:
        return "UPTREND"
    return "DOWNTREND"

def statistical_probability(df):
    df["Return"] = df["Close"].pct_change()
    wins = (df["Return"] > 0).sum()
    total = df["Return"].count()

    if total < 30:
        return "Not enough data"

    test = binomtest(wins, total, 0.5)
    probability = round((wins / total) * 100, 2)
    confidence = "HIGH" if test.pvalue < 0.05 else "LOW"

    return {
        "Win_Rate_%": probability,
        "Stat_Confidence": confidence
    }

# ---------------------------------
# API ROUTES
# ---------------------------------

@app.get("/health")
def health():
    return {"status": "Market AI Server Running"}

@app.get("/analyze")
def analyze(
    symbol: str = Query("RELIANCE.NS", description="Stock Symbol Search")
):
    timeframes = {
        "1H": ("60d", "1h"),
        "4H": ("180d", "4h"),
        "Daily": ("1y", "1d"),
        "Weekly": ("5y", "1wk"),
        "Monthly": ("10y", "1mo"),
    }

    results = {}
    daily_df = None

    for tf, (period, interval) in timeframes.items():
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False
        )

        if df.empty:
            continue

        df = clean_dataframe(df)
        df = add_ema(df)
        df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

        results[tf] = {
            "Price": round(float(df.iloc[-1]["Close"]), 2),
            "EMA_Cross": ema_crossover_signal(df),
            "Trend": trend_strength(df),
            "RSI": round(float(df.iloc[-1]["RSI"]), 2)
        }

        if tf == "Daily":
            daily_df = df

    # ------------------------------
    # DAILY PREDICTION ONLY
    # ------------------------------
    prediction = {}
    if daily_df is not None:
        cross = ema_crossover_signal(daily_df)
        trend = trend_strength(daily_df)
        stats = statistical_probability(daily_df)

        score = 0
        if cross == "BULLISH CROSSOVER":
            score += 1
        if trend == "UPTREND":
            score += 1
        if daily_df.iloc[-1]["RSI"] > 50:
            score += 1

        signal_map = {
            3: "STRONG BUY",
            2: "BUY",
            1: "HOLD",
            0: "SELL"
        }

        prediction = {
            "Based_On": "Daily Chart",
            "Signal": signal_map.get(score, "HOLD"),
            "Probability": stats
        }

    return {
        "Symbol": symbol,
        "Multi_Timeframe_Analysis": results,
        "Daily_Prediction": prediction
    }
