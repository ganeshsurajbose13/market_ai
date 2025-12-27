from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine", version="1.0")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(inplace=True)
    return df

def add_ema(df: pd.DataFrame):
    df["EMA_5"] = EMAIndicator(df["Close"], window=5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    return df

def ema_crossover_signal(df: pd.DataFrame) -> str:
    prev = df.iloc[-2]
    last = df.iloc[-1]

    if prev["EMA_5"] < prev["EMA_10"] and last["EMA_5"] > last["EMA_10"]:
        return "BULLISH CROSS"
    elif prev["EMA_5"] > prev["EMA_10"] and last["EMA_5"] < last["EMA_10"]:
        return "BEARISH CROSS"
    return "NO CROSS"

def probability_prediction(df: pd.DataFrame) -> float:
    """
    Simple statistical probability:
    % of last 20 candles closing above EMA 20
    """
    recent = df.tail(20)
    wins = (recent["Close"] > recent["EMA_20"]).sum()
    probability = (wins / 20) * 100
    return round(probability, 2)

# --------------------------------------------------
# Health Check (Render needs this)
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "OK"}

# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

# --------------------------------------------------
# MAIN ANALYSIS ENDPOINT
# --------------------------------------------------

@app.get("/analyze")
def analyze_stock(
    symbol: str = Query("RELIANCE.NS", description="Stock symbol"),
):
    """
    Uses multiple timeframes
    Prediction is based ONLY on Daily chart
    """

    timeframes = {
        "1h": ("7d", "1h"),
        "4h": ("60d", "4h"),
        "1d": ("6mo", "1d"),
        "1wk": ("2y", "1wk"),
        "1mo": ("5y", "1mo"),
    }

    results = {}

    for tf, (period, interval) in timeframes.items():
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False
        )

        if df.empty:
            results[tf] = "NO DATA"
            continue

        df = clean_df(df)
        df = add_ema(df)

        signal = ema_crossover_signal(df)
        trend = "UPTREND" if df["Close"].iloc[-1] > df["EMA_20"].iloc[-1] else "DOWNTREND"

        results[tf] = {
            "price": round(float(df["Close"].iloc[-1]), 2),
            "ema_5": round(float(df["EMA_5"].iloc[-1]), 2),
            "ema_10": round(float(df["EMA_10"].iloc[-1]), 2),
            "ema_20": round(float(df["EMA_20"].iloc[-1]), 2),
            "trend": trend,
            "ema_cross": signal,
        }

        # Prediction ONLY on daily
        if tf == "1d":
            prob = probability_prediction(df)
            results[tf]["prediction_probability"] = f"{prob}%"
            results[tf]["prediction"] = (
                "HIGH PROBABILITY BUY" if prob > 65 else
                "NEUTRAL" if prob > 45 else
                "HIGH RISK"
            )

    return {
        "symbol": symbol,
        "analysis": results
    }
