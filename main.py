from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

app = FastAPI(title="Market AI Engine – Stable")

# -----------------------------
# Utility Functions
# -----------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df

def ema_cross_signal(df: pd.DataFrame):
    ema5 = EMAIndicator(df["Close"], window=5).ema_indicator()
    ema10 = EMAIndicator(df["Close"], window=10).ema_indicator()
    ema20 = EMAIndicator(df["Close"], window=20).ema_indicator()

    if ema5.iloc[-2] < ema10.iloc[-2] and ema5.iloc[-1] > ema10.iloc[-1]:
        return "BULLISH CROSS"
    elif ema5.iloc[-2] > ema10.iloc[-2] and ema5.iloc[-1] < ema10.iloc[-1]:
        return "BEARISH CROSS"
    else:
        return "NO CROSS"

def probability_score(rsi, ema_signal):
    score = 0
    if 50 < rsi < 70:
        score += 50
    if ema_signal == "BULLISH CROSS":
        score += 30
    return min(score, 90)

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/analyze")
def analyze(symbol: str = Query("RELIANCE.NS")):
    df = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        progress=False
    )

    if df.empty:
        return {"error": "No data found"}

    df = clean_dataframe(df)

    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    ema_signal = ema_cross_signal(df)
    last = df.iloc[-1]

    probability = probability_score(last["RSI"], ema_signal)

    return {
        "symbol": symbol,
        "price": round(float(last["Close"]), 2),
        "RSI": round(float(last["RSI"]), 2),
        "EMA_Cross": ema_signal,
        "Prediction_Probability_%": probability
    }
