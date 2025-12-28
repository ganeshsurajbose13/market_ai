from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable v2")

# -----------------------------
# Utility Functions
# -----------------------------

def clean_df(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema_cross(df, s, l):
    e1 = EMAIndicator(df["Close"], s).ema_indicator()
    e2 = EMAIndicator(df["Close"], l).ema_indicator()
    if e1.iloc[-2] < e2.iloc[-2] and e1.iloc[-1] > e2.iloc[-1]:
        return "BULLISH"
    if e1.iloc[-2] > e2.iloc[-2] and e1.iloc[-1] < e2.iloc[-1]:
        return "BEARISH"
    return "NO CROSS"

def analyze_tf(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        return {"error": "No data"}

    df = clean_df(df)

    df["EMA_5"] = EMAIndicator(df["Close"], 5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], 10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["VWAP"] = calculate_vwap(df)

    return {
        "Price": round(float(df["Close"].iloc[-1]), 2),
        "VWAP": round(float(df["VWAP"].iloc[-1]), 2),
        "EMA_5": round(float(df["EMA_5"].iloc[-1]), 2),
        "EMA_10": round(float(df["EMA_10"].iloc[-1]), 2),
        "EMA_20": round(float(df["EMA_20"].iloc[-1]), 2),
        "EMA_Cross": ema_cross(df, 5, 10)
    }

# -----------------------------
# API
# -----------------------------

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_tf(symbol, "1d", "6mo")
    weekly = analyze_tf(symbol, "1wk", "2y")
    monthly = analyze_tf(symbol, "1mo", "5y")

    if "error" in daily:
        return {"error": "Invalid symbol"}

    score = 0
    if daily["EMA_5"] > daily["EMA_10"]:
        score += 1
    if daily["EMA_10"] > daily["EMA_20"]:
        score += 1
    if daily["Price"] > daily["VWAP"]:
        score += 1

    signal = "BUY" if score == 3 else "SELL" if score == 0 else "HOLD"

    return {
        "symbol": symbol.upper(),
        "signal": signal,
        "probability": round(score / 3 * 100, 2),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# -----------------------------
# FRONTEND
# -----------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
