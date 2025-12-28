from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable v3")

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
    return "NONE"

def trend_strength(df):
    e5 = EMAIndicator(df["Close"], 5).ema_indicator().iloc[-1]
    e20 = EMAIndicator(df["Close"], 20).ema_indicator().iloc[-1]
    e50 = EMAIndicator(df["Close"], 50).ema_indicator().iloc[-1]

    score = 0
    if e5 > e20: score += 1
    if e20 > e50: score += 1

    strength = int((score / 2) * 100)
    signal = "BUY" if strength == 100 else "SELL" if strength == 0 else "HOLD"
    return strength, signal

# -----------------------------
# Analysis Core
# -----------------------------

def analyze_tf(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        return {"error": "No data"}

    df = clean_df(df)
    df["EMA_5"] = EMAIndicator(df["Close"], 5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], 10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["VWAP"] = calculate_vwap(df)

    strength, signal = trend_strength(df)

    return {
        "Price": round(df["Close"].iloc[-1], 2),
        "EMA_5": round(df["EMA_5"].iloc[-1], 2),
        "EMA_10": round(df["EMA_10"].iloc[-1], 2),
        "EMA_20": round(df["EMA_20"].iloc[-1], 2),
        "EMA_50": round(df["EMA_50"].iloc[-1], 2),
        "VWAP": round(df["VWAP"].iloc[-1], 2),
        "Trend_Strength_%": strength,
        "Signal": signal
    }

# -----------------------------
# API
# -----------------------------

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    return {
        "symbol": symbol,
        "strategy": "Adaptive EMA Trend Learning",
        "analysis": {
            "Daily": analyze_tf(symbol, "1d", "6mo"),
            "Weekly": analyze_tf(symbol, "1wk", "2y"),
            "Monthly": analyze_tf(symbol, "1mo", "5y"),
            "Intraday_75m": analyze_tf(symbol, "75m", "15d"),
            "Intraday_4H": analyze_tf(symbol, "240m", "60d"),
        }
    }

# -----------------------------
# Frontend
# -----------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
