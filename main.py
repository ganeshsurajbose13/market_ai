from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable v3")

# -----------------------------
# Utility safety helpers
# -----------------------------
def safe_last(series):
    if series is None or len(series) < 2:
        return None
    return round(float(series.iloc[-1]), 2)

def clean_df(df):
    if df.empty:
        return df
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
    if len(df) < l + 2:
        return "NO DATA"
    ema_s = EMAIndicator(df["Close"], s).ema_indicator()
    ema_l = EMAIndicator(df["Close"], l).ema_indicator()
    if ema_s.iloc[-2] < ema_l.iloc[-2] and ema_s.iloc[-1] > ema_l.iloc[-1]:
        return "BULLISH"
    if ema_s.iloc[-2] > ema_l.iloc[-2] and ema_s.iloc[-1] < ema_l.iloc[-1]:
        return "BEARISH"
    return "NO CROSS"

def guppy_trend(df):
    if len(df) < 60:
        return "NO DATA"
    short = np.mean([EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [3,5,8,10,12,15]])
    long = np.mean([EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [30,35,40,45,50,60]])
    return "BULLISH GMMA" if short > long else "BEARISH GMMA"

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

    return {
        "Price": safe_last(df["Close"]),
        "VWAP": safe_last(df["VWAP"]),
        "EMA_5": safe_last(df["EMA_5"]),
        "EMA_10": safe_last(df["EMA_10"]),
        "EMA_20": safe_last(df["EMA_20"]),
        "EMA_50": safe_last(df["EMA_50"]),
        "EMA_Cross": ema_cross(df, 5, 10),
        "Guppy": guppy_trend(df)
    }

# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_tf(symbol, "1d", "6mo")
    weekly = analyze_tf(symbol, "1wk", "2y")
    monthly = analyze_tf(symbol, "1mo", "5y")
    h1 = analyze_tf(symbol, "60m", "7d")

    if "error" in daily:
        return {"error": "Invalid symbol"}

    score = 0
    if daily["EMA_5"] and daily["EMA_10"] and daily["EMA_5"] > daily["EMA_10"]:
        score += 1
    if daily["EMA_10"] and daily["EMA_20"] and daily["EMA_10"] > daily["EMA_20"]:
        score += 1
    if daily["Price"] and daily["VWAP"] and daily["Price"] > daily["VWAP"]:
        score += 1

    signal = "BUY" if score == 3 else "SELL" if score == 0 else "HOLD"

    return {
        "symbol": symbol,
        "signal": signal,
        "confidence_%": round((score/3)*100, 2),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly,
            "Intraday_1H": h1
        }
    }

# -----------------------------
# Serve frontend SAFELY
# -----------------------------
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
