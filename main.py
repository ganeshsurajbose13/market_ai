from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable FINAL")

# -----------------------------
# Utility Functions
# -----------------------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df


def calculate_vwap(df: pd.DataFrame):
    if df.empty:
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def safe_last(series):
    if series is None or len(series) < 1:
        return None
    return round(float(series.iloc[-1]), 2)


def ema_cross(df, s, l):
    if len(df) < l + 2:
        return "NA"
    es = EMAIndicator(df["Close"], s).ema_indicator()
    el = EMAIndicator(df["Close"], l).ema_indicator()
    if es.iloc[-2] < el.iloc[-2] and es.iloc[-1] > el.iloc[-1]:
        return "BULLISH"
    if es.iloc[-2] > el.iloc[-2] and es.iloc[-1] < el.iloc[-1]:
        return "BEARISH"
    return "NO CROSS"


def guppy_trend(df):
    if len(df) < 60:
        return "NA"
    short = [EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [3,5,8,10,12,15]]
    long = [EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [30,35,40,45,50,60]]
    return "BULLISH GMMA" if np.mean(short) > np.mean(long) else "BEARISH GMMA"


def analyze_tf(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    df = clean_df(df)
    if df.empty:
        return {"error": "No data"}

    df["EMA_5"] = EMAIndicator(df["Close"], 5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], 10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["VWAP"] = calculate_vwap(df)

    return {
        "Price": safe_last(df["Close"]),
        "VWAP": safe_last(df["VWAP"]),
        "EMA_5": safe_last(df["EMA_5"]),
        "EMA_10": safe_last(df["EMA_10"]),
        "EMA_20": safe_last(df["EMA_20"]),
        "EMA_Cross": ema_cross(df, 5, 10),
        "Guppy": guppy_trend(df)
    }

# -----------------------------
# API ENDPOINTS
# -----------------------------
@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_tf(symbol, "1d", "6mo")
    if "error" in daily:
        return {"error": "Invalid symbol"}

    weekly = analyze_tf(symbol, "1wk", "2y")
    monthly = analyze_tf(symbol, "1mo", "5y")

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
        "confidence_%": round(score / 3 * 100, 2),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# -----------------------------
# FRONTEND (SAFE)
# -----------------------------
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
