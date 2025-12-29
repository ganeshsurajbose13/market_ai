from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI – Stable Final")

# =========================
# DATA HELPERS
# =========================

def clean_df(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema_cross(df, s, l):
    if len(df) < l + 2:
        return "NA"
    ema_s = EMAIndicator(df["Close"], s).ema_indicator()
    ema_l = EMAIndicator(df["Close"], l).ema_indicator()
    if ema_s.iloc[-2] < ema_l.iloc[-2] and ema_s.iloc[-1] > ema_l.iloc[-1]:
        return "BULLISH"
    if ema_s.iloc[-2] > ema_l.iloc[-2] and ema_s.iloc[-1] < ema_l.iloc[-1]:
        return "BEARISH"
    return "NONE"

def guppy(df):
    short = [EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [3,5,8,10,12,15]]
    long = [EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [30,35,40,45,50,60]]
    return "BULLISH" if np.mean(short) > np.mean(long) else "BEARISH"

def analyze_tf(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty or len(df) < 60:
        return {"error": "No data"}
    df = clean_df(df)

    df["EMA5"] = EMAIndicator(df["Close"], 5).ema_indicator()
    df["EMA10"] = EMAIndicator(df["Close"], 10).ema_indicator()
    df["EMA20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["VWAP"] = vwap(df)

    return {
        "Price": round(df["Close"].iloc[-1], 2),
        "VWAP": round(df["VWAP"].iloc[-1], 2),
        "EMA5": round(df["EMA5"].iloc[-1], 2),
        "EMA10": round(df["EMA10"].iloc[-1], 2),
        "EMA20": round(df["EMA20"].iloc[-1], 2),
        "EMA_Cross": ema_cross(df, 5, 10),
        "Guppy": guppy(df)
    }

# =========================
# API ROUTES
# =========================

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_tf(symbol, "1d", "6mo")
    weekly = analyze_tf(symbol, "1wk", "2y")
    monthly = analyze_tf(symbol, "1mo", "5y")

    if "error" in daily:
        return {"error": "Invalid symbol or no data"}

    score = 0
    if daily["EMA5"] > daily["EMA10"]: score += 1
    if daily["EMA10"] > daily["EMA20"]: score += 1
    if daily["Price"] > daily["VWAP"]: score += 1

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

# =========================
# STATIC UI (LAST LINE ONLY)
# =========================

app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")
