from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI(title="Market AI")

# -----------------------------
# Utility Functions
# -----------------------------

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def guppy_ema(df):
    short = [3,5,8,10,12,15]
    long = [30,35,40,45,50,60]

    for p in short:
        df[f"EMA_S_{p}"] = ema(df["Close"], p)

    for p in long:
        df[f"EMA_L_{p}"] = ema(df["Close"], p)

    return df

def identify_trend(df):
    if df.empty or len(df) < 60:
        return "NO DATA"

    last = df.iloc[-1]

    short_avg = np.mean([last[f"EMA_S_{p}"] for p in [3,5,8,10,12,15]])
    long_avg  = np.mean([last[f"EMA_L_{p}"] for p in [30,35,40,45,50,60]])

    if short_avg > long_avg:
        return "UPTREND"
    elif short_avg < long_avg:
        return "DOWNTREND"
    else:
        return "SIDEWAYS"

def dow_confirmation(df):
    hh = df["High"].rolling(20).max()
    ll = df["Low"].rolling(20).min()

    if df["Close"].iloc[-1] > hh.iloc[-2]:
        return "Higher High (Bullish)"
    elif df["Close"].iloc[-1] < ll.iloc[-2]:
        return "Lower Low (Bearish)"
    else:
        return "Range"

def fetch_data(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    df.dropna(inplace=True)
    return df

def resample_tf(df, rule):
    ohlc = {
        'Open':'first',
        'High':'max',
        'Low':'min',
        'Close':'last',
        'Volume':'sum'
    }
    return df.resample(rule).apply(ohlc).dropna()

# -----------------------------
# API ROUTE
# -----------------------------

@app.get("/analyze")
def analyze(symbol: str):
    try:
        results = {}

        # ---- Daily ----
        daily = fetch_data(symbol, "1d", "6mo")
        daily = guppy_ema(daily)
        results["Daily"] = {
            "Trend": identify_trend(daily),
            "Dow": dow_confirmation(daily)
        }

        # ---- Weekly ----
        weekly = fetch_data(symbol, "1wk", "1y")
        weekly = guppy_ema(weekly)
        results["Weekly"] = {
            "Trend": identify_trend(weekly),
            "Dow": dow_confirmation(weekly)
        }

        # ---- 4H (resampled) ----
        h1 = fetch_data(symbol, "60m", "60d")
        h4 = resample_tf(h1, "4H")
        h4 = guppy_ema(h4)
        results["4H"] = {
            "Trend": identify_trend(h4),
            "Dow": dow_confirmation(h4)
        }

        # ---- 75 Min (resampled) ----
        m15 = fetch_data(symbol, "15m", "30d")
        m75 = resample_tf(m15, "75T")
        m75 = guppy_ema(m75)
        results["75Min"] = {
            "Trend": identify_trend(m75),
            "Dow": dow_confirmation(m75)
        }

        return JSONResponse({
            "symbol": symbol,
            "analysis": results
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
# Serve Frontend
# -----------------------------

app.mount("/", StaticFiles(directory="static", html=True), name="static")
