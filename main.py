from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- HELPERS --------------------
def calculate_ema(series, period):
    return EMAIndicator(series, period).ema_indicator().iloc[-1]

def calculate_vwap(df):
    return (df["Volume"] * df["Close"]).sum() / df["Volume"].sum()

def guppy_trend(close):
    short_emas = [3, 5, 8, 10, 12, 15]
    long_emas = [30, 35, 40, 45, 50, 60]

    short_avg = np.mean([calculate_ema(close, p) for p in short_emas])
    long_avg = np.mean([calculate_ema(close, p) for p in long_emas])

    if short_avg > long_avg:
        return "BULLISH GMMA"
    elif short_avg < long_avg:
        return "BEARISH GMMA"
    else:
        return "NEUTRAL"

def analyze(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty or len(df) < 60:
        return {"error": "No data"}

    close = df["Close"]

    return {
        "Price": round(close.iloc[-1], 2),
        "VWAP": round(calculate_vwap(df), 2),
        "EMA_5": round(calculate_ema(close, 5), 2),
        "EMA_10": round(calculate_ema(close, 10), 2),
        "EMA_20": round(calculate_ema(close, 20), 2),
        "Guppy_Trend": guppy_trend(close)
    }

# -------------------- API --------------------
@app.get("/analyze")
def full_analysis(symbol: str):
    return {
        "symbol": symbol,
        "analysis": {
            "Daily": analyze(symbol, "6mo", "1d"),
            "Weekly": analyze(symbol, "1y", "1wk"),
            "Monthly": analyze(symbol, "5y", "1mo"),
            "Intraday_75m": analyze(symbol, "60d", "75m"),
            "Intraday_4H": analyze(symbol, "60d", "4h"),
        }
    }

# -------------------- FRONTEND --------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
