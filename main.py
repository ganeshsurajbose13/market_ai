from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI – Adaptive EMA Learning Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTILITIES ----------------
def ema(series, p):
    return EMAIndicator(series, p).ema_indicator()

def clean(df):
    df = df[["Open","High","Low","Close","Volume"]]
    df.dropna(inplace=True)
    return df

# ---------------- LEARNING ENGINE ----------------
def ema_learning_score(df):
    score = 0
    success = 0
    total = 0

    e5 = ema(df["Close"],5)
    e10 = ema(df["Close"],10)

    for i in range(-50, -1):
        if e5.iloc[i-1] < e10.iloc[i-1] and e5.iloc[i] > e10.iloc[i]:
            total += 1
            if df["Close"].iloc[i+3] > df["Close"].iloc[i]:
                success += 1

        if e5.iloc[i-1] > e10.iloc[i-1] and e5.iloc[i] < e10.iloc[i]:
            total += 1
            if df["Close"].iloc[i+3] < df["Close"].iloc[i]:
                success += 1

    confidence = round((success / total) * 100, 2) if total > 0 else 50
    return confidence

# ---------------- TREND IDENTIFIER ----------------
def trend_engine(df):
    close = df["Close"]

    e5 = ema(close,5)
    e10 = ema(close,10)
    e20 = ema(close,20)
    e50 = ema(close,50)

    price = round(float(close.iloc[-1]),2)

    trend_points = 0

    if e5.iloc[-1] > e10.iloc[-1]:
        trend_points += 1
    if e10.iloc[-1] > e20.iloc[-1]:
        trend_points += 1
    if e20.iloc[-1] > e50.iloc[-1]:
        trend_points += 1

    if e5.iloc[-1] < e10.iloc[-1]:
        trend_points -= 1
    if e10.iloc[-1] < e20.iloc[-1]:
        trend_points -= 1
    if e20.iloc[-1] < e50.iloc[-1]:
        trend_points -= 1

    confidence = ema_learning_score(df)

    if trend_points >= 3:
        signal = "BUY"
    elif trend_points <= -3:
        signal = "SELL"
    else:
        signal = "HOLD"

    strength = abs(trend_points) * 33.33

    return {
        "Price": price,
        "EMA_5": round(float(e5.iloc[-1]),2),
        "EMA_10": round(float(e10.iloc[-1]),2),
        "EMA_20": round(float(e20.iloc[-1]),2),
        "EMA_50": round(float(e50.iloc[-1]),2),
        "Trend_Points": trend_points,
        "Trend_Strength_%": round(strength,2),
        "Learning_Confidence_%": confidence,
        "Signal": signal
    }

# ---------------- ANALYSIS ----------------
def analyze(symbol, period, interval):
    df = yf.download(symbol, period=period, interval=interval, progress=False)

    if df.empty or len(df) < 60:
        return {"error": "No data"}

    df = clean(df)
    return trend_engine(df)

# ---------------- API ----------------
@app.get("/analyze")
def analyze_stock(symbol: str):
    return {
        "symbol": symbol,
        "strategy": "Adaptive EMA Learning Engine",
        "analysis": {
            "Daily": analyze(symbol,"6mo","1d"),
            "Weekly": analyze(symbol,"2y","1wk"),
            "Monthly": analyze(symbol,"5y","1mo"),
            "Intraday_75m": analyze(symbol,"60d","75m"),
            "Intraday_4H": analyze(symbol,"60d","240m")
        }
    }

# ---------------- FRONTEND ----------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
