from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI – Stable Final")

# =========================
# Utility
# =========================
def safe_last(series):
    if series is None or len(series) < 2:
        return None
    return float(series.iloc[-1])

def clean_df(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

# =========================
# Trend Logics
# =========================
def ema_structure(df):
    return df["EMA_5"].iloc[-1] > df["EMA_10"].iloc[-1] > df["EMA_20"].iloc[-1]

def dow_trend(df):
    if len(df) < 10:
        return "NEUTRAL"
    highs = df["High"].rolling(5).max()
    lows = df["Low"].rolling(5).min()
    if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
        return "UPTREND"
    if highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
        return "DOWNTREND"
    return "SIDEWAYS"

def guppy(df):
    fast = [3,5,8,10,12,15]
    slow = [30,35,40,45,50,60]
    fast_avg = np.mean([EMAIndicator(df["Close"], p).ema_indicator().iloc[-1] for p in fast])
    slow_avg = np.mean([EMAIndicator(df["Close"], p).ema_indicator().iloc[-1] for p in slow])
    if fast_avg > slow_avg:
        return "BULLISH"
    if fast_avg < slow_avg:
        return "BEARISH"
    return "NEUTRAL"

# =========================
# Self-Learning Engine
# =========================
accuracy_log = []

def update_accuracy(signal, future_move):
    accuracy_log.append(1 if signal == future_move else 0)
    if len(accuracy_log) > 100:
        accuracy_log.pop(0)

def accuracy():
    if not accuracy_log:
        return 50
    return round(sum(accuracy_log)/len(accuracy_log)*100,2)

# =========================
# Analyzer
# =========================
def analyze_tf(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty or len(df) < 30:
        return None

    df = clean_df(df)
    df["EMA_5"] = EMAIndicator(df["Close"], 5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], 10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["VWAP"] = vwap(df)

    score = 0
    if ema_structure(df): score += 1
    if df["Close"].iloc[-1] > df["VWAP"].iloc[-1]: score += 1
    if dow_trend(df) == "UPTREND": score += 1
    if guppy(df) == "BULLISH": score += 1

    if score >= 3:
        signal = "BUY"
    elif score <= 1:
        signal = "SELL"
    else:
        signal = "HOLD"

    confidence = round((score/4)*100,2)

    return {
        "Price": round(df["Close"].iloc[-1],2),
        "VWAP": round(df["VWAP"].iloc[-1],2),
        "EMA_5": round(df["EMA_5"].iloc[-1],2),
        "EMA_10": round(df["EMA_10"].iloc[-1],2),
        "EMA_20": round(df["EMA_20"].iloc[-1],2),
        "Dow": dow_trend(df),
        "Guppy": guppy(df),
        "Signal": signal,
        "Confidence_%": confidence
    }

# =========================
# API
# =========================
@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_tf(symbol,"1d","6mo")
    weekly = analyze_tf(symbol,"1wk","2y")
    four_hr = analyze_tf(symbol,"240m","60d")
    seventy_five = analyze_tf(symbol,"75m","30d")

    if not daily:
        return {"error":"No data"}

    final_signal = daily["Signal"]
    update_accuracy(final_signal,"BUY")  # placeholder learning

    return {
        "symbol": symbol,
        "final_signal": final_signal,
        "model_accuracy_%": accuracy(),
        "analysis":{
            "Daily":daily,
            "Weekly":weekly,
            "4H":four_hr,
            "75Min":seventy_five
        }
    }

# =========================
# Frontend
# =========================
app.mount("/", StaticFiles(directory="static", html=True), name="static")
