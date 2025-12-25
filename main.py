from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

app = FastAPI(title="Market AI Engine – Advanced")

# -----------------------------
# Utility Functions
# -----------------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    price = (df["High"] + df["Low"] + df["Close"]) / 3
    return (price * df["Volume"]).cumsum() / df["Volume"].cumsum()

def dow_trend(df: pd.DataFrame) -> str:
    recent_high = df["High"].rolling(5).max()
    recent_low = df["Low"].rolling(5).min()
    if df["Close"].iloc[-1] > recent_high.iloc[-2]:
        return "UPTREND"
    elif df["Close"].iloc[-1] < recent_low.iloc[-2]:
        return "DOWNTREND"
    return "SIDEWAYS"

def wave_theory(df: pd.DataFrame) -> str:
    momentum = df["Close"].pct_change().rolling(5).mean().iloc[-1]
    if momentum > 0.01:
        return "IMPULSE WAVE"
    elif momentum < -0.01:
        return "CORRECTIVE WAVE"
    return "NEUTRAL"

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/analyze")
def analyze(symbol: str = "RELIANCE.NS"):
    df = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        return {"error": "No data found for symbol"}

    df = clean_dataframe(df)

    # Technical Indicators
    df["SMA_20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["VWAP"] = calculate_vwap(df)

    last = df.iloc[-1]
    score = 0

    # Scoring Logic
    if last["Close"] > last["VWAP"]:
        score += 1
    if last["Close"] > last["EMA_20"]:
        score += 1
    if 50 < last["RSI"] < 70:
        score += 1

    trend = dow_trend(df)
    wave = wave_theory(df)

    if trend == "UPTREND":
        score += 1
    if wave == "IMPULSE WAVE":
        score += 1

    signal_map = {
        5: "STRONG BUY",
        4: "BUY",
        3: "HOLD",
        2: "SELL",
        1: "STRONG SELL"
    }

    return {
        "symbol": symbol,
        "price": round(float(last["Close"]), 2),
        "VWAP": round(float(last["VWAP"]), 2),
        "RSI": round(float(last["RSI"]), 2),
        "Trend": trend,
        "Wave": wave,
        "Score": score,
        "Signal": signal_map.get(score, "HOLD")
    }
