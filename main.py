from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

app = FastAPI(title="Market AI Engine")

# -----------------------------
# Utilities
# -----------------------------

def clean_df(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)
    return df

def vwap(df):
    price = (df["High"] + df["Low"] + df["Close"]) / 3
    return (price * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema(df, period):
    return EMAIndicator(df["Close"], period).ema_indicator()

def ema_cross(df, short, long):
    return (
        df[f"EMA_{short}"].iloc[-2] < df[f"EMA_{long}"].iloc[-2]
        and df[f"EMA_{short}"].iloc[-1] > df[f"EMA_{long}"].iloc[-1]
    )

# -----------------------------
# API
# -----------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/health")
def health():
    return {"health": "OK"}

@app.get("/analyze")
def analyze(symbol: str = "RELIANCE.NS"):

    df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    if df.empty:
        return {"error": "No data"}

    df = clean_df(df)

    # Indicators
    df["EMA_5"] = ema(df, 5)
    df["EMA_10"] = ema(df, 10)
    df["EMA_20"] = ema(df, 20)
    df["EMA_50"] = ema(df, 50)
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()
    df["VWAP"] = vwap(df)

    last = df.iloc[-1]

    crosses = {
        "EMA_5_10": ema_cross(df, 5, 10),
        "EMA_10_20": ema_cross(df, 10, 20),
        "EMA_20_50": ema_cross(df, 20, 50),
    }

    score = sum(crosses.values())
    probability = round((score / 3) * 100, 2)

    signal = (
        "STRONG BUY" if probability >= 70 else
        "BUY" if probability >= 50 else
        "HOLD"
    )

    return {
        "symbol": symbol,
        "price": round(float(last["Close"]), 2),
        "RSI": round(float(last["RSI"]), 2),
        "VWAP": round(float(last["VWAP"]), 2),
        "EMA_Cross": crosses,
        "Probability_%": probability,
        "Signal": signal
    }
