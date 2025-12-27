from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from scipy.stats import norm

app = FastAPI(title="Market AI – Stable Engine")

# -----------------------------
# Utility Functions
# -----------------------------

def clean_df(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema_cross_signal(df):
    ema5 = EMAIndicator(df["Close"], 5).ema_indicator()
    ema10 = EMAIndicator(df["Close"], 10).ema_indicator()
    ema20 = EMAIndicator(df["Close"], 20).ema_indicator()
    ema50 = EMAIndicator(df["Close"], 50).ema_indicator()

    if ema5.iloc[-1] > ema10.iloc[-1] > ema20.iloc[-1] > ema50.iloc[-1]:
        return "BULLISH"
    elif ema5.iloc[-1] < ema10.iloc[-1] < ema20.iloc[-1] < ema50.iloc[-1]:
        return "BEARISH"
    else:
        return "NEUTRAL"

def probability_estimation(df):
    returns = df["Close"].pct_change().dropna()
    mean = returns.mean()
    std = returns.std()
    prob_up = 1 - norm.cdf(0, mean, std)
    return round(prob_up * 100, 2)

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/health")
def health():
    return {"health": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query("RELIANCE.NS")):

    df = yf.download(
        symbol,
        period="6mo",
        interval="1d",
        progress=False
    )

    if df.empty:
        return {"error": "No data found"}

    df = clean_df(df)

    df["VWAP"] = calculate_vwap(df)
    signal = ema_cross_signal(df)
    probability = probability_estimation(df)

    last = df.iloc[-1]

    return {
        "symbol": symbol,
        "price": round(float(last["Close"]), 2),
        "VWAP": round(float(last["VWAP"]), 2),
        "EMA_Cross": signal,
        "Probability_Up_%": probability,
        "Timeframe": "Daily (Prediction based)"
    }
