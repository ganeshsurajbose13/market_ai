from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable Core")

# -----------------------------
# Utilities
# -----------------------------

def clean_df(df):
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.dropna(inplace=True)
    return df

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema(df, period):
    return EMAIndicator(df["Close"], window=period).ema_indicator()

def ema_cross(fast, slow):
    if fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]:
        return "BULLISH"
    if fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1]:
        return "BEARISH"
    return "NONE"

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = "RELIANCE.NS"):
    try:
        tf_data = {}

        for tf, interval in {
            "Daily": "1d",
            "Weekly": "1wk",
            "Monthly": "1mo"
        }.items():

            df = yf.download(
                symbol,
                period="6mo",
                interval=interval,
                progress=False
            )

            if df.empty:
                continue

            df = clean_df(df)

            df["EMA5"] = ema(df, 5)
            df["EMA10"] = ema(df, 10)
            df["EMA20"] = ema(df, 20)
            df["EMA50"] = ema(df, 50)
            df["VWAP"] = vwap(df)

            cross = ema_cross(df["EMA5"], df["EMA10"])

            last = df.iloc[-1]

            tf_data[tf] = {
                "Price": round(float(last["Close"]), 2),
                "VWAP": round(float(last["VWAP"]), 2),
                "EMA_Cross_5_10": cross
            }

        return {
            "symbol": symbol,
            "analysis": tf_data
        }

    except Exception as e:
        return {"error": str(e)}
