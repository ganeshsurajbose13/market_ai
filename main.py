from fastapi import FastAPI
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from stocktrends import Renko

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

def renko_signal(df: pd.DataFrame, brick_size: float = 2) -> str:
    df_reset = df[["Open","High","Low","Close","Volume"]].copy()
    df_reset = df_reset.reset_index()
    renko = Renko(df_reset)
    renko.brick_size = brick_size
    renko_df = renko.get_ohlc_data()
    if renko_df.empty:
        return "HOLD"
    if renko_df["uptrend"].iloc[-1]:
        return "BUY"
    else:
        return "SELL"

# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/health")
def health():
    return {
        "status": "OK",
        "service": "Market AI",
        "version": "1.0.0"
    }

@app.get("/analyze")
def analyze(symbol: str = "RELIANCE.NS"):
    # -----------------------------
    # Download Data
    # -----------------------------
    df = yf.download(symbol, period="6mo", interval="1d", progress=False)
    if df.empty:
        return {"error": "No data found for symbol"}
    
    df = clean_dataframe(df)

    # -----------------------------
    # Indicators
    # -----------------------------
    # VWAP
    df["VWAP"] = calculate_vwap(df)
    # RSI
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    # EMA 5 & 10
    df["EMA_5"] = EMAIndicator(df["Close"], window=5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], window=10).ema_indicator()
    # GMMA (short-term EMAs)
    short_emas = [3,5,8,10,12,15]
    for s in short_emas:
        df[f"EMA_{s}"] = EMAIndicator(df["Close"], window=s).ema_indicator()
    # GMMA (long-term EMAs)
    long_emas = [30,35,40,45,50,60]
    for l in long_emas:
        df[f"EMA_{l}"] = EMAIndicator(df["Close"], window=l).ema_indicator()

    last = df.iloc[-1]

    # -----------------------------
    # EMA Crossover Signal
    # -----------------------------
    if last["EMA_5"] > last["EMA_10"]:
        ema_signal = "BUY"
    elif last["EMA_5"] < last["EMA_10"]:
        ema_signal = "SELL"
    else:
        ema_signal = "HOLD"

    # -----------------------------
    # GMMA Trend Signal
    # -----------------------------
    short_mean = df[[f"EMA_{s}" for s in short_emas]].iloc[-1].mean()
    long_mean = df[[f"EMA_{l}" for l in long_emas]].iloc[-1].mean()
    gmma_signal = "UPTREND" if short_mean > long_mean else "DOWNTREND"

    # -----------------------------
    # Renko Signal
    # -----------------------------
    renko_sig = renko_signal(df)

    # -----------------------------
    # Score and Combine Signals
    # -----------------------------
    score = 0
    if last["Close"] > last["VWAP"]:
        score += 1
    if 50 < last["RSI"] < 70:
        score += 1
    if gmma_signal == "UPTREND":
        score += 1
    if ema_signal == "BUY":
        score += 1
    if renko_sig == "BUY":
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
        "EMA_5": round(float(last["EMA_5"]), 2),
        "EMA_10": round(float(last["EMA_10"]), 2),
        "GMMA_Trend": gmma_signal,
        "Renko_Signal": renko_sig,
        "EMA_Signal": ema_signal,
        "Score": score,
        "Final_Signal": signal_map.get(score, "HOLD")
    }
