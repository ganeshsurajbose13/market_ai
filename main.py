from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Market AI Engine – Stable v2 with Frontend")

# -----------------------------
# Utility Functions
# -----------------------------

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    return (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema_cross(df: pd.DataFrame, short: int, long: int) -> str:
    ema_short = EMAIndicator(df["Close"], short).ema_indicator()
    ema_long = EMAIndicator(df["Close"], long).ema_indicator()

    if ema_short.iloc[-2] < ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]:
        return "BULLISH CROSS"
    elif ema_short.iloc[-2] > ema_long.iloc[-2] and ema_short.iloc[-1] < ema_long.iloc[-1]:
        return "BEARISH CROSS"
    else:
        return "NO CROSS"

# -----------------------------
# Guppy GMMA and SuperTrend placeholders
# -----------------------------
def guppy_trend(df: pd.DataFrame) -> str:
    # Simplified GMMA logic for demonstration
    short_emas = [EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [3,5,8,10,12,15]]
    long_emas = [EMAIndicator(df["Close"], x).ema_indicator().iloc[-1] for x in [30,35,40,45,50,60]]
    if np.mean(short_emas) > np.mean(long_emas):
        return "BULLISH GMMA"
    else:
        return "BEARISH GMMA"

def supertrend(df: pd.DataFrame) -> str:
    # Placeholder: return NEUTRAL for simplicity
    return "NEUTRAL"

# -----------------------------
# Analyze function
# -----------------------------
def analyze_timeframe(symbol: str, interval: str, period: str) -> dict:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        return {"error": "No data"}

    df = clean_df(df)

    df["EMA_5"] = EMAIndicator(df["Close"], 5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"], 10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"], 20).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["Close"], 50).ema_indicator()
    df["VWAP"] = calculate_vwap(df)

    return {
        "Price": round(float(df["Close"].iloc[-1]),2),
        "VWAP": round(float(df["VWAP"].iloc[-1]),2),
        "EMA_5": round(float(df["EMA_5"].iloc[-1]),2),
        "EMA_10": round(float(df["EMA_10"].iloc[-1]),2),
        "EMA_20": round(float(df["EMA_20"].iloc[-1]),2),
        "EMA_50": round(float(df["EMA_50"].iloc[-1]),2),
        "EMA_5_10_Cross": ema_cross(df,5,10),
        "Guppy_Trend": guppy_trend(df),
        "SuperTrend": supertrend(df)
    }

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "Market AI Server Running"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(..., description="Stock symbol like RELIANCE.NS")):
    # Multi-timeframe analysis
    daily = analyze_timeframe(symbol, "1d", "6mo")
    weekly = analyze_timeframe(symbol, "1wk", "2y")
    monthly = analyze_timeframe(symbol, "1mo", "5y")
    intraday_1h = analyze_timeframe(symbol, "60m", "7d")
    intraday_4h = analyze_timeframe(symbol, "240m", "60d")  # may fail if no data

    if "error" in daily:
        return {"error": "Invalid symbol or no data"}

    # SIMPLE DAILY PREDICTION LOGIC
    score = 0
    if daily["EMA_5"] > daily["EMA_10"]:
        score += 1
    if daily["EMA_10"] > daily["EMA_20"]:
        score += 1
    if daily["Price"] > daily["VWAP"]:
        score += 1

    signal = "HOLD"
    if score == 3:
        signal = "BUY"
    elif score == 0:
        signal = "SELL"

    probability = round((score/3)*100,2)

    # ML placeholder
    ml_prediction = {"prob_up": None, "prob_down": None, "note": "ML model not loaded"}

    return {
        "symbol": symbol,
        "prediction_based_on": "DAILY",
        "signal": signal,
        "probability_%": probability,
        "ml_prediction": ml_prediction,
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly,
            "Intraday_1H": intraday_1h,
            "Intraday_4H": intraday_4h
        }
    }

# -----------------------------
# Serve frontend
# -----------------------------
app.mount("/", StaticFiles(directory=".", html=True), name="static")
