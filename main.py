from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable v8 with Frontend")

# -----------------------------
# CORS for frontend
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing; later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if len(ema_short) < 2:  # avoid index error
        return "NO CROSS"
    if ema_short.iloc[-2] < ema_long.iloc[-2] and ema_short.iloc[-1] > ema_long.iloc[-1]:
        return "BULLISH CROSS"
    elif ema_short.iloc[-2] > ema_long.iloc[-2] and ema_short.iloc[-1] < ema_long.iloc[-1]:
        return "BEARISH CROSS"
    else:
        return "NO CROSS"

def guppy_ema(df: pd.DataFrame):
    short_emas = [EMAIndicator(df["Close"], span).ema_indicator() for span in [3,5,8,10,12,15]]
    long_emas  = [EMAIndicator(df["Close"], span).ema_indicator() for span in [30,35,40,45,50,60]]
    df["Guppy_Short"] = pd.concat(short_emas, axis=1).mean(axis=1)
    df["Guppy_Long"]  = pd.concat(long_emas, axis=1).mean(axis=1)
    if df["Guppy_Short"].iloc[-1] > df["Guppy_Long"].iloc[-1]:
        return "BULLISH GMMA"
    elif df["Guppy_Short"].iloc[-1] < df["Guppy_Long"].iloc[-1]:
        return "BEARISH GMMA"
    else:
        return "NEUTRAL GMMA"

def supertrend(df: pd.DataFrame, period=7, multiplier=3):
    hl2 = (df['High'] + df['Low']) / 2
    atr = df['High'].rolling(period).max() - df['Low'].rolling(period).min()
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    trend = "NEUTRAL"
    if df['Close'].iloc[-1] > upperband.iloc[-1]:
        trend = "BULLISH SUPERTREND"
    elif df['Close'].iloc[-1] < lowerband.iloc[-1]:
        trend = "BEARISH SUPERTREND"
    return trend

# -----------------------------
# Analyze Timeframe
# -----------------------------
def analyze_timeframe(symbol: str, interval: str, period: str) -> dict:
    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        return {"error": "No data"}
    df = clean_df(df)
    df["EMA_5"] = EMAIndicator(df["Close"],5).ema_indicator()
    df["EMA_10"] = EMAIndicator(df["Close"],10).ema_indicator()
    df["EMA_20"] = EMAIndicator(df["Close"],20).ema_indicator()
    df["EMA_50"] = EMAIndicator(df["Close"],50).ema_indicator()
    df["VWAP"] = calculate_vwap(df)
    g_trend = guppy_ema(df)
    st_trend = supertrend(df)
    return {
        "Price": round(float(df["Close"].iloc[-1]),2),
        "VWAP": round(float(df["VWAP"].iloc[-1]),2),
        "EMA_5": round(float(df["EMA_5"].iloc[-1]),2),
        "EMA_10": round(float(df["EMA_10"].iloc[-1]),2),
        "EMA_20": round(float(df["EMA_20"].iloc[-1]),2),
        "EMA_50": round(float(df["EMA_50"].iloc[-1]),2),
        "EMA_5_10_Cross": ema_cross(df,5,10),
        "Guppy_Trend": g_trend,
        "SuperTrend": st_trend
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
    daily = analyze_timeframe(symbol,"1d","6mo")
    weekly = analyze_timeframe(symbol,"1wk","2y")
    monthly = analyze_timeframe(symbol,"1mo","5y")
    intraday_1h = analyze_timeframe(symbol,"60m","60d")
    intraday_4h = analyze_timeframe(symbol,"240m","60d")

    if "error" in daily:
        return {"error": "Invalid symbol or no data"}

    # Daily prediction logic
    score = 0
    if daily["EMA_5"] > daily["EMA_10"]: score+=1
    if daily["EMA_10"] > daily["EMA_20"]: score+=1
    if daily["Price"] > daily["VWAP"]: score+=1

    signal = "HOLD"
    if score==3: signal="BUY"
    elif score==0: signal="SELL"
    probability = round((score/3)*100,2)

    return {
        "symbol": symbol,
        "prediction_based_on": "DAILY",
        "signal": signal,
        "probability_%": probability,
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly,
            "Intraday_1H": intraday_1h,
            "Intraday_4H": intraday_4h
        }
    }
