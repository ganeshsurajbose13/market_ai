from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable v3 FINAL")

# =============================
# SAFETY CONSTANTS
# =============================
MIN_EMA = 60
MIN_CROSS = 3
MIN_GMMA = 70
MIN_INTRADAY = 30

# =============================
# UTILITY FUNCTIONS
# =============================

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df[list(required)]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)
    return df


def calculate_vwap(df: pd.DataFrame):
    if len(df) < 5:
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def ema_safe(series, period):
    if len(series) < period:
        return None
    return EMAIndicator(series, period).ema_indicator()


def ema_cross(df, s, l):
    if len(df) < MIN_CROSS:
        return "INSUFFICIENT DATA"

    es = ema_safe(df["Close"], s)
    el = ema_safe(df["Close"], l)

    if es is None or el is None:
        return "INSUFFICIENT DATA"

    if es.iloc[-2] < el.iloc[-2] and es.iloc[-1] > el.iloc[-1]:
        return "BULLISH CROSS"
    if es.iloc[-2] > el.iloc[-2] and es.iloc[-1] < el.iloc[-1]:
        return "BEARISH CROSS"
    return "NO CROSS"


def guppy_trend(df):
    if len(df) < MIN_GMMA:
        return "INSUFFICIENT DATA"

    short = [3,5,8,10,12,15]
    long = [30,35,40,45,50,60]

    s_vals = []
    l_vals = []

    for p in short:
        e = ema_safe(df["Close"], p)
        if e is None:
            return "INSUFFICIENT DATA"
        s_vals.append(e.iloc[-1])

    for p in long:
        e = ema_safe(df["Close"], p)
        if e is None:
            return "INSUFFICIENT DATA"
        l_vals.append(e.iloc[-1])

    return "BULLISH GMMA" if np.mean(s_vals) > np.mean(l_vals) else "BEARISH GMMA"


# =============================
# CORE ANALYSIS ENGINE
# =============================

def analyze_timeframe(symbol, interval, period, intraday=False):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False
    )

    df = clean_df(df)

    if df.empty:
        return {"error": "No data"}

    if intraday and len(df) < MIN_INTRADAY:
        return {"error": "Insufficient intraday data"}

    if len(df) < MIN_EMA:
        return {"error": "Insufficient candles"}

    ema5 = ema_safe(df["Close"], 5)
    ema10 = ema_safe(df["Close"], 10)
    ema20 = ema_safe(df["Close"], 20)
    ema50 = ema_safe(df["Close"], 50)
    vwap = calculate_vwap(df)

    if None in (ema5, ema10, ema20, ema50, vwap):
        return {"error": "Indicator calculation failed"}

    return {
        "Price": round(float(df["Close"].iloc[-1]), 2),
        "VWAP": round(float(vwap.iloc[-1]), 2),
        "EMA_5": round(float(ema5.iloc[-1]), 2),
        "EMA_10": round(float(ema10.iloc[-1]), 2),
        "EMA_20": round(float(ema20.iloc[-1]), 2),
        "EMA_50": round(float(ema50.iloc[-1]), 2),
        "EMA_5_10_Cross": ema_cross(df, 5, 10),
        "Guppy_Trend": guppy_trend(df)
    }


# =============================
# API ENDPOINTS
# =============================

@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_timeframe(symbol, "1d", "6mo")
    weekly = analyze_timeframe(symbol, "1wk", "2y")
    monthly = analyze_timeframe(symbol, "1mo", "5y")
    h1 = analyze_timeframe(symbol, "60m", "7d", intraday=True)
    h4 = analyze_timeframe(symbol, "240m", "60d", intraday=True)

    if "error" in daily:
        return {"error": "Invalid symbol or insufficient data"}

    score = 0
    if daily["EMA_5"] > daily["EMA_10"]: score += 1
    if daily["EMA_10"] > daily["EMA_20"]: score += 1
    if daily["Price"] > daily["VWAP"]: score += 1

    signal = "BUY" if score == 3 else "SELL" if score == 0 else "HOLD"

    return {
        "symbol": symbol.upper(),
        "signal": signal,
        "confidence_%": round(score / 3 * 100, 2),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly,
            "Intraday_1H": h1,
            "Intraday_4H": h4
        }
    }


# =============================
# FRONTEND
# =============================
app.mount("/", StaticFiles(directory="static", html=True), name="static")
