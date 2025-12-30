from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import yfinance as yf

app = FastAPI(title="Market AI Engine – Stable v10")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTIL ----------------
def safe_series(x):
    """Force 1D pandas Series"""
    return pd.Series(x.values.reshape(-1))

def analyze_timeframe(symbol, interval, period):
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False
        )

        if df.empty:
            return {"error": "No data from Yahoo"}

        # 🔴 FIX MULTIINDEX (RENDER ISSUE)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        close = safe_series(df["Close"])
        high = safe_series(df["High"])
        low = safe_series(df["Low"])
        volume = safe_series(df["Volume"])

        # EMA manual (NO ta LIBRARY → SAFE)
        ema5 = close.ewm(span=5).mean().iloc[-1]
        ema10 = close.ewm(span=10).mean().iloc[-1]
        ema20 = close.ewm(span=20).mean().iloc[-1]

        vwap = ((high + low + close) / 3 * volume).cumsum() / volume.cumsum()

        return {
            "Price": round(float(close.iloc[-1]), 2),
            "VWAP": round(float(vwap.iloc[-1]), 2),
            "EMA_5": round(float(ema5), 2),
            "EMA_10": round(float(ema10), 2),
            "EMA_20": round(float(ema20), 2),
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------- API ----------------
@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_timeframe(symbol, "1d", "6mo")

    if "error" in daily:
        return {"error": daily["error"]}

    score = 0
    if daily["EMA_5"] > daily["EMA_10"]:
        score += 1
    if daily["EMA_10"] > daily["EMA_20"]:
        score += 1
    if daily["Price"] > daily["VWAP"]:
        score += 1

    signal = "BUY" if score == 3 else "SELL" if score == 0 else "HOLD"

    return {
        "symbol": symbol,
        "signal": signal,
        "probability_%": round((score / 3) * 100, 2),
        "analysis": {
            "Daily": daily
        }
    }

# ---------------- STATIC ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
