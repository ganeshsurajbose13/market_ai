from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import yfinance as yf

app = FastAPI(title="Market AI Engine – Stable v12")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTILITIES ----------------
def safe_series(x):
    return pd.Series(x.values.reshape(-1))

def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def analyze_timeframe(symbol, interval, period):
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False
        )

        if df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        close = safe_series(df["Close"])
        high = safe_series(df["High"])
        low = safe_series(df["Low"])
        volume = safe_series(df["Volume"])

        ema5 = compute_ema(close, 5)
        ema10 = compute_ema(close, 10)
        ema20 = compute_ema(close, 20)

        vwap = ((high + low + close) / 3 * volume).cumsum() / volume.cumsum()

        return {
            "Price": round(float(close.iloc[-1]), 2),
            "VWAP": round(float(vwap.iloc[-1]), 2),
            "EMA_5": round(float(ema5.iloc[-1]), 2),
            "EMA_10": round(float(ema10.iloc[-1]), 2),
            "EMA_20": round(float(ema20.iloc[-1]), 2),
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
    weekly = analyze_timeframe(symbol, "1wk", "2y")
    monthly = analyze_timeframe(symbol, "1mo", "5y")

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
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# ---------------- CHART DATA API ----------------
@app.get("/chart-data")
def chart_data(symbol: str = Query(...)):
    try:
        df = yf.download(
            symbol,
            interval="1d",
            period="6mo",
            progress=False
        )

        if df.empty:
            return {"error": "No data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close"]].dropna()

        close = safe_series(df["Close"])
        df["EMA_5"] = compute_ema(close, 5)
        df["EMA_10"] = compute_ema(close, 10)
        df["EMA_20"] = compute_ema(close, 20)

        return {
            "date": df.index.strftime("%Y-%m-%d").tolist(),
            "open": df["Open"].round(2).tolist(),
            "high": df["High"].round(2).tolist(),
            "low": df["Low"].round(2).tolist(),
            "close": df["Close"].round(2).tolist(),
            "ema5": df["EMA_5"].round(2).tolist(),
            "ema10": df["EMA_10"].round(2).tolist(),
            "ema20": df["EMA_20"].round(2).tolist(),
        }

    except Exception as e:
        return {"error": str(e)}

# ---------------- STATIC ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
