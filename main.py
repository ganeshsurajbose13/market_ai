from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Market AI Engine – Stable v13")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTIL ----------------
def safe_series(x):
    return pd.Series(x.values.reshape(-1))

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def fetch_df(symbol, interval, period):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# ---------------- ANALYSIS ----------------
def analyze_timeframe(symbol, interval, period):
    df = fetch_df(symbol, interval, period)
    if df is None:
        return {"error": "No data"}

    close = safe_series(df["Close"])
    high = safe_series(df["High"])
    low = safe_series(df["Low"])
    volume = safe_series(df["Volume"])

    vwap = ((high + low + close) / 3 * volume).cumsum() / volume.cumsum()

    return {
        "Price": round(float(close.iloc[-1]), 2),
        "VWAP": round(float(vwap.iloc[-1]), 2),
        "EMA_5": round(float(ema(close, 5).iloc[-1]), 2),
        "EMA_10": round(float(ema(close, 10).iloc[-1]), 2),
        "EMA_20": round(float(ema(close, 20).iloc[-1]), 2),
    }

# ---------------- REGRESSION AI ----------------
def regression_prediction(symbol):
    df = fetch_df(symbol, "1d", "1y")
    if df is None or len(df) < 50:
        return "NA"

    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    future_price = model.predict([[len(df)]])[0]
    last_price = y.iloc[-1]

    return "UP" if future_price > last_price else "DOWN"

# ---------------- LSTM (SAFE PLACEHOLDER) ----------------
def lstm_prediction(symbol):
    # Render-safe placeholder (no TensorFlow crash)
    # We simulate trend using EMA slope
    df = fetch_df(symbol, "1d", "6mo")
    if df is None:
        return "NA"

    close = safe_series(df["Close"])
    slope = ema(close, 5).iloc[-1] - ema(close, 20).iloc[-1]

    return "UP" if slope > 0 else "DOWN"

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
        return {"error": "Invalid symbol or no data"}

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
        "probability_%": round(score / 3 * 100, 2),
        "ai_prediction": {
            "Regression": regression_prediction(symbol),
            "LSTM": lstm_prediction(symbol)
        },
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# ---------------- CHART DATA ----------------
@app.get("/chart-data")
def chart_data(symbol: str = Query(...)):
    df = fetch_df(symbol, "1d", "6mo")
    if df is None:
        return {"error": "No data"}

    close = safe_series(df["Close"])
    df["EMA_5"] = ema(close, 5)
    df["EMA_10"] = ema(close, 10)
    df["EMA_20"] = ema(close, 20)

    return {
        "date": df.index.strftime("%Y-%m-%d").tolist(),
        "open": df["Open"].tolist(),
        "high": df["High"].tolist(),
        "low": df["Low"].tolist(),
        "close": df["Close"].tolist(),
        "ema5": df["EMA_5"].round(2).tolist(),
        "ema10": df["EMA_10"].round(2).tolist(),
        "ema20": df["EMA_20"].round(2).tolist(),
    }

# ---------------- STATIC ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
