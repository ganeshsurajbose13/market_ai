from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# APP
# =========================
app = FastAPI(title="Market AI Engine – Stable Professional")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# UTILITIES
# =========================
def fetch_df(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except:
        return None

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

# =========================
# TIMEFRAME ANALYSIS
# =========================
def analyze_tf(symbol, interval, period):
    df = fetch_df(symbol, interval, period)
    if df is None or len(df) < 50:
        return None

    df["EMA_5"] = ema(df["Close"], 5)
    df["EMA_10"] = ema(df["Close"], 10)
    df["EMA_20"] = ema(df["Close"], 20)
    df["VWAP"] = vwap(df)

    return {
        "Price": round(float(df["Close"].iloc[-1]), 2),
        "VWAP": round(float(df["VWAP"].iloc[-1]), 2),
        "EMA_5": round(float(df["EMA_5"].iloc[-1]), 2),
        "EMA_10": round(float(df["EMA_10"].iloc[-1]), 2),
        "EMA_20": round(float(df["EMA_20"].iloc[-1]), 2),
    }

# =========================
# TREND ML MODEL (SAFE)
# =========================
def trend_ml(symbol):
    df = fetch_df(symbol, "1d", "2y")
    if df is None or len(df) < 120:
        return "UNKNOWN"

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["R10"] = df["Close"].pct_change(10)
    df["R20"] = df["Close"].pct_change(20)

    df["Future"] = df["EMA20"].shift(-10) - df["EMA20"]
    df["Trend"] = np.where(df["Future"] > 0, 1, -1)
    df.dropna(inplace=True)

    X = df[["EMA20","EMA50","R10","R20"]]
    y = df["Trend"]

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X[:-1], y[:-1])

    pred = model.predict(X.iloc[[-1]])[0]
    return "UPTREND" if pred == 1 else "DOWNTREND"

# =========================
# REGRESSION AI
# =========================
def regression_prediction(symbol):
    df = fetch_df(symbol, "1d", "1y")
    if df is None or len(df) < 60:
        return "UNKNOWN"

    X = np.arange(len(df)).reshape(-1,1)
    y = df["Close"].values

    model = LinearRegression()
    model.fit(X[:-1], y[:-1])

    next_price = model.predict([[len(df)]])[0]
    return "UP" if next_price > y[-1] else "DOWN"

# =========================
# SAFE LSTM PROXY (NO TENSORFLOW)
# =========================
def lstm_proxy(symbol):
    df = fetch_df(symbol, "1d", "6mo")
    if df is None:
        return "UNKNOWN"

    slope = ema(df["Close"], 5).iloc[-1] - ema(df["Close"], 20).iloc[-1]
    return "UP" if slope > 0 else "DOWN"

# =========================
# FUTURES & OPTIONS OI (PROXY)
# =========================
def futures_oi(symbol):
    df = fetch_df(symbol, "1d", "1mo")
    if df is None:
        return "UNKNOWN"

    price = df["Close"].iloc[-1] - df["Close"].iloc[-5]
    vol = df["Volume"].iloc[-1] - df["Volume"].iloc[-5]

    if price > 0 and vol > 0:
        return "LONG_BUILDUP"
    if price < 0 and vol > 0:
        return "SHORT_BUILDUP"
    return "NEUTRAL"

def options_oi(symbol):
    df = fetch_df(symbol, "1d", "1mo")
    if df is None:
        return "UNKNOWN"

    vol = df["Volume"].iloc[-1]
    avg = df["Volume"].mean()

    if vol > avg * 1.5:
        return "HIGH_ACTIVITY"
    if vol < avg * 0.7:
        return "LOW_ACTIVITY"
    return "NORMAL"

# =========================
# BACKTEST (EMA STRATEGY)
# =========================
def backtest(symbol):
    df = fetch_df(symbol, "1d", "2y")
    if df is None:
        return None

    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)

    df["Signal"] = np.where(df["EMA5"] > df["EMA20"], 1, -1)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Returns"]

    return round(((1 + df["Strategy"].fillna(0)).prod() - 1) * 100, 2)

# =========================
# API ENDPOINTS
# =========================
@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_tf(symbol, "1d", "6mo")
    weekly = analyze_tf(symbol, "1wk", "2y")
    monthly = analyze_tf(symbol, "1mo", "5y")

    if daily is None:
        return {"error": "Invalid symbol or no data"}

    base_score = 0
    if daily["EMA_5"] > daily["EMA_10"]: base_score += 1
    if daily["EMA_10"] > daily["EMA_20"]: base_score += 1
    if daily["Price"] > daily["VWAP"]: base_score += 1

    trend = trend_ml(symbol)
    fut_oi = futures_oi(symbol)
    opt_oi = options_oi(symbol)

    score = base_score
    if trend == "UPTREND": score += 1
    if fut_oi == "LONG_BUILDUP": score += 1

    if score >= 5:
        final_signal = "STRONG BUY"
    elif score == 4:
        final_signal = "BUY"
    elif score == 3:
        final_signal = "HOLD"
    elif score == 2:
        final_signal = "SELL"
    else:
        final_signal = "STRONG SELL"

    return {
        "symbol": symbol,
        "final_signal": final_signal,
        "base_score": base_score,
        "trend_model": trend,
        "futures_oi": fut_oi,
        "options_oi": opt_oi,
        "ai_prediction": {
            "Regression": regression_prediction(symbol),
            "LSTM": lstm_proxy(symbol)
        },
        "backtest_return_%": backtest(symbol),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

@app.get("/chart-data")
def chart_data(symbol: str):
    df = fetch_df(symbol, "1d", "6mo")
    if df is None:
        return {"error": "No data"}

    df["EMA5"] = ema(df["Close"], 5)
    df["EMA10"] = ema(df["Close"], 10)
    df["EMA20"] = ema(df["Close"], 20)

    return {
        "date": df.index.strftime("%Y-%m-%d").tolist(),
        "close": df["Close"].tolist(),
        "ema5": df["EMA5"].round(2).tolist(),
        "ema10": df["EMA10"].round(2).tolist(),
        "ema20": df["EMA20"].round(2).tolist()
    }

# =========================
# STATIC
# =========================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
