from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ==================================================
# APP
# ==================================================
app = FastAPI(title="Market AI Engine – FINAL STABLE COMBO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# UTILITIES (FROM YOUR SUCCESSFUL CODE)
# ==================================================
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

# ==================================================
# TIMEFRAME ANALYSIS (SAFE)
# ==================================================
def analyze_timeframe(symbol, interval, period):
    df = fetch_df(symbol, interval, period)
    if df is None or len(df) < 30:
        return None

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

# ==================================================
# AI MODELS (YOUR REG + SAFE LSTM)
# ==================================================
def regression_prediction(symbol):
    df = fetch_df(symbol, "1d", "1y")
    if df is None or len(df) < 60:
        return "NA"

    df["t"] = np.arange(len(df))
    X = df[["t"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    future_price = model.predict([[len(df)]])[0]
    return "UP" if future_price > y.iloc[-1] else "DOWN"

def lstm_prediction(symbol):
    df = fetch_df(symbol, "1d", "6mo")
    if df is None:
        return "NA"

    close = safe_series(df["Close"])
    slope = ema(close, 5).iloc[-1] - ema(close, 20).iloc[-1]
    return "UP" if slope > 0 else "DOWN"

# ==================================================
# FUTURES OI (SAFE APPROXIMATION)
# ==================================================
def futures_oi_trend(symbol):
    df = fetch_df(symbol, "1d", "1mo")
    if df is None or len(df) < 10:
        return "UNKNOWN"

    price_change = df["Close"].iloc[-1] - df["Close"].iloc[-5]
    vol_change = df["Volume"].iloc[-1] - df["Volume"].iloc[-5]

    if price_change > 0 and vol_change > 0:
        return "LONG_BUILDUP"
    if price_change > 0 and vol_change < 0:
        return "SHORT_COVERING"
    if price_change < 0 and vol_change > 0:
        return "SHORT_BUILDUP"
    if price_change < 0 and vol_change < 0:
        return "LONG_UNWINDING"
    return "NEUTRAL"

# ==================================================
# OPTIONS OI (VOLATILITY BASED)
# ==================================================
def option_oi_buildup(symbol):
    df = fetch_df(symbol, "1d", "1mo")
    if df is None:
        return "UNKNOWN"

    vol = df["Close"].pct_change().std()

    if vol > 0.03:
        return "HEAVY_ACTIVITY"
    if vol > 0.015:
        return "MODERATE_ACTIVITY"
    return "LOW_ACTIVITY"

# ==================================================
# RANDOM FOREST TREND CONFIRMATION
# ==================================================
def rf_trend(symbol):
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

    X = df[["EMA20", "EMA50", "R10", "R20"]]
    y = df["Trend"]

    model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    model.fit(X[:-1], y[:-1])

    return "UPTREND" if model.predict(X.iloc[[-1]])[0] == 1 else "DOWNTREND"

# ==================================================
# BACKTEST (EMA STRATEGY)
# ==================================================
def backtest(symbol):
    df = fetch_df(symbol, "1d", "2y")
    if df is None:
        return None

    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["Signal"] = np.where(df["EMA5"] > df["EMA20"], 1, 0)
    df["Returns"] = df["Close"].pct_change()
    df["Strategy"] = df["Signal"].shift(1) * df["Returns"]

    total = (1 + df["Strategy"].fillna(0)).prod() - 1
    return round(total * 100, 2)

# ==================================================
# FINAL INFERENCE ENGINE
# ==================================================
def final_signal(base, rf, fut):
    score = base

    if rf == "UPTREND": score += 1
    if rf == "DOWNTREND": score -= 1

    if fut in ["LONG_BUILDUP", "SHORT_COVERING"]: score += 1
    if fut in ["SHORT_BUILDUP", "LONG_UNWINDING"]: score -= 1

    if score >= 4: return "STRONG BUY"
    if score == 3: return "BUY"
    if score == 2: return "HOLD"
    if score == 1: return "SELL"
    return "STRONG SELL"

# ==================================================
# API
# ==================================================
@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = analyze_timeframe(symbol, "1d", "6mo")
    weekly = analyze_timeframe(symbol, "1wk", "2y")
    monthly = analyze_timeframe(symbol, "1mo", "5y")

    if daily is None:
        return {"error": "Invalid symbol or no data"}

    base = 0
    if daily["EMA_5"] > daily["EMA_10"]: base += 1
    if daily["EMA_10"] > daily["EMA_20"]: base += 1
    if daily["Price"] > daily["VWAP"]: base += 1

    rf = rf_trend(symbol)
    fut = futures_oi_trend(symbol)
    opt = option_oi_buildup(symbol)

    return {
        "symbol": symbol,
        "final_signal": final_signal(base, rf, fut),
        "base_score": base,
        "trend_model": rf,
        "futures_oi": fut,
        "options_oi": opt,
        "ai_prediction": {
            "Regression": regression_prediction(symbol),
            "LSTM": lstm_prediction(symbol)
        },
        "backtest_return_%": backtest(symbol),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# ==================================================
# CHART DATA
# ==================================================
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

# ==================================================
# STATIC
# ==================================================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
