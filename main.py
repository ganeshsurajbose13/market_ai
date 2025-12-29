from fastapi import FastAPI, Query
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
import requests
from io import StringIO
import joblib
import os

app = FastAPI(title="Market AI Engine – Stable v7 with Intraday Analysis")

# -----------------------------
# Fetch NSE + BSE stock list automatically
# -----------------------------
def fetch_nse_stocks():
    url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(StringIO(response.text))
        df = df[["SC_NAME", "SYMBOL"]]
        df["SYMBOL"] = df["SYMBOL"].astype(str) + ".NS"
        return [{"symbol": row["SYMBOL"], "name": row["SC_NAME"]} for _, row in df.iterrows()]
    except Exception as e:
        print("Error fetching NSE stocks:", e)
        return []

def fetch_bse_stocks():
    url = "https://www.bseindia.com/download/BhavCopy/Equity/EQ_ISINCODE.csv"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(StringIO(response.text))
        df = df[["SC_NAME", "SC_CODE"]]
        df["SYMBOL"] = df["SC_CODE"].astype(str) + ".BO"
        return [{"symbol": row["SYMBOL"], "name": row["SC_NAME"]} for _, row in df.iterrows()]
    except Exception as e:
        print("Error fetching BSE stocks:", e)
        return []

stock_symbols = fetch_nse_stocks() + fetch_bse_stocks()

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
# ML Placeholder
# -----------------------------
ML_MODEL_PATH = "ml_model.pkl"
ml_model = None
if os.path.exists(ML_MODEL_PATH):
    try:
        ml_model = joblib.load(ML_MODEL_PATH)
        print("ML model loaded successfully")
    except Exception as e:
        print("Error loading ML model:", e)
else:
    print("No ML model found yet")

def ml_predict(features: np.ndarray):
    if ml_model is None:
        return {"prob_up": None, "prob_down": None, "note": "ML model not loaded"}
    probs = ml_model.predict_proba(features.reshape(1,-1))[0]
    return {"prob_up": round(float(probs[1]*100),2), "prob_down": round(float(probs[0]*100),2)}

# -----------------------------
# Analyze Timeframe (supports intraday)
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
    daily = analyze_timeframe(symbol, "1d", "6mo")
    weekly = analyze_timeframe(symbol, "1wk", "2y")
    monthly = analyze_timeframe(symbol, "1mo", "5y")
    intraday_1h = analyze_timeframe(symbol, "60m", "60d")   # 1H intraday, last 60 days
    intraday_4h = analyze_timeframe(symbol, "240m", "60d")  # 4H intraday, last 60 days

    if "error" in daily:
        return {"error": "Invalid symbol or no data"}

    # SIMPLE DAILY PREDICTION LOGIC (SAFE)
    score = 0
    if daily["EMA_5"] > daily["EMA_10"]:
        score += 1
    if daily["EMA_10"] > daily["EMA_20"]:
        score += 1
    if daily["Price"] > daily["VWAP"]:
        score += 1
    signal = "HOLD"
    if score==3:
        signal="BUY"
    elif score==0:
        signal="SELL"
    probability = round((score/3)*100,2)

    # Placeholder ML feature vector
    features = np.array([daily["EMA_5"], daily["EMA_10"], daily["EMA_20"]])
    ml_prediction = ml_predict(features)

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

@app.get("/search")
def search(q: str = Query(..., description="Enter stock symbol or name")):
    q_lower = q.lower()
    results = [s for s in stock_symbols if q_lower in s["symbol"].lower() or q_lower in s["name"].lower()]
    return {"query": q, "results": results[:30]}
