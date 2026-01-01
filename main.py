from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ================= APP =================
app = FastAPI(title="Market AI Professional Engine – Stable")

# ================= UTIL =================
def fetch_df(symbol, interval="1d", period="1y"):
    df = yf.download(symbol, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open","High","Low","Close","Volume"]].dropna()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

# ================= TIMEFRAME ANALYSIS =================
def timeframe_analysis(symbol, interval, period):
    df = fetch_df(symbol, interval, period)
    if df is None or len(df) < 50:
        return None

    return {
        "Price": round(df["Close"].iloc[-1], 2),
        "VWAP": round(vwap(df).iloc[-1], 2),
        "EMA_5": round(ema(df["Close"], 5).iloc[-1], 2),
        "EMA_10": round(ema(df["Close"], 10).iloc[-1], 2),
        "EMA_20": round(ema(df["Close"], 20).iloc[-1], 2)
    }

# ================= ML TREND MODEL =================
def trend_ml_model(symbol):
    df = fetch_df(symbol, "1d", "2y")
    if df is None or len(df) < 120:
        return "UNKNOWN"

    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["RET"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    X = df[["EMA5","EMA20","RET"]]
    y = np.where(df["EMA5"] > df["EMA20"], 1, 0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=150, random_state=42)
    model.fit(Xs[:-1], y[:-1])

    pred = model.predict(Xs.iloc[[-1]])[0]
    return "UPTREND" if pred == 1 else "DOWNTREND"

# ================= FUTURES OI PROXY =================
def futures_oi_proxy(symbol):
    df = fetch_df(symbol, "1d", "1mo")
    if df is None or len(df) < 10:
        return "UNKNOWN"

    p = df["Close"].iloc[-1] - df["Close"].iloc[-5]
    v = df["Volume"].iloc[-1] - df["Volume"].iloc[-5]

    if p > 0 and v > 0:
        return "LONG_BUILDUP"
    if p < 0 and v > 0:
        return "SHORT_BUILDUP"
    if p > 0 and v < 0:
        return "SHORT_COVERING"
    return "NEUTRAL"

# ================= OPTIONS OI PROXY =================
def options_oi_proxy(symbol):
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

# ================= REGRESSION =================
def regression_prediction(symbol):
    df = fetch_df(symbol, "1d", "6mo")
    if df is None or len(df) < 50:
        return "UNKNOWN"

    X = np.arange(len(df)).reshape(-1,1)
    y = df["Close"].values

    model = LinearRegression()
    model.fit(X[:-1], y[:-1])

    pred = model.predict([[len(df)]])[0]
    return "UP" if pred > y[-1] else "DOWN"

# ================= BACKTEST =================
def backtest(symbol):
    df = fetch_df(symbol, "1d", "2y")
    if df is None:
        return None

    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["Signal"] = np.where(df["EMA5"] > df["EMA20"], 1, -1)
    df["Ret"] = df["Close"].pct_change()
    df["Strat"] = df["Signal"].shift(1) * df["Ret"]

    return round((1 + df["Strat"].fillna(0)).prod() * 100 - 100, 2)

# ================= FINAL API =================
@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = timeframe_analysis(symbol, "1d", "6mo")
    weekly = timeframe_analysis(symbol, "1wk", "2y")
    monthly = timeframe_analysis(symbol, "1mo", "5y")

    if daily is None:
        return {"error": "Invalid symbol"}

    base_score = 0
    if daily["EMA_5"] > daily["EMA_20"]: base_score += 1
    if weekly["EMA_5"] > weekly["EMA_20"]: base_score += 1
    if monthly["EMA_5"] > monthly["EMA_20"]: base_score += 1

    trend = trend_ml_model(symbol)
    fut = futures_oi_proxy(symbol)
    opt = options_oi_proxy(symbol)
    reg = regression_prediction(symbol)

    score = base_score
    if trend == "UPTREND": score += 1
    if fut in ["LONG_BUILDUP","SHORT_COVERING"]: score += 1
    if reg == "UP": score += 1

    if score >= 6:
        final_signal = "STRONG BUY"
    elif score >= 4:
        final_signal = "BUY"
    elif score == 3:
        final_signal = "HOLD"
    elif score >= 1:
        final_signal = "SELL"
    else:
        final_signal = "STRONG SELL"

    return {
        "symbol": symbol,
        "final_signal": final_signal,
        "score": score,
        "trend_model": trend,
        "futures_oi": fut,
        "options_oi": opt,
        "regression": reg,
        "backtest_%": backtest(symbol),
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# ================= CHART =================
@app.get("/chart-data")
def chart_data(symbol: str):
    df = fetch_df(symbol, "1d", "6mo")
    if df is None:
        return {"error":"No data"}

    return {
        "date": df.index.strftime("%Y-%m-%d").tolist(),
        "close": df["Close"].tolist(),
        "ema5": ema(df["Close"],5).tolist(),
        "ema20": ema(df["Close"],20).tolist()
    }

# ================= STATIC =================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
