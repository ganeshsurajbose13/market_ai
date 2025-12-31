from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ================= APP =================
app = FastAPI(title="Market AI Professional Engine")

# ================= UTIL =================
def fetch_df(symbol, interval="1d", period="1y"):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            return None
        return df
    except:
        return None

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def vwap(df):
    return (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

# ================= CORE ANALYSIS =================
def timeframe_analysis(symbol, interval, period):
    df = fetch_df(symbol, interval, period)
    close = df["Close"]

    return {
        "Price": round(close.iloc[-1], 2),
        "VWAP": round(vwap(df).iloc[-1], 2),
        "EMA_5": round(ema(close, 5).iloc[-1], 2),
        "EMA_10": round(ema(close, 10).iloc[-1], 2),
        "EMA_20": round(ema(close, 20).iloc[-1], 2)
    }

# ================= TREND MODEL =================
def trend_ml_model(symbol):
    df = fetch_df(symbol, "1d", "2y")
    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)
    df["RET"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    X = df[["EMA5", "EMA20", "RET"]]
    y = np.where(df["EMA5"] > df["EMA20"], 1, 0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = GradientBoostingClassifier()
    model.fit(Xs[:-1], y[:-1])

    pred = model.predict(Xs[-1].reshape(1, -1))[0]
    return "UPTREND" if pred == 1 else "DOWNTREND"

# ================= FUTURES OI (PROXY) =================
def futures_oi_proxy(symbol):
    df = fetch_df(symbol, "1d", "3mo")
    price_change = df["Close"].pct_change().iloc[-1]
    vol_change = df["Volume"].pct_change().iloc[-1]

    if price_change > 0 and vol_change > 0:
        return "LONG_BUILDUP"
    if price_change < 0 and vol_change > 0:
        return "SHORT_BUILDUP"
    return "NO_BUILDUP"

# ================= OPTIONS OI (RULE BASED) =================
def options_oi_proxy(symbol):
    df = fetch_df(symbol, "1d", "1mo")
    vol = df["Volume"].iloc[-1]
    avg = df["Volume"].mean()

    if vol > avg * 1.5:
        return "HIGH_ACTIVITY"
    elif vol < avg * 0.7:
        return "LOW_ACTIVITY"
    return "NORMAL_ACTIVITY"

# ================= REGRESSION =================
def regression_prediction(symbol):
    df = fetch_df(symbol, "1d", "6mo")
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    model = LinearRegression().fit(X, y)
    pred = model.predict([[len(df)+5]])[0]

    return "UP" if pred > y[-1] else "DOWN"

# ================= LSTM =================
def lstm_prediction(symbol):
    df = fetch_df(symbol, "1d", "1y")
    prices = df["Close"].values[-60:]

    X = []
    y = []
    for i in range(50):
        X.append(prices[i:i+5])
        y.append(prices[i+5])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(5,1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, verbose=0)

    pred = model.predict(prices[-5:].reshape(1,5,1), verbose=0)[0][0]
    return "UP" if pred > prices[-1] else "DOWN"

# ================= BACKTEST =================
def backtest(symbol):
    df = fetch_df(symbol, "1d", "2y")
    df["EMA5"] = ema(df["Close"], 5)
    df["EMA20"] = ema(df["Close"], 20)

    df["signal"] = np.where(df["EMA5"] > df["EMA20"], 1, -1)
    df["ret"] = df["Close"].pct_change()
    df["strategy"] = df["signal"].shift(1) * df["ret"]

    return round(df["strategy"].sum()*100, 2)

# ================= FINAL INFERENCE =================
@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    daily = timeframe_analysis(symbol, "1d", "6mo")
    weekly = timeframe_analysis(symbol, "1wk", "2y")
    monthly = timeframe_analysis(symbol, "1mo", "5y")

    score = 0
    if daily["EMA_5"] > daily["EMA_20"]: score += 1
    if weekly["EMA_5"] > weekly["EMA_20"]: score += 1
    if monthly["EMA_5"] > monthly["EMA_20"]: score += 1

    trend = trend_ml_model(symbol)
    fut_oi = futures_oi_proxy(symbol)
    opt_oi = options_oi_proxy(symbol)

    backtest_ret = backtest(symbol)
    reg = regression_prediction(symbol)
    lstm = lstm_prediction(symbol)

    final_score = score
    if trend == "UPTREND": final_score += 1
    if fut_oi == "LONG_BUILDUP": final_score += 1
    if reg == "UP": final_score += 1
    if lstm == "UP": final_score += 1

    if final_score >= 6:
        final_signal = "STRONG BUY"
    elif final_score >= 4:
        final_signal = "BUY"
    elif final_score == 3:
        final_signal = "HOLD"
    elif final_score >= 1:
        final_signal = "SELL"
    else:
        final_signal = "STRONG SELL"

    return {
        "symbol": symbol,
        "final_signal": final_signal,
        "base_score": score,
        "trend_model": trend,
        "futures_oi": fut_oi,
        "options_oi": opt_oi,
        "ai_prediction": {
            "Regression": reg,
            "LSTM": lstm
        },
        "backtest_return_%": backtest_ret,
        "analysis": {
            "Daily": daily,
            "Weekly": weekly,
            "Monthly": monthly
        }
    }

# ================= CHART DATA =================
@app.get("/chart-data")
def chart_data(symbol: str):
    df = fetch_df(symbol, "1d", "6mo")
    df["EMA5"] = ema(df["Close"], 5)
    df["EMA10"] = ema(df["Close"], 10)
    df["EMA20"] = ema(df["Close"], 20)

    return {
        "date": df.index.strftime("%Y-%m-%d").tolist(),
        "close": df["Close"].tolist(),
        "ema5": df["EMA5"].tolist(),
        "ema10": df["EMA10"].tolist(),
        "ema20": df["EMA20"].tolist()
    }

# ================= STATIC =================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse("/static/index.html")
