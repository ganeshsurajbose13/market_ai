from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator

app = FastAPI(title="Market AI Engine – Stable v9")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- UTILITIES ----------------
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)

    # 🔴 FORCE 1-D SERIES (CRITICAL FIX)
    df["Close"] = pd.Series(df["Close"].values.flatten())
    df["High"] = pd.Series(df["High"].values.flatten())
    df["Low"] = pd.Series(df["Low"].values.flatten())
    df["Volume"] = pd.Series(df["Volume"].values.flatten())

    return df

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

def ema(series: pd.Series, period: int) -> pd.Series:
    series = pd.Series(series.values.flatten())  # 🔴 FORCE 1D AGAIN
    return EMAIndicator(series, period).ema_indicator()

# ---------------- ANALYSIS ----------------
def analyze_timeframe(symbol: str, interval: str, period: str) -> dict:
    try:
        df = yf.download(
            symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return {"error": "No data from Yahoo"}

        df = clean_df(df)

        if len(df) < 60:
            return {"error": "Not enough data"}

        ema5 = ema(df["Close"], 5)
        ema10 = ema(df["Close"], 10)
        ema20 = ema(df["Close"], 20)
        ema50 = ema(df["Close"], 50)

        df["VWAP"] = calculate_vwap(df)

        return {
            "Price": round(float(df["Close"].iloc[-1]), 2),
            "VWAP": round(float(df["VWAP"].iloc[-1]), 2),
            "EMA_5": round(float(ema5.iloc[-1]), 2),
            "EMA_10": round(float(ema10.iloc[-1]), 2),
            "EMA_20": round(float(ema20.iloc[-1]), 2),
            "EMA_50": round(float(ema50.iloc[-1]), 2),
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

# ---------------- STATIC ----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")
