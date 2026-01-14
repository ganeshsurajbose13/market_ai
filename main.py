from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# =========================
# APP INIT
# =========================
app = FastAPI(title="Market AI – Complete Combined Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# DATA FETCH
# =========================
def fetch_df(symbol, interval, period):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except:
        return None

# =========================
# INDICATORS
# =========================
def ema(series, n): return series.ewm(span=n, adjust=False).mean()
def vwap(df): return ((df["High"]+df["Low"]+df["Close"])/3*df["Volume"]).cumsum()/df["Volume"].cumsum()
def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain/loss
    return 100-(100/(1+rs))
def macd(series):
    fast = ema(series,12)
    slow = ema(series,26)
    signal = ema(fast-slow,9)
    return fast-slow, signal
def atr(df, n=14):
    tr = np.maximum(df["High"]-df["Low"], np.maximum(abs(df["High"]-df["Close"].shift()), abs(df["Low"]-df["Close"].shift())))
    return tr.rolling(n).mean()

# =========================
# TIMEFRAME ANALYSIS
# =========================
def analyze_tf(symbol, interval, period):
    df = fetch_df(symbol, interval, period)
    if df is None or len(df)<50: return None
    df["EMA5"]=ema(df["Close"],5)
    df["EMA10"]=ema(df["Close"],10)
    df["EMA20"]=ema(df["Close"],20)
    df["VWAP"]=vwap(df)
    bias = "BUY" if df["EMA5"].iloc[-1]>df["EMA10"].iloc[-1] and df["Close"].iloc[-1]>df["VWAP"].iloc[-1] else "SELL"
    return {
        "Price": round(float(df["Close"].iloc[-1]),2),
        "VWAP": round(float(df["VWAP"].iloc[-1]),2),
        "EMA_5": round(float(df["EMA5"].iloc[-1]),2),
        "EMA_10": round(float(df["EMA10"].iloc[-1]),2),
        "EMA_20": round(float(df["EMA20"].iloc[-1]),2),
        "Bias": bias
    }

# =========================
# PRIMARY ENGINE
# =========================
def ai_model(symbol):
    df = fetch_df(symbol,"1d","10y")
    if df is None or len(df)<300: return "UNKNOWN"
    df["EMA20"]=ema(df["Close"],20)
    df["EMA50"]=ema(df["Close"],50)
    df["RSI"]=rsi(df["Close"])
    macd_line,macd_sig=macd(df["Close"])
    df["MACD"]=macd_line
    df["MACD_SIG"]=macd_sig
    df["Target"]=np.where(df["Close"].shift(-10)>df["Close"],1,0)
    df.dropna(inplace=True)
    X = df[["EMA20","EMA50","RSI","MACD","MACD_SIG"]]
    y = df["Target"]
    model = RandomForestClassifier(n_estimators=300,max_depth=6,random_state=42)
    model.fit(X[:-1],y[:-1])
    return "UP" if model.predict(X.iloc[[-1]])[0]==1 else "DOWN"

def regression_trend(symbol):
    weekly = fetch_df(symbol,"1wk","5y")
    monthly = fetch_df(symbol,"1mo","10y")
    if weekly is None or monthly is None: return "UNKNOWN", None, None
    def trend(df):
        X=np.arange(len(df)).reshape(-1,1)
        y=df["Close"].values
        model=LinearRegression()
        model.fit(X[:-1],y[:-1])
        return "UP" if model.predict([[len(df)]])[0]>y[-1] else "DOWN"
    wk = trend(weekly)
    mo = trend(monthly)
    if wk==mo: return wk, wk, mo
    return "SIDEWAYS", wk, mo

def futures_oi(symbol):
    df = fetch_df(symbol,"1d","1mo")
    if df is None: return "UNKNOWN"
    price=df["Close"].iloc[-1]-df["Close"].iloc[-5]
    vol=df["Volume"].iloc[-1]-df["Volume"].iloc[-5]
    if price>0 and vol>0: return "LONG_BUILDUP"
    if price<0 and vol>0: return "SHORT_BUILDUP"
    return "NEUTRAL"

def options_pcr_oi(symbol):
    try:
        url=f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol.replace('.NS','')}"
        headers={"User-Agent":"Mozilla/5.0"}
        s=requests.Session()
        s.get("https://www.nseindia.com", headers=headers)
        r=s.get(url, headers=headers)
        data = r.json().get("records",{}).get("data",[])
    except: return {"signal":"DATA_UNAVAILABLE"}
    ce=pe=0
    for row in data:
        if "CE" in row: ce+=row["CE"]["openInterest"]
        if "PE" in row: pe+=row["PE"]["openInterest"]
    if ce==0: return {"signal":"INVALID"}
    pcr = round(pe/ce,2)
    if pcr>1.2: return {"PCR":pcr,"signal":"STRONG_BULLISH"}
    if pcr>1.0: return {"PCR":pcr,"signal":"BULLISH"}
    if pcr<0.8: return {"PCR":pcr,"signal":"BEARISH"}
    return {"PCR":pcr,"signal":"NEUTRAL"}

def primary_engine(symbol):
    daily = analyze_tf(symbol,"1d","6mo")
    h1 = analyze_tf(symbol,"1h","1mo")
    h4 = analyze_tf(symbol,"4h","3mo")
    if daily is None: return None
    df=fetch_df(symbol,"1d","1y")
    df["ATR"]=atr(df)
    score=0
    reasons=[]
    if daily["Bias"]=="BUY": score+=3; reasons.append("Daily EMA+VWAP bullish")
    if h1 and h1["Bias"]=="BUY": score+=1
    if h4 and h4["Bias"]=="BUY": score+=1
    if futures_oi(symbol)=="LONG_BUILDUP": score+=2; reasons.append("Futures long buildup")
    ai=ai_model(symbol)
    if ai=="UP": score+=1; reasons.append("AI trend positive")
    reg, wk, mo = regression_trend(symbol)
    if reg=="UP": score+=2; reasons.append("Weekly & Monthly regression up")
    options=options_pcr_oi(symbol)
    if options.get("signal")=="STRONG_BULLISH": score+=3
    elif options.get("signal")=="BULLISH": score+=2
    prob = min(92,50+score*5)
    price=daily["Price"]
    atr_val=df["ATR"].iloc[-1]
    return {
        "symbol":symbol,
        "final_signal":"STRONG BUY" if prob>=80 else "BUY" if prob>=65 else "HOLD",
        "probability_%":prob,
        "ai_model":ai,
        "regression_trend":reg,
        "weekly_trend":wk,
        "monthly_trend":mo,
        "options_chain":options,
        "stop_loss":round(price-atr_val*1.2,2),
        "target":round(price+atr_val*2.5,2),
        "analysis":{"Daily":daily,"1H":h1,"4H":h4},
        "reason":reasons
    }

# =========================
# COMBO ENGINE
# =========================
def forecast_future(df, periods=5):
    if df is None or len(df)<20: return [],[]
    X=np.arange(len(df)).reshape(-1,1)
    y=df['Close'].values
    model=LinearRegression()
    model.fit(X,y)
    future_X=np.arange(len(df),len(df)+periods).reshape(-1,1)
    forecast=model.predict(future_X).round(2)
    trend=["NEUTRAL" if i==0 else "UP" if forecast[i]>forecast[i-1] else "DOWN" for i in range(len(forecast))]
    return forecast.tolist(), trend

def ai_engine(symbol):
    df=fetch_df(symbol,"1d","10y")
    if df is None or len(df)<300: return 50.0,"UNKNOWN",[],[]
    df["EMA20"]=ema(df["Close"],20)
    df["EMA50"]=ema(df["Close"],50)
    df["RSI"]=rsi(df["Close"])
    df["RET"]=df["Close"].pct_change()
    df["Target"]=np.where(df["Close"].shift(-5)>df["Close"],1,0)
    df.dropna(inplace=True)
    X=df[["EMA20","EMA50","RSI","RET"]]
    y=df["Target"]
    model=RandomForestClassifier(n_estimators=300,max_depth=6,random_state=42)
    model.fit(X[:-1],y[:-1])
    prob=model.predict_proba(X.iloc[[-1]])[0][1]*100
    trend="UP" if prob>=50 else "DOWN"
    fc,fc_trend=forecast_future(df,5)
    return round(prob,2),trend,fc,fc_trend

def regression_trend_forecast(symbol):
    daily_df=fetch_df(symbol,"1d","1y")
    weekly_df=fetch_df(symbol,"1wk","5y")
    monthly_df=fetch_df(symbol,"1mo","10y")
    def calc(df):
        if df is None or len(df)<20: return "UNKNOWN",[],[]
        X=np.arange(len(df)).reshape(-1,1)
        y=df["Close"].values
        model=LinearRegression()
        model.fit(X,y)
        slope=model.coef_[0]
        trend="UP" if slope>0 else "DOWN"
        fc,fc_trend=forecast_future(df,5)
        return trend,fc,fc_trend
    daily_tr,daily_fc,daily_fc_trend=calc(daily_df)
    weekly_tr,weekly_fc,weekly_fc_trend=calc(weekly_df)
    monthly_tr,monthly_fc,monthly_fc_trend=calc(monthly_df)
    overall_tr=daily_tr if daily_tr==weekly_tr==monthly_tr else "SIDEWAYS"
    return overall_tr,daily_tr,weekly_tr,monthly_tr,daily_fc,weekly_fc,monthly_fc,daily_fc_trend,weekly_fc_trend,monthly_fc_trend

def final_combo_engine(symbol):
    daily=analyze_tf(symbol,"1d","6mo")
    h1=analyze_tf(symbol,"1h","1mo")
    h4=analyze_tf(symbol,"4h","3mo")
    if daily is None: return {"error":"DATA_UNAVAILABLE"}
    df=fetch_df(symbol,"1d","1y")
    df["ATR"]=atr(df)
    score=50
    reasons=[]
    if daily["Bias"]=="BUY": score+=10; reasons.append("Daily bullish structure")
    opt=options_pcr_oi(symbol)
    if opt.get("signal")=="BULLISH": score+=5
    elif opt.get("signal")=="BEARISH": score-=5
    ai_prob,ai_trend,ai_fc,ai_fc_trend=ai_engine(symbol)
    if ai_prob>=60: score+=10; reasons.append("AI bullish probability")
    overall_tr,daily_tr,weekly_tr,monthly_tr,daily_fc,weekly_fc,monthly_fc,daily_fc_trend,weekly_fc_trend,monthly_fc_trend=regression_trend_forecast(symbol)
    if overall_tr=="UP": score+=5
    final_signal="STRONG BUY" if score>=75 else "BUY" if score>=60 else "SELL" if score<=40 else "HOLD"
    band="HIGH" if score>=70 else "MEDIUM" if score>=55 else "LOW"
    price=daily["Price"]
    atr_val=df["ATR"].iloc[-1]
    return {
        "symbol":symbol,
        "final_signal":final_signal,
        "confidence_score_%":score,
        "confidence_band":band,
        "ai_probability_%":ai_prob,
        "ai_trend":ai_trend,
        "ai_model":"RandomForest",
        "regression_trend":overall_tr,
        "daily_trend":daily_tr,
        "weekly_trend":weekly_tr,
        "monthly_trend":monthly_tr,
        "forecast":{
            "daily_next_5_days":daily_fc,
            "daily_trend_next_5_days":daily_fc_trend,
            "weekly_next_5_weeks":weekly_fc,
            "weekly_trend_next_5_weeks":weekly_fc_trend,
            "monthly_next_5_months":monthly_fc,
            "monthly_trend_next_5_months":monthly_fc_trend,
            "ai_next_5_days":ai_fc,
            "ai_trend_next_5_days":ai_fc_trend
        },
        "analysis":{"Daily":daily,"1H":h1,"4H":h4},
        "options_chain":opt,
        "risk":{"stop_loss":round(price-atr_val*1.2,2),
                "target":round(price+atr_val*2.5,2),
                "atr":round(float(atr_val),2)},
        "reason":reasons
    }

# =========================
# API
# =========================
@app.get("/health")
def health(): return {"status":"OK"}

@app.get("/analyze")
def analyze(symbol: str = Query(...)):
    return {
        "primary_engine": primary_engine(symbol),
        "combo_engine": final_combo_engine(symbol)
    }

# =========================
# STATIC FILES
# =========================
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html", headers={"Cache-Control":"no-cache"})
