import yfinance as yf
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

MODEL_DIR = "cache/models"
os.makedirs(MODEL_DIR, exist_ok=True)

def lstm_predict(symbol):
    model_path = f"{MODEL_DIR}/{symbol.replace('.','_')}.h5"

    df = yf.download(symbol, period="15y", interval="1d", progress=False)
    if df.empty or len(df) < 300:
        return "UNKNOWN"

    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])

    X, y = np.array(X), np.array(y)

    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=3, batch_size=32, verbose=0)
        model.save(model_path)

    pred = model.predict(X[-1].reshape(1,60,1), verbose=0)

    return "UP" if pred[0][0] > data[-1][0] else "DOWN"

