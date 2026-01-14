import yfinance as yf
import matplotlib.pyplot as plt
from main import final_engine

symbol = "RELIANCE.NS"
df = yf.download(symbol, period="3y")

signals = []
for i in range(200, len(df)):
    s = final_engine(symbol)
    signals.append(s["final_signal"] if s else "HOLD")

plt.plot(df["Close"], label="Price")
plt.title(symbol + " Backtest")
plt.show()

