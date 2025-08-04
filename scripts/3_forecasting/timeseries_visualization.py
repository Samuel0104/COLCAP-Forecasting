import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import pandas as pd

data = pd.read_csv("../../assets/dataset.csv")

ohlc = data.loc[:, ["date", "open", "high", "low", "close"]]
ohlc["date"] = pd.to_datetime(ohlc["date"])
ohlc["date"] = ohlc["date"].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

fig, ax = plt.subplots(figsize=(10.5, 4.5))
candlestick_ohlc(ax, ohlc.values, width=0.8,
                 colorup="green", colordown="red", alpha=0.6)
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor")
date_format = mpl_dates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()

trm = data.loc[:, ["date", "dollar"]]
trm["date"] = pd.to_datetime(trm["date"])
trm["date"] = trm["date"].apply(mpl_dates.date2num)
trm = trm.astype(float)

fig, ax = plt.subplots(figsize=(10.5, 4.5))
ax.plot(trm["date"], trm["dollar"], linewidth=1)
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor")
date_format = mpl_dates.DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
fig.tight_layout()
plt.show()