# Time Series Analysis of Brent Crude Oil Spot Prices
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.chdir(os.path.dirname(os.path.realpath("time_series.py")))

start = dt.datetime(1997, 1, 1)
end = dt.datetime(2006, 12, 31)

# Step 1: Preparing the data, log of monthly average of daily data
# Federal Reserve Economic Data (FRED) for Brent Spot Price (DCOILBRENTEU)
brent = web.DataReader("DCOILBRENTEU", "fred", start, end)
brent_m = brent.resample("M").mean()

# Plotting brent_m
plt.figure(figsize=(10, 4))
plt.plot(brent_m, label="Brent spot price (monthly avg)")
plt.title("Brent crude oil spot price (monthly) from 1997 to 2006")
plt.xlabel("Date")
plt.ylabel("USD per barrel")
plt.legend()
plt.tight_layout()
plt.savefig("brent_monthly.png", dpi=300, bbox_inches="tight")
plt.show()

# Log transform + split into train/validation/test
brent_log = np.log(brent_m)
train_df = brent_log.iloc[brent_log.index <= "2003-01-01", 0]
validation_df = brent_log.iloc[(brent_log.index > "2003-01-01") & (brent_log.index <= "2005-12-31"), 0]
test_df = brent_log.iloc[brent_log.index > "2005-12-31", 0]

# Step 2: Decomposing the time series, I part
train_df_d1 = train_df.diff().dropna()

# Visual Test
plt.figure(figsize=(10, 4))
plt.plot(train_df_d1, label="Brent monthly log returns")
plt.title("Brent monthly log returns first order difference")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.legend()
plt.tight_layout()
plt.show()

series = train_df_d1.iloc[:, 0]
