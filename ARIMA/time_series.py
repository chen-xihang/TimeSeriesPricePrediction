# Time Series Analysis of Brent Crude Oil Spot Prices
import pandas as pd
import pandas_datareader.data as web
import datetime as dt
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath("time_series.py")))

start = dt.datetime(1997, 1, 1)
end = dt.datetime(2007,1,1)

# Step 1: Preparing the data, log of monthly average of daily data
# Federal Reserve Economic Data (FRED) for Brent Spot Price (DCOILBRENTEU)
brent = web.DataReader("DCOILBRENTEU", "fred", start, end)
brent_m = brent.resample("M").mean()

# Plotting brent_m
plt.figure(figsize=(10, 4))
plt.plot(brent_m, label="Brent spot price (monthly avg)")
plt.title("Brent crude oil spot price (monthly)")
plt.xlabel("Date")
plt.ylabel("USD per barrel")
plt.legend()
plt.tight_layout()
plt.savefig("brent_monthly.png", dpi=300, bbox_inches="tight")
plt.show()

# Log transform
brent_log = np.log(brent_m)

plt.figure(figsize=(10, 4))
plt.plot(brent_log, label="Brent spot log price (monthly avg)")
plt.title("Brent crude oil spot log price (monthly)")
plt.xlabel("Date")
plt.ylabel("USD per barrel")
plt.legend()
plt.tight_layout()
plt.show()

# Step 2: Decomposing the time series, I part
brent_log_d1 = brent_log.diff().dropna()

# Visual Test
plt.figure(figsize=(10, 4))
plt.plot(brent_log_d1, label="Brent monthly log returns")
plt.title("Brent monthly log returns first order difference")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.legend()
plt.tight_layout()
plt.show()

series = brent_log_d1.iloc[:, 0]

# Augmented Dickey-Fuller (Root Test)
adf_result = adfuller(series)
print(f"Augmented Dickey-Fuller Test Results: This gives p-value: {adf_result[1]:.4g}, for the number of differencing: {adf_result[2]} times.")

# KPSS Test
kpss_result = kpss(
    series,
    regression="c",  
    nlags="auto"
)
print(f"KPSS Test Results: This gives p-value: {kpss_result[1]:.4g}")

# Step 3: Decomposing the time series, AR part
fig_acf = plot_acf(series, lags=20)
fig_acf.savefig("acf_brent_log_d1.png", dpi=300, bbox_inches="tight")
plt.close(fig_acf)

fig_pacf = plot_pacf(series, lags=20)
fig_pacf.savefig("pacf_brent_log_d1.png", dpi=300, bbox_inches="tight")
plt.close(fig_pacf)