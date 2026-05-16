# Forecasting Brent Crude Oil Prices Using ARIMA in Python

## Data Preparation

We use Brent crude oil spot prices from FRED (`DCOILBRENTEU`).

Daily prices are aggregated to monthly averages to reduce high-frequency noise.

The series is log-transformed to:

- Account better for multiplicative structrue of the price  
- Interpret differences as approximate percentage changes  
- Improve statistical properties for time-series modelling as ARIMA is modelling additive relationship

The data is split to 
- 1997-2004 Training
- 2004-2006 Validation
- 2006-2007 Testing
---

## Differencing and Stationarity Testing

Brent prices are non-stationary in levels.  
We difference the log-transformed series once:

$\[
\Delta \log P_t = \log P_t - \log P_{t-1}
\]$

To determine the appropriate order of integration, we apply:

- **Augmented Dickey–Fuller (ADF) test**  
  - Null hypothesis: the series has a unit root (non-stationary)

- **KPSS test**  
  - Null hypothesis: the series is stationary

### Results

- The ADF test rejects the unit root hypothesis for the first-differenced series.
- The KPSS test fails to reject stationarity.

This confirms that the log price series is **integrated of order one**, $\(I(1)\)$, and that first differencing is sufficient.
