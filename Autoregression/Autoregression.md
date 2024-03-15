Part 1: Autoregression Model
---

The Autoregressive model is a time series modeling technique used to represent a type of random process. It is used when there's a direct relationship between an observation and the observations at previous steps. The model specifies that the output variable depends linearly on its own previous values and on a stochastic term.

The math behind the Autoregressive (AR) model of order p (AR(p)) can be written as:

    Yt = c + φ1Yt-1 + φ2Yt-2 + ... + φpYt-p + εt

Where,
  Yt: Output at time t
  c: Constant
  φi: Coefficient for the lagged output at i steps previous to time t
  εt: Error at time t

Assumptions:
1. The model assumes stationarity, that is, the properties of the series do not depend on the time at which the series is observed.
2. The errors are normally distributed with constant variance (homoscedasticity).
3. Errors are uncorrelated.

Examples:
Autoregression models are widely used in signal processing, economics, and climate modeling. For instance, predicting the value of a stock based on previous days' prices.

Part 2: Data Generation and Checking Assumptions
---

Let's generate some data and check whether it meets the assumptions of an autoregressive model.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Generate autoregressive data
np.random.seed(0)
n_samples = 100
a = 0.6
x = w = np.random.normal(size=n_samples)

for t in range(n_samples):
    x[t] = a*x[t-1] + w[t]

plt.plot(x)
plt.title("Autoregressive Data")
plt.show()

# Check for stationarity
result = adfuller(x)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
```

In the above code, we first generate some autoregressive data, and then we check for stationarity using the Augmented Dickey-Fuller test (ADF test). If the p-value of the ADF test is less than our chosen significance level (say 5%), we can reject the null hypothesis of non-stationarity.

Part 3: Model Selection for Autoregression
---

Model selection involves determining the order of the autoregressive model, i.e., how many lagged past values we should use in our prediction. We use the Akaike Information Criterion (AIC) to determine the best model.

```python
from statsmodels.tsa.ar_model import AutoReg
from itertools import product

# Define a range of lag values
max_lag = 10
lags = range(1, max_lag + 1)

# Find the best model (the one with the lowest AIC)
best_aic, best_order = np.inf, None

for lag in lags:
    model = AutoReg(x, lags=lag)
    result = model.fit()
    if result.aic < best_aic:
        best_aic = result.aic
        best_order = lag

print(f'Best model: AR({best_order}), AIC = {best_aic}')
```

In this code, we fit an autoregressive model to our data for a range of different lag values. We then select the model which has the smallest AIC value.

Part 4: Interpreting Model Results
---

After fitting our autoregressive model, we can inspect several quantities to interpret the results.

```python
# Fit the best model
model = AutoReg(x, lags=best_order)
result = model.fit()

# Print the summary
print(result.summary())
```

The `summary` method provides a lot of statistical information. The most important is probably the coefficient table, which shows the estimated coefficients and their standard errors. The "coef" column shows the value of the coefficients, and the "P>|z|" column shows the p-value of a two-sided hypothesis test for each coefficient. If the p-value is less than 0.05, this indicates that there is a statistically significant relationship between the lagged output and the output.