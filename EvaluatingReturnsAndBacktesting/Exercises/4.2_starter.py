import pandas as pd
import numpy as np
import yfinance as yf
from utils import plot_returns, print_metrics


# Download front-month futures data of S&P500, 10-year Treasuries, gold and US dollar
symbols = ['ES=F', 'ZN=F', 'GC=F', 'DX=F']
data = yf.download(symbols)
# Resample data so that we deal with monthly data instead of daily to reduce noise
data = data.resample("M").last()
data.index = pd.to_datetime(data.index)
# Subset adjusted close prices and fill NaNs with value know at time t
# Drop rows with unknown prices in the beginning of the dataset
prices = data["Adj Close"].ffill().dropna()
prices.index = pd.to_datetime(prices.index)

# Compute logarithmic returns
log_returns = np.log(prices).diff()

def compute_risk_parity_weights(returns, window_size=36):
    # compute volatility known at time t
    rolling_vol = None
    rolling_inverse_vol = None
    # divide inverse volatility by the sum of inverse volatilities
    risk_parity_weights = None
    return risk_parity_weights

# Compute risk parity weights
risk_parity_weights = compute_risk_parity_weights(log_returns)
# shift weights by one period to use only information available at time t
risk_parity_weights = None
weighted_returns = None

print_metrics(weighted_returns)
plot_returns(weighted_returns)
