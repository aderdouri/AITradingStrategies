import pandas as pd
import numpy as np
import yfinance as yf
from utils import plot_returns


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
# Compute risk parity weights
annual_vol = None
inverse_vol = None
risk_parity_weights = None
weighted_returns = None

# Annualize returns and volatility
annualized_return = None
annualized_vol = None

# Calculate the Sharpe Ratio
sharpe_ratio = None
# Calculate the Sortino Ratio
downside_vol = None
sortino_ratio = None
# Calculate the Calmar Ratio
cum_returns = None
drawdowns = None
max_drawdown = None
calmar_ratio = None

print()
print(f"annualized_return: {np.round(annualized_return * 100, 1)}%")
print(f"annualized_volatility: {np.round(annualized_vol * 100, 1)}%")
print(f"downside_volatility: {np.round(downside_vol * 100, 1)}%")
print(f"max_drawdown: {np.round(max_drawdown * 100, 1)}%")
print()
print(f"sharpe_ratio: {np.round(sharpe_ratio, 2)}")
print(f"sortino_ratio: {np.round(sortino_ratio, 2)}")
print(f"calmar_ratio: {np.round(calmar_ratio, 2)}")
print()

plot_returns(weighted_returns)
