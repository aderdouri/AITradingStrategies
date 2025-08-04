# Import libraries
import numpy as np
import yfinance as yf

# Use the `yfinance` library to download the front month S&P500 futures price data.
sp500_prices = yf.download('ES=F')['Adj Close']

# Calculate the daily logarithmic returns of the futures prices.
# YOUR CODE HERE
log_returns = None

# Annualize the mean of the logarithmic returns.
# YOUR CODE HERE
annualized_mean_return = None
print()
print(f"annualized_mean_return: {np.round(annualized_mean_return * 100, 4)}%")
print()
