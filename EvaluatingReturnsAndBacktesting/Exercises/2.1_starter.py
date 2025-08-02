# Import libraries
import numpy as np
import yfinance as yf

# Use the `yfinance` library to download the front month S&P500 futures price data.
sp500_prices = yf.download('ES=F')['Adj Close']

# Calculate the daily logarithmic returns of the futures prices.
# YOUR CODE HERE
log_returns = None

# Calculate the standard deviation (volatility) of the logarithmic returns
# YOUR CODE HERE
daily_volatility = None

# Annualize the volatility
# There are approximately 252 trading days in a year
# YOUR CODE HERE
annualized_volatility = None
print()
print(f"annualized_volatility: {np.round(annualized_volatility * 100, 2)}%")
print()
