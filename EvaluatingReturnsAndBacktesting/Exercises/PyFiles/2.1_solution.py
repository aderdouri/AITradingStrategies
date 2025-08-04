# Import libraries
import numpy as np
import yfinance as yf

# Use the `yfinance` library to download the front month S&P500 futures price data.
sp500_prices = yf.download('ES=F')['Adj Close']

# Calculate the daily logarithmic returns of the futures prices.
log_returns = np.log(sp500_prices).diff()

# Calculate the standard deviation (volatility) of the logarithmic returns
volatility = log_returns.std()

# Annualize the volatility
# There are approximately 252 trading days in a year
annualized_volatility = volatility * np.sqrt(252)
print()
print(f"annualized_volatility: {np.round(annualized_volatility * 100, 2)}%")
print()
