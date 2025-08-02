
import numpy as np
import yfinance as yf
from utils import plot_returns

# Download S&P 500 front month futures data
prices = yf.download(['ES=F', 'ZN=F'])["Adj Close"].resample("M").last()
# Calculate logarithmic returns
log_returns = np.log(prices).diff()
# Set portfolio weights
weights = np.array([.6, .4])
# Calculate weighted returns
weighted_returns = log_returns * weights
# Calculate portfolio returns
portfolio_returns = weighted_returns.sum(axis=1)

plot_returns(portfolio_returns)
