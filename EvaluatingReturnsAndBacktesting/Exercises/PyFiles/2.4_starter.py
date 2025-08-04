import numpy as np
import yfinance as yf
from utils import plot_returns

# Download S&P 500 front month futures data
prices = yf.download(['ES=F', 'ZN=F'])["Adj Close"].resample("M").last()
# Calculate logarithmic returns
log_returns = np.log(prices).diff()
# Set portfolio weights
# YOUR CODE HERE
weights = None
# Calculate weighted returns
# YOUR CODE HERE
weighted_returns = None
# YOUR CODE HERE
# Calculate portfolio returns
portfolio_returns = None

plot_returns(portfolio_returns)
