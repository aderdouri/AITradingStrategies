import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


def compute_drawdowns(log_returns):
    # Calculate cumulative returns
    # YOUR CODE HERE
    cumulative_returns = None
    # Calculate ongoing drawdown
    # YOUR CODE HERE
    rolling_max = None
    drawdowns = None
    return drawdowns

def plot_returns(log_returns):
    # Create figure and axis objects
    fig, ax = plt.subplots()
    # Calculate cumulative returns
    # YOUR CODE HERE
    cumulative_returns = None
    # Calculate ongoing drawdown
    # YOUR CODE HERE
    drawdowns = None
    # Plotting the cumulative returns
    # YOUR CODE HERE
    # Plotting the drawdowns
    # YOUR CODE HERE
    # Setting x-axis major locator to each year and formatter
    # YOUR CODE HERE
    # Setting y-axis ticks every 20%
    # YOUR CODE HERE
    # Adding grid with vertical lines for each year
    # YOUR CODE HERE
    # Rotate x-axis labels by 45 degrees
    # YOUR CODE HERE
    ax.legend()
    ax.set_title('S&P500 Cumulative Returns and Drawdowns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns/Drawdown')
    fig.tight_layout()
    plt.show()

# Download S&P 500 front month futures data
sp500_futures_prices = yf.download('ES=F')["Adj Close"]

# Calculate logarithmic returns
log_returns = np.log(sp500_futures_prices).diff()

plot_returns(log_returns)
