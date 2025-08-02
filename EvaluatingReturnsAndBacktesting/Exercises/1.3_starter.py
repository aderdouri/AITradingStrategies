import pandas as pd
import numpy as np
import yfinance as yf
from utils import plot_rolling_mean


# Downloading S&P 500 futures data
ticker = "ES=F"  # S&P 500 front-month futures ticker symbol
data = yf.download(ticker)["Adj Close"]

# Creating a pandas dataframe
sp500_data = pd.DataFrame(data)

# Computing log returns
log_returns = None

# Calculating cumulative returns
cumulative_returns = None

# Calculating rolling annualized mean return
rolling_mean_1yr = None
rolling_mean_3yrs = None
rolling_mean_10yrs = None

plot_rolling_mean(cumulative_returns, rolling_mean_1yr)
plot_rolling_mean(cumulative_returns, rolling_mean_3yrs)
plot_rolling_mean(cumulative_returns, rolling_mean_10yrs)
