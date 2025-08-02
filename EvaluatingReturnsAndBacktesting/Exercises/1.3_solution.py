import pandas as pd
import numpy as np
import yfinance as yf
from utils import plot_rolling_mean


# Creating a pandas dataframe
prices = pd.read_csv("data/sp500.csv", index_col=0, parse_dates=True)

# Computing log returns
returns = prices.pct_change()

# Calculating cumulative returns
cumulative_returns = (1 + returns).cumprod()

# Calculating rolling annualized mean return
rolling_mean_1yr = returns.rolling(252).mean() * 252
rolling_mean_3yrs = returns.rolling(756).mean() * 252
rolling_mean_10yrs = returns.rolling(2520).mean() * 252

plot_rolling_mean(cumulative_returns, rolling_mean_1yr)
plot_rolling_mean(cumulative_returns, rolling_mean_3yrs)
plot_rolling_mean(cumulative_returns, rolling_mean_10yrs)
