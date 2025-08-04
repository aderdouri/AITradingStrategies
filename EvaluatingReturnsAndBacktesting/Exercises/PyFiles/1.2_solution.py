# Import libraries
import pandas as pd

# Use the `yfinance` library to download the front month S&P500 futures price data.
sp500_prices = pd.read_csv("data/sp500.csv", index_col=0)

# Calculate the daily logarithmic returns of the futures prices.
returns = sp500_prices.pct_change()

# Annualize the mean of the returns.
annualized_mean_return = returns.mean()[0] * 252
print()
print(f"Annualized Mean Return: {annualized_mean_return:.2%}")
print()

