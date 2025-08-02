import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Downloading S&P 500 futures data
ticker = "ES=F"  # S&P 500 front-month futures ticker symbol
# data = yf.download(ticker)["Adj Close"]

# Creating a pandas dataframe
sp500_data = pd.read_csv("data/sp500.csv", index_col=0)

# Computing log returns
returns = sp500_data.pct_change()

# Calculating cumulative returns
cumulative_return = (1 + returns).cumprod()

# Plotting cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_return - 1, label='Cumulative Returns')
plt.title('S&P500 Front-Month Futures Cumulative Returns')
plt.xlabel('Year')
plt.ylabel('Cumulative Returns')
# Setting x-axis major locator to each year and formatter
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# Adding grid with vertical lines for each year
plt.grid(True, which='major', linestyle='--', color='grey')
# Rotate x-axis labels by 45 degrees
plt.xticks(rotation=45)
plt.legend()
plt.show()

