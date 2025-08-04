import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Downloading S&P 500 futures data
ticker = "ES=F"  # S&P 500 front-month futures ticker symbol
data = yf.download(ticker)["Adj Close"]

# Creating a pandas dataframe
sp500_data = pd.DataFrame(data)

# Computing log returns
# YOUR CODE HERE
log_returns = None

# Calculating cumulative returns
# YOUR CODE HERE
cumulative_returns = None

# Plotting cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns - 1, label='Cumulative Returns')
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

