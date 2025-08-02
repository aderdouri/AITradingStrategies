import numpy as np
import yfinance as yf
from utils import plot_returns

# Download S&P 500 front month futures data
prices = yf.download('ES=F')['Adj Close']

# Calculate daily logarithmic returns
log_returns = np.log(prices).diff()

# Annualize returns and volatility
annualized_return = None
annualized_vol = None
# Download the 3-month Treasury bill rate as the risk-free rate
sp500_start_date = str(prices.index[0])[:10]
risk_free_rates = yf.download('^IRX', start=sp500_start_date)['Adj Close']
avg_risk_free_rate = risk_free_rates.mean() / 100

# Calculate the Sharpe Ratio
sharpe_ratio = None
# Calculate the Sortino Ratio
downside_vol = None
sortino_ratio = None
# Calculate the Calmar Ratio
cum_returns = None
drawdowns = None
max_drawdown = None
calmar_ratio = None

print()
print(f"annualized_return: {np.round(annualized_return * 100, 1)}%")
print(f"avg_risk_free_rate: {np.round(avg_risk_free_rate * 100, 1)}%")
print(f"annualized_volatility: {np.round(annualized_vol * 100, 1)}%")
print(f"downside_volatility: {np.round(downside_vol * 100, 1)}%")
print(f"max_drawdown: {np.round(max_drawdown * 100, 1)}%")
print()
print(f"sharpe_ratio: {np.round(sharpe_ratio, 2)}")
print(f"sortino_ratio: {np.round(sortino_ratio, 2)}")
print(f"calmar_ratio: {np.round(calmar_ratio, 2)}")
print()

plot_returns(log_returns)
