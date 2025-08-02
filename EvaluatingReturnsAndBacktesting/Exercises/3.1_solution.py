import numpy as np
import yfinance as yf

# Download S&P 500 front month futures data
prices = yf.download('ES=F')['Adj Close']

# Calculate daily logarithmic returns
log_returns = np.log(prices).diff()

# Annualize returns and volatility
annualized_return = log_returns.mean() * 252
annualized_vol = log_returns.std() * np.sqrt(252)

# Download the 3-month Treasury bill rate as the risk-free rate
sp500_start_date = str(prices.index[0])[:10]
risk_free_rates = yf.download('^IRX', start=sp500_start_date)['Adj Close']
avg_risk_free_rate = risk_free_rates.mean() / 100

# Calculate the Sharpe Ratio
sharpe_ratio = (annualized_return - avg_risk_free_rate) / annualized_vol
# Calculate the Sortino Ratio
downside_vol = log_returns[log_returns<0].std() * np.sqrt(252)
sortino_ratio = (annualized_return - avg_risk_free_rate) / downside_vol
# Calculate the Calmar Ratio
cum_returns = np.exp(log_returns.cumsum())
drawdowns = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
max_drawdown = np.max(drawdowns)
calmar_ratio = annualized_return / max_drawdown

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
