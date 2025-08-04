import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator


def compute_drawdowns(returns):
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    # Calculate ongoing drawdown
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return drawdowns


def plot_returns(returns):
    # Create figure and axis objects
    fig, ax = plt.subplots()
    # Calculate cumulative returns
    cumulative_returns = (1 + returns).cumprod()
    # Calculate ongoing drawdown
    drawdowns = compute_drawdowns(returns)
    # Plotting the results
    ax.plot(cumulative_returns - 1, label='Cumulative Returns')
    ax.fill_between(drawdowns.index, drawdowns, label='Drawdowns', color='red', alpha=0.3)
    # Setting x-axis major locator to each year and formatter
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Adding grid with vertical lines for each year
    ax.grid(True, which='major', linestyle='--', color='grey')
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    ax.legend()
    ax.set_title('Cumulative Returns and Drawdowns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns/Drawdown')
    fig.tight_layout()
    plt.show()


def print_metrics(returns):
    # Annualize returns and volatility
    annualized_return = returns.mean() * 12
    annualized_vol = returns.std() * np.sqrt(12)

    # Calculate the Sharpe Ratio
    sharpe_ratio = annualized_return / annualized_vol
    # Calculate the Sortino Ratio
    downside_vol = returns[returns<0].std() * np.sqrt(12)
    sortino_ratio = annualized_return / downside_vol
    # Calculate the Calmar Ratio
    cum_returns = np.exp(returns.cumsum())
    drawdowns = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
    max_drawdown = np.max(drawdowns)
    calmar_ratio = annualized_return / max_drawdown

    print()
    print(f"annualized_return: {np.round(annualized_return * 100, 1)}%")
    print(f"annualized_volatility: {np.round(annualized_vol * 100, 1)}%")
    print(f"downside_volatility: {np.round(downside_vol * 100, 1)}%")
    print(f"max_drawdown: {np.round(max_drawdown * 100, 1)}%")
    print()
    print(f"sharpe_ratio: {np.round(sharpe_ratio, 2)}")
    print(f"sortino_ratio: {np.round(sortino_ratio, 2)}")
    print(f"calmar_ratio: {np.round(calmar_ratio, 2)}")
    print()


def plot_rolling_mean(cumulative_returns, rolling_mean):
    # Create figure and axis objects
    fig, ax = plt.subplots()
    # Plotting the results
    ax.plot(cumulative_returns - 1, label='Cumulative Returns')
    ax.plot(rolling_mean, label='Rolling Mean', linestyle="--", linewidth=1, color="green")
    rolling_mean_min = rolling_mean.cummin().values.reshape(-1)
    rolling_mean_max = rolling_mean.cummax().values.reshape(-1)
    ax.fill_between(
        rolling_mean.index, rolling_mean_min, rolling_mean_max,
        label='Rolling Mean Range', alpha=0.2, color="green"
        )
    # Setting x-axis major locator to each year and formatter
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Setting y-axis ticks every 20%
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # Adding grid with vertical lines for each year
    ax.grid(True, which='major', linestyle='--', color='grey')
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    # Ensure date labels are visible
    ax.legend()
    ax.set_title('Cumulative Returns and Rolling Mean')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    fig.tight_layout()
    plt.show()
