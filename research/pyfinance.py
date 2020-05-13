import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs import COMPANIES_LIST, start_date, end_date
# User pandas_reader.data.DataReader to load the desired data. As simple as that.
from fincrossval.utils import smartDataReader

df = smartDataReader(name=COMPANIES_LIST,
                     data_source='yahoo',
                     start=start_date,
                     end=end_date,
                     force_reload=False,
                     retry_count=30)
columns_leave = df['Adj Close'].columns[~df['Adj Close'][:3].isna().any()]
columns_leave = columns_leave
df = df['Adj Close'][columns_leave].dropna()


def calc_portfolio_perf(weights, mean_returns, cov, rf):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    return portfolio_return, portfolio_std, sharpe_ratio


def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    results_matrix = np.zeros((len(mean_returns) + 3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio = calc_portfolio_perf(weights, mean_returns, cov, rf)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio
        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j + 3, i] = weights[j]

    results_df = pd.DataFrame(results_matrix.T,
                              columns=['ret', 'stdev', 'sharpe'] + [ticker for ticker in mean_returns.index])

    return results_df


mean_returns = df.pct_change().mean()
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
results_frame = simulate_random_portfolios(num_portfolios, mean_returns, cov, rf)

# locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
# locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
# create scatter plot coloured by Sharpe Ratio
plt.subplots(figsize=(15, 10))
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
plt.colorbar()
# plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=500)
# plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=500)
plt.show()
