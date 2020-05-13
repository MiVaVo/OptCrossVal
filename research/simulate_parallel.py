import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.extend(["c:/PycharmProjects/finance/research/"])
from configs import COMPANIES_LIST, start_date, end_date
from utils import smartDataReader

pd.set_option("display.max_columns", 101)
df = smartDataReader(name=COMPANIES_LIST, data_source='yahoo', start=start_date, end=end_date, force_reload=False,
                     retry_count=30)
df['Adj Close'].head(2)
tickers = df['Adj Close'].columns[~df['Adj Close'][:3].isna().any()]
tickers
df = df['Adj Close'][tickers].dropna()


def calc_neg_div_ratio(weights, mean_returns, cov, rf):
    V = cov.values
    sigmas = np.sqrt(np.diag(V))
    weights = np.asarray(weights)
    negative_div_ratio = -np.dot(weights, sigmas) / np.sqrt(
        np.matmul(np.matmul(weights.reshape(1, -1), V), weights.reshape(-1, 1)))
    return negative_div_ratio


def calc_portfolio_perf_VaR(weights, mean_returns, cov, alpha=0.05, days=252):
    portfolio_return = np.sum(mean_returns * weights) * days
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(days)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_return, portfolio_std, portfolio_var


def calc_portfolio_perf(weights, mean_returns, cov, rf, alpha=0.05):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(252)
    sharpe_ratio = (portfolio_return - rf) / portfolio_std
    negative_div_ratio = calc_neg_div_ratio(weights, mean_returns, cov, rf)
    portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - alpha)))
    return portfolio_return, portfolio_std, sharpe_ratio, negative_div_ratio, portfolio_var


def simulate_random_portfolios(num_portfolios, mean_returns, cov, rf):
    results_matrix = np.zeros((len(mean_returns) + 5, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio, negative_div_ratio, VaR = calc_portfolio_perf(weights,
                                                                                                     mean_returns, cov,
                                                                                                     rf)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio
        results_matrix[3, i] = negative_div_ratio
        results_matrix[4, i] = VaR

        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j + 5, i] = weights[j]

    results_df = pd.DataFrame(results_matrix.T,
                              columns=['ret', 'stdev', 'sharpe', "divcoef", "VaR"] + [ticker for ticker in tickers])

    return results_df


def _return_perf(m, mean_returns, cov, rf):
    num_portfolios = 10
    results_matrix = np.zeros((len(mean_returns) + 5, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)
        portfolio_return, portfolio_std, sharpe_ratio, negative_div_ratio, VaR = calc_portfolio_perf(weights,
                                                                                                     mean_returns, cov,
                                                                                                     rf)
        results_matrix[0, i] = portfolio_return
        results_matrix[1, i] = portfolio_std
        results_matrix[2, i] = sharpe_ratio
        results_matrix[3, i] = negative_div_ratio
        results_matrix[4, i] = VaR

        # iterate through the weight vector and add data to results array
        for j in range(len(weights)):
            results_matrix[j + 5, i] = weights[j]
    return results_matrix

    # L.append(results_df)
    # return results_df


from joblib import Parallel, delayed


def simulate_random_portfolios_parallel(num_portfolios, mean_returns, cov, rf):
    results_list = Parallel(n_jobs=8)(delayed(_return_perf)(i, mean_returns, cov, rf) for i in range(num_portfolios))

    # results_df=pd.concat(L)
    # iterate through the weight vector and add data to results array
    #
    #
    # results_df = pd.DataFrame(results_matrix.T,
    #                           columns=['ret', 'stdev', 'sharpe', "divcoef", "VaR"] + [ticker for ticker in tickers])
    results_matrix = np.concatenate(results_list, axis=1)
    results_df = pd.DataFrame(results_matrix.T,
                              columns=['ret', 'stdev', 'sharpe', "divcoef", "VaR"] + [ticker for ticker in tickers])
    return results_df


mean_returns = df.pct_change().mean()
print(mean_returns.shape)
cov = df.pct_change().cov()
num_portfolios = 100000
rf = 0.0
from datetime import datetime as dt

st = dt.now()
results_frame = simulate_random_portfolios_parallel(num_portfolios, mean_returns, cov, rf)
print(dt.now() - st)

import matplotlib.pyplot as plt

# locate position of portfolio with highest Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
# locate positon of portfolio with minimum standard deviation
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
max_div_port = results_frame.iloc[results_frame['divcoef'].idxmin()]
min_VaR_port = results_frame.iloc[results_frame['VaR'].idxmin()]
# create scatter plot coloured by Sharpe Ratio
plt.subplots(figsize=(10, 10))
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
plt.xlabel('Standard Deviation')
plt.ylabel('Returns')
# plt.xlim((0.00,0.2))
# plt.ylim((0.00,0.2))
plt.colorbar()
# plot red star to highlight position of portfolio with highest Sharpe Ratio
plt.scatter(max_sharpe_port[1], max_sharpe_port[0], marker=(5, 1, 0), color='r', s=200)
# plot green star to highlight position of minimum variance portfolio
plt.scatter(min_vol_port[1], min_vol_port[0], marker=(5, 1, 0), color='g', s=100)
plt.scatter(max_div_port[1], max_div_port[0], marker=(5, 1, 0), color='b', s=100)
plt.scatter(min_VaR_port[1], min_VaR_port[0], marker=(5, 1, 0), color='y', s=100)

plt.show()
