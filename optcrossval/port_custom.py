import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sco
from scipy import stats

from optcrossval.utils import deal_with_add_constraints


class PortfolioPerformance():
    def __init__(self, portfolio, weights, tag):
        self.ret = None
        self.stdev = None
        self.sharpe = None
        self.diversif = None
        self.VaR = None
        self.calc_portfolio_perf(portfolio, weights)
        self.tag = tag
        [self.__dict__.update({i: j}) for (i, j) in zip(portfolio.assets_names, weights)]

    def calc_portfolio_perf(self, portfolio, weights):
        self.ret = np.sum(portfolio.mean_returns * weights) * portfolio.obs_in_year
        self.stdev = np.sqrt(np.dot(weights.T, np.dot(portfolio.cov, weights))) * portfolio.sr_obs_year
        self.sharpe = (self.ret - portfolio.rf) / self.stdev
        self.diversif = -np.dot(weights, portfolio.sigmas) / (self.stdev / portfolio.sr_obs_year)
        self.VaR = abs(self.ret - (self.stdev * stats.norm.ppf(1 - portfolio.VaR_alpha)))


class Portfolio:

    def __init__(self, df, regularize_cov=False):
        self.returns = (1 + df.pct_change()).apply(np.log)
        self.mean_returns = self.returns.mean()
        self.cov = self.returns.cov()
        self.sigmas = np.sqrt(np.diag(self.cov))
        self.VaR_alpha = 0.05
        self.obs_in_year = 252
        self.weights = None
        self.rf = 0
        self.sr_obs_year = np.sqrt(self.obs_in_year)
        self.num_assets = len(self.mean_returns)
        self.x0 = np.asarray(self.num_assets * [1. / self.num_assets, ])
        self.assets_names = df.columns


class PortfolioOptomizationCostFunctionsMixin:

    def calc_neg_sharpe(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - self.rf) / portfolio_std
        return -sharpe_ratio

    def calc_neg_div_ratio(self, weights):
        negative_div_ratio = -np.dot(weights, self.sigmas) / np.sqrt(
            np.matmul(np.matmul(weights.reshape(1, -1), self.cov.values), weights.reshape(-1, 1)))
        return negative_div_ratio

    def calc_portfolio_VaR(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights) * self.obs_in_year
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov, weights))) * self.sr_obs_year
        portfolio_var = abs(portfolio_return - (portfolio_std * stats.norm.ppf(1 - self.VaR_alpha)))
        return portfolio_var


class PortfolioOptimization(PortfolioOptomizationCostFunctionsMixin):
    def __init__(self, input_portfolio):
        if isinstance(input_portfolio, Portfolio):
            self.__dict__.update(input_portfolio.__dict__)
        elif isinstance(input_portfolio, pd.DataFrame):
            portfolio = Portfolio(input_portfolio)
            self.__dict__.update(portfolio.__dict__)

        bound = (0.0, 1.0)
        self.bounds = tuple(bound for asset in range(self.num_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        add_constraints = deal_with_add_constraints(self.assets_names, general_constraints=20)
        # add_constraints = []
        self.constraints = constraints + add_constraints

    def all_optims(self):
        return {"diverse": self.max_diversification().x,
                "sharpe": self.max_sharpe_ratio().x,
                "VaR": self.min_VaR().x}

    def max_diversification(self):
        result = sco.minimize(self.calc_neg_div_ratio, self.x0, method='SLSQP', bounds=self.bounds,
                              constraints=self.constraints)
        return result

    def max_sharpe_ratio(self):
        result = sco.minimize(self.calc_neg_sharpe, self.x0, method='SLSQP', bounds=self.bounds,
                              constraints=self.constraints)
        return result

    def min_VaR(self):
        result = sco.minimize(self.calc_portfolio_VaR, self.x0, method='SLSQP', bounds=self.bounds,
                              constraints=self.constraints)
        return result


import pandas as pd


class BeyoundPortolios():
    def __init__(self, input_portfolio):
        if isinstance(input_portfolio, Portfolio):
            self.portfolio = input_portfolio
        else:
            self.portfolio = Portfolio(input_portfolio)

        self.colors_availible = {"b": "blue",
                                 "g": "green",
                                 "r": "red",
                                 "c": "cyan",
                                 "m": "magenta",
                                 "k": "black"}

    def simmulate(self, n_portfolios=100, priors=None):
        list_of_prformances_and_weights = []
        for i in range(n_portfolios):
            weights = np.random.random(len(self.portfolio.mean_returns))
            if priors is not None:
                priors = np.random.normal(priors, 0.00001)
                priors[priors < 0] = 0
                weights = weights * priors
            weights /= np.sum(weights)
            portfolio_attributes = PortfolioPerformance(self.portfolio, weights, tag="simul").__dict__
            list_of_prformances_and_weights.append(portfolio_attributes)

        return pd.DataFrame(list_of_prformances_and_weights)

    def optimize(self):
        portfolio_optimization = PortfolioOptimization(self.portfolio)
        all_tags_weights = portfolio_optimization.all_optims()

        list_of_perf = []
        for tag, weights in all_tags_weights.items():
            list_of_perf.append(PortfolioPerformance(self.portfolio, weights, tag=tag).__dict__)

        return pd.DataFrame(list_of_perf)

    def prepare_df_of_best(self, df_simmulated, df_optimized):

        max_sharpe_port = df_simmulated.iloc[df_simmulated['sharpe'].idxmax()]

        # locate positon of portfolio with minimum standard deviation
        min_vol_port = df_simmulated.iloc[df_simmulated['stdev'].idxmin()]
        max_diverse_port = df_simmulated.iloc[df_simmulated['diversif'].idxmin()]
        min_VaR_port = df_simmulated.iloc[df_simmulated['VaR'].idxmin()]
        dict_optimals = {"sharpe_sim": max_sharpe_port,
                         "vol_sim": min_vol_port,
                         "diverse_sim": max_diverse_port,
                         "VaR_sim": min_VaR_port}
        for idx, row in df_optimized.iterrows():
            dict_optimals.update({row['tag']: row})
        optim_df = pd.DataFrame(dict_optimals).T
        return optim_df

    def visualize(self, df_simmulated, optim_df):
        plt.subplots(figsize=(10, 10))
        plt.scatter(df_simmulated.stdev,
                    df_simmulated.ret,
                    c=df_simmulated.sharpe, cmap='RdYlBu')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Returns')
        # plt.xlim((0.00,0.2))
        # plt.ylim((0.00,0.2))
        plt.colorbar()
        for idx, row in optim_df.iterrows():
            color = self.colors_availible[random.choice(list(self.colors_availible.keys()))]
            if "_sim" not in idx:
                marker = (5, 1, 0)
                plt.scatter(row.stdev, row.ret, marker=marker, color=color, s=100)
                plt.annotate(idx, (row.stdev + 0.001, row.ret + 0.003), color=color)

            else:
                marker = "s"
                plt.scatter(row.stdev, row.ret, marker=marker, color=color, s=100)
                plt.annotate(idx, (row.stdev + 0.001, row.ret - 0.003), color=color)

            print(marker)
            # plt.annotate(idx, (row.stdev +0.01, row.ret))
        plt.show()
