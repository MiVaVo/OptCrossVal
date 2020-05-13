from typing import Tuple

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models, objective_functions, \
    DiscreteAllocation
from sklearn.base import BaseEstimator

from configs import *
from fincrossval.utils import autodetect_frequency


class OptPortfolio(BaseEstimator):
    '''
    Portfolio optimization used both for train and validation
    '''

    def __init__(self, target_function: str = "efficient_risk", target_function_params: dict = {}, budget: int = 10000):
        self.target_function = target_function
        self.target_function_params = target_function_params
        self.budget = budget
        self.frequency = None
        # self.mu = expected_returns.mean_historical_return(prices)
        # self.S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        # self.ef = EfficientFrontier(self.mu, self.S)
        # self.ef.add_objective(objective_functions.L2_reg)
        # getattr(self.ef,target_function)(**target_function_params)
        # self.weights = self.ef.clean_weights()
        #
        # self.da = DiscreteAllocation(self.weights, prices.iloc[-1], total_portfolio_value=money)
        # alloc, leftover = self.da.lp_portfolio()
        # print(f"Leftover: ${leftover:.2f}")
        # print(f"Alloc: ${alloc:.2f}")
        # self.discrete_portfolio = (
        #         prices[[k for k, v in alloc.items()]] * np.asarray([[v for k, v in alloc.items()]])).dropna().sum(
        # axis=1)

    def fit(self, prices: pd.DataFrame):
        # TODO: make prices instance of portfolio
        '''
        Given prices finds wieghts (coefficients), most optimal portfolio given target function.
        :param prices:
        :return:
        '''
        self.frequency = autodetect_frequency(prices)
        self.mu = expected_returns.mean_historical_return(prices, frequency=self.frequency)
        self.S = risk_models.CovarianceShrinkage(prices, frequency=self.frequency).ledoit_wolf()
        # self.S=risk_models.exp_cov(prices)
        self.ef = EfficientFrontier(self.mu, self.S)
        self.ef.add_objective(objective_functions.L2_reg)
        getattr(self.ef, self.target_function)(**self.target_function_params)
        self.coef_ = self.ef.clean_weights()
        return self

    def predict(self, prices: pd.DataFrame):
        '''
        Predicts portfolio given found optimal portoflio weights.
        :param prices:
        :return: optimal portofolio (in currency of input)
        '''
        self.allocate(prices)
        # da = DiscreteAllocation(self.coef_, prices.iloc[-1], total_portfolio_value=self.budget)
        # alloc, leftover = da.lp_portfolio()
        # print(f"Leftover: ${leftover:.2f}")
        # print(f"Alloc: ${alloc}")
        discrete_portfolio = (
                prices[[k for k, v in self.alloc.items()]] * np.asarray([[v for k, v in self.alloc.items()]])).dropna()
        return discrete_portfolio

    def allocate(self, prices: pd.DataFrame) -> Tuple[dict, float]:
        '''
        Given weights for portfolio and last prices, finds best discrete allocation of assets.
        :param prices:
        :return: allocation and leftover of portfolio
        '''
        da = DiscreteAllocation(self.coef_, prices.iloc[-1], total_portfolio_value=self.budget)
        self.alloc, self.leftover = da.lp_portfolio()
        print("Period end:", prices.index[0])
        print(f"Leftover: ${self.leftover:.2f}")
        print(f"Alloc: ${self.alloc}")

        return self.alloc, self.leftover

    @property
    def weights(self):
        return self.coef_

    @property
    def allocations(self):
        return self.alloc

    @property
    def leftover(self):
        return self.leftover
