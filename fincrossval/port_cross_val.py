from inspect import getmembers, isfunction

import pandas as pd
import quantstats as qs

from fincrossval.custom_ttt import TimeSeriesSplitLargeTest
from fincrossval.optimizer import OptPortfolio

# type(OptPortfolio)
functions_list = [o for o in getmembers(qs.stats) if isfunction(o[1])]
QuantStatMetricsMixin = type('Metrics', (), {k: staticmethod(v) for (k, v) in functions_list})


class PortfolioMetrics(QuantStatMetricsMixin):
    '''
    Class to create custom metrics
    '''
    # TODO: Make possible custom portfolio metrics creation
    pass


class FinancialCrossValidation:
    '''
    Cross validation of metrics
    '''

    def __init__(self, n_splits: int, **kwargs):
        '''
        Initialises cross-validation instance
        :param n_splits: number of train/test splits
        :param kwargs:
        '''
        self.tss = TimeSeriesSplitLargeTest(n_splits=n_splits)
        self.metrics_train = {"sharpe": [], "volatility": [], "cvar": []}
        self.metrics_test = {"sharpe": [], "volatility": [], "cvar": []}
        self.calc_allocations_train = []
        self.calc_allocations_test = []

    def _calc_train_metrics(self, prices_or_returns: pd.DataFrame):
        '''
        Given class attribute dict of train metrics, updates it by calling function by name on input
        :param prices_or_returns: pd.Dataframe
        :return:
        '''
        for metric_key in self.metrics_train.keys():
            self.metrics_train[metric_key].append(getattr(qs.stats, metric_key)(prices_or_returns))

    def _calc_test_metrics(self, prices_or_returns: pd.DataFrame):
        '''
        Given class attribute dict of test metrics, updates it by calling function by name on input
        :param prices_or_returns: pd.Dataframe
        :return:
        '''
        for metric_key in self.metrics_test.keys():
            self.metrics_test[metric_key].append(getattr(qs.stats, metric_key)(prices_or_returns))

    def validate(self, df: pd.DataFrame, portfolio_optimizer: OptPortfolio, **kwargs):
        '''
        Make cross-validation of portfolio by finding optimal weights on each split given optimizer and
        also calculates metrics on train and test. There is no forward-looking in this implementation.
        :param df: pd.Dataframe , where indecies are datetime and columns are assets
        :param portfolio_optimizer: Porftolio's weights estimator,e.f. an algorithm that finds weights w1,w2,..,wn ,
        where n is the n-th asset, so that optimum is achieved
        :param kwargs: other kwargs, currently not used
        :return: nothing
        '''
        for train_index, test_index in self.tss.split(df):
            df_train = df.loc[train_index]
            df_test = df.loc[test_index]
            print(df_test.shape)
            df_train = df_train.dropna(axis=0)
            df_test = df_test.dropna(axis=0)
            # print(df_train.shape[0])
            portfolio_optimizer.fit(df_train)
            optimal_train_portfolio = portfolio_optimizer.predict(df_train).sum(axis=1)
            self.calc_allocations_train.append(portfolio_optimizer.alloc)

            optimal_test_portfolio = portfolio_optimizer.predict(df_test).sum(axis=1)
            self.calc_allocations_test.append(portfolio_optimizer.alloc)

            self._calc_train_metrics(optimal_train_portfolio)
            self._calc_test_metrics(optimal_test_portfolio)
