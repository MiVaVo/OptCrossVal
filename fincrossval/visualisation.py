from typing import List

import pandas as pd
from pypfopt import Plotting

from fincrossval.optimizer import OptPortfolio
from fincrossval.port_cross_val import FinancialCrossValidation


class Visualizer():
    @staticmethod
    def draw_comparing_boxplots(cv: FinancialCrossValidation):
        '''
        Visualizes metrics of cross-validation in box-plots
        :param cv: instance of class FinancialCrossValidation on which .validate was called
        '''
        cv.metrics_train["tag"] = "Train"
        cv.metrics_test["tag"] = "Test"
        res_metrics_df = pd.concat([pd.DataFrame(cv.metrics_train), pd.DataFrame(cv.metrics_test)])
        res_metrics_df.groupby("tag").boxplot(figsize=(10, 5))

    @staticmethod
    def draw_weights(opt_portfolio: OptPortfolio):
        '''
        Draws weights of optimal portfolio
        :param opt_portfolio:
        '''
        Plotting.plot_weights(opt_portfolio.coef_)

    @staticmethod
    def draw_pies(list_of_portfolios: List[OptPortfolio]):
        raise NotImplementedError
