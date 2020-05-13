import warnings

import numpy as np
from pandas.core.base import PandasObject as _po
from quantstats import utils

from optcrossval.utils import autodetect_frequency


def sharpe(self, rf=0., annualize=True):
    warnings.simplefilter('always', DeprecationWarning)

    """
    calculates the sharpe ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Frequency of returns (252 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
    """
    self.periods = autodetect_frequency(self)

    if rf != 0 and self.periods is None:
        raise Exception('Must provide periods if rf != 0')

    returns = utils._prepare_returns(self, rf, self.periods)
    res = returns.mean() / returns.std()

    if annualize:
        return res * np.sqrt(1 if self.periods is None else self.periods)
    print("good")

    return res


def extend_pandas_custom():
    _po.sharpe = sharpe
