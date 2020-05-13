import sys

from fincrossval.utils import *

sys.path.extend(['c:/PycharmProjects/finance/src/'])
sys.path.extend(['c:/PycharmProjects/finance/'])
from fincrossval.port_cross_val import *

pd.set_option("display.max_columns", 101)

prices = prepare_FX()
tickers = list(prices.columns)
prices.head(2)

portfolio_optimizer = OptPortfolio(target_function="max_sharpe")
cv = FinancialCrossValidation(n_splits=5)
cv.validate(prices, portfolio_optimizer)
