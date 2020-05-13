import numpy as np
import pandas as pd

from optcrossval.utils import diversify_portfolio

x = np.random.rand(10000)
# ax=3*x+np.random.rand(10000)
# bx=4*x+np.random.rand(10000)

y = np.random.rand(10000)
# ay=2*y++np.random.rand(10000)


cx = np.random.rand(10000) * np.random.rand(10000) + np.random.rand(10000)

df = pd.DataFrame([x, y]).T
df.corr()
_, w = diversify_portfolio(df)
np.corrcoef([x, y])
print(w.round(3))
