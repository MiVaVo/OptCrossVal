from datetime import datetime

import pandas as pd
from pandas_datareader import data
from statsmodels.regression.rolling import RollingOLS

ickers = ['AAPL', 'MSFT', '^GSPC', "RDS-B", 'RUB=X']

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = datetime.strptime('2010-01-01', '%Y-%m-%d')
end_date = datetime.strptime('2020-12-31', '%Y-%m-%d')

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_df_1 = data.DataReader(ickers, 'yahoo', start_date, end_date)
panel_df_2 = data.DataReader("EFFR", 'fred', start_date, end_date)
df = panel_df_1.join(panel_df_2, how="left", on="date")
RollingOLS

df_joined = pd.concat([panel_df_1, panel_df_2], axis=1, join='outer')
# python -m ipykernel install --name=finance2

# import pyflux as pf
# model = pf.DynReg(formula=formula, data=df_differenced)
