import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs
from pandas_datareader import get_data_yahoo

from fincrossval.custom_ttt import TimeSeriesSplitLargeTest

qs.extend_pandas()

from datetime import datetime
from configs import start_date, end_date, COMPANIES_LIST
from research.hypothesis_testing import \
    Ho_mean_expected_returns_are_equal_for_diversivied_and_random_weights
from fincrossval.utils import get_last_raw, diversify_portfolio, roll, smartDataReader, \
    boostraped_diversify_portfolio

# We would like all available data from 01/01/2000 until 12/31/2016.
# User pandas_reader.data.DataReader to load the desired data. As simple as that.
panel_df_1 = smartDataReader(name=COMPANIES_LIST,
                             data_source='yahoo',
                             start=start_date,
                             end=end_date,
                             force_reload=True,
                             retry_count=30)
bac = get_data_yahoo('KO', start_date, end_date)

# panel_df_1[-10:]
# panel_df_1[-20:]
# panel_df_1['Close'][-20:].isna().any()
# panel_df_1['Close']['GPS'][-20000:]
# panel_df_1 = panel_df_1[panel_df_1.index <= datetime.strptime("2020-01-01", '%Y-%m-%d')]
# panel_df_1['Close']['DAL'].dropna()
columns_leave = panel_df_1['Close'].columns[~panel_df_1['Close'][:3].isna().any()]

df_day = panel_df_1['Close'][columns_leave].dropna()
df_week = panel_df_1['Close'][columns_leave].dropna().groupby(pd.Grouper(key=None, freq='W')).apply(
    get_last_raw).dropna()
df_month = panel_df_1['Close'][columns_leave].dropna().groupby(pd.Grouper(key=None, freq='M')).apply(
    get_last_raw).dropna()
# different_intervals_dfs={
#     "day": df_day,
#     "week": df_week,
#     "month": df_month,
# }
# different_type_of_returns={
#     "day_returns": df_day.diff(),
#     "day_log_returns": np.log(df_day).diff(),
#     "week_returns":df_week.diff(),
#     "week_log_returns":np.log(df_week).diff(),
#     "month_returns":df_month.diff(),
#     "month_log_returns": np.log(df_month).diff(),
#     # "week_pct_change":df_week.pct_change()
# }

weights_results = []
# df_month=df_month.reset_index()
# df_month.loc[train_index]
sharpes = []
df_ = df_week
cross_validated_portfolio = []
tss = TimeSeriesSplitLargeTest(n_splits=3)
list_of_weights_df = []
weights_from_previous_it = None
chra = 100
# sharpes=[]
for train_index, test_index in tss.split(df_):
    print(len(train_index))
    # if len(train_index)<int(tss.max_train_size*3/4):
    #     continue
    # if np.random.rand()>0.5:
    #     break
    # print(len(train_index))
    # print(test_index)
    df_train = df_.loc[train_index]
    df_test = df_.loc[test_index]
    returns_train = df_train.diff().dropna()
    # return_test=np.log(df_test).diff().dropna()
    res, weights = boostraped_diversify_portfolio(returns_train, n_iters=2,
                                                  bootsrap=False)
    print(weights['DAL'])
    # if weights_from_previous_it is None:
    #
    #     res, weights = boostraped_diversify_portfolio(returns_train,n_iters=2,
    #                                                   bootsrap=False)
    #     weights_from_previous_it=weights
    # else:
    #     specific_costraints=[[name,
    #                          'ineq',
    #                          [1/chra*v,
    #                           chra*v if v!=0 else 1.0]] for (name,v) in weights_from_previous_it.reset_index().values]
    #     res, weights = boostraped_diversify_portfolio(returns_train, n_iters=2,
    #                                                  bootsrap=False,specific_costraints=specific_costraints)
    #     weights_from_previous_it = weights
    #
    # #
    list_of_weights_df.append(weights)
    # portfolio_dynamic=(df_test.diff()*weights).sum(axis=1).dropna()
    # sharpe_dynamic=qs.stats.sharpe(portfolio_dynamic)
    # sharpes.append(sharpe_dynamic)
    # portfolio_dynamic_EW = (np.log(df_test).diff()).mean(axis=1).dropna()
    # ew_sharpe=qs.stats.sharpe(portfolio_dynamic_EW)

    # weights_df=pd.DataFrame([weights]*len(test_index))
    # weights_df.index=test_index
    # list_of_weights_df.append(weights_df)

weights_df_Res = pd.concat(list_of_weights_df, axis=1)
weights_res = weights_df_Res.mean(axis=1)
moeney_I_have = 100
df_day.iloc[-1, :]
weights_res * moeney_I_have
# weights_df_Res.T[weights_df_Res.mean(axis=0)>0.01].T.boxplot()
# weights_df_Res
plt.show()
# weights_df_Res=weights_df_Res.reset_index().groupby("Date").mean()
portfolio_dynamic = (df_.loc[weights_df_Res.index] * weights_df_Res).sum(axis=1)
qs.plots.snapshot(portfolio_dynamic.dropna(), title="portfolio_dynamic")

# portfolio_dynamic.grouby(index=True).mean()
portfolio_dynamic_EW = (df_day.loc[min(weights_df_Res.index):]).mean(axis=1).dropna()
qs.plots.snapshot(portfolio_dynamic_EW.dropna(), title="portfolio_dynamic_EW")

# df_month.set_index("Date")

return_plots = []
k = 'month_log_returns'
# df=different_type_of_returns[k]
# for k,df in different_type_of_returns.items():
# df=df.dropna()
# df=df[:-15]
df = df[df.index >= datetime.strptime("2015-01-01", '%Y-%m-%d')]

weights_df = pd.DataFrame(weights.round(3), df.columns).T

weights_df.index = [k]
weights_results.append(weights_df)
# print(k)
# print(weights_df)
# plt.plot(weights_df.T)
# k.split("_")[0]:
# sharps={"ew":[calc_sharp_ratio(df_day,np.repeat(1/len(weights),len(weights)))],
#         "opt":[calc_sharp_ratio(df_day,weights)]}

f1 = qs.plots.snapshot(portfolio, title=k)  # print(pd.DataFrame(sharps))
weights_ew = np.repeat(1 / len(weights), len(weights))
portfolio = (different_intervals_dfs[k.split("_")[0]].dropna() * weights_ew).sum(axis=1).pct_change()
f2 = qs.plots.snapshot(portfolio, title=k + "_EW")
# return_plots.append(f1.axes[0])
# f2.show()
# plt.plot(weights)
# return_plots[1].plot()
# plt.show()
# fig, axes = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]})
# return_plots[0].plot()
# plt.show()
# [np.put(axes,i,return_plots[i]) for i in range(len(return_plots))]
pd.concat(weights_results).T.plot()
plt.show()


# np.log(weights)/sum(np.log(weights))
# print(k,weights.round(3))

def diversify_portfolio_rolling(df):
    res, weights = diversify_portfolio(df)
    return pd.DataFrame(data=weights, columns=df.columns)


df = different_type_of_returns['week_log_returns'].dropna()
Ho_mean_expected_returns_are_equal_for_diversivied_and_random_weights(df)
roll(df, 1000).apply(diversify_portfolio_rolling)

win = sliding_windows(df, 3)
different_type_of_returns['week_log_returns'].window(100).apply(diversify_portfolio_rolling)
res, weights = diversify_portfolio(different_type_of_returns['week_log_returns'])

# weights_distributions=pd.DataFrame(np.asarray(statistics))
# weights_distributions.columns=sampled_working_data.columns
# weights_distributions.median().round(2)
# weights_distributions.hist()
plt.show()
# res,weights=diversify_portfolio(df_week.diff()[:20].dropna())
# df_week.diff()['RUBUSD=X'].plot()
# plt.show()
# panel_df_1_close=panel_df_1.reset_index().resample('W-Mon', on='Date').last()
# panel_df_1_close['Close'].corr()
# plt.show()
# df = panel_df_1.reset_index().groupby(['Date', pd.Grouper(key='Date', freq='W-MON')]).apply(get_first_price)

# offset = pd.offsets.timedelta(days=-6)
# panel_df_1.resample('W', loffset=offset).apply(logic)

df_close = panel_df_1.T.xs("Close", level=0).T

df_close_diff = df_close.diff()
# panel_df_1.groupby()
# normalized_df=(df_close-df_close.mean())/df_close.std()
df_close_diff.corr()
# normalized_df.plot()
# panel_df_1.plot()
# plt.show()
