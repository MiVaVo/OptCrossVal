import matplotlib.pyplot as plt
import pandas  as pd
import quantstats as qs
from pypfopt import CovarianceShrinkage

from configs import *
from optcrossval.custom_ttt import TimeSeriesSplitLargeTest
from optcrossval.port_cross_val import OptPortfolio
from optcrossval.port_custom import Portfolio, BeyoundPortolios, PortfolioPerformance, PortfolioOptimization
from optcrossval.utils import smartDataReader

df = smartDataReader(name=COMPANIES_LIST, data_source='yahoo', start=start_date, end=end_date, force_reload=False,
                     retry_count=30)
tickers = df['Adj Close'].columns[~df['Adj Close'][:3].isna().any()]

df = df['Adj Close'][tickers].dropna()


def do_all(df):
    portfolio = Portfolio(df)
    upon_portf = BeyoundPortolios(portfolio)
    df_optimized = upon_portf.optimize()
    priors = df_optimized[df_optimized['tag'] == 'VaR'][upon_portf.portfolio.assets_names].values[0]
    df_simmulates = upon_portf.simmulate(10000, priors=priors)
    optim_df = upon_portf.prepare_df_of_best(df_simmulates, df_optimized)
    upon_portf.visualize(df_simmulates, optim_df)
    return optim_df


def do_all(df, weights):
    portfolio = Portfolio(df)
    performance = PortfolioPerformance(portfolio, weights)


tss = TimeSeriesSplitLargeTest(n_splits=2, max_train_size=900)
perf = []
#
# # pts='C:/PycharmProjects/finance/data/yahoo_20191130000000_AAALAAPLABCABTADBEADIADMADPADSKAEEAESAFLAGNAIGALLAMATAMGNAONAPAAPDAVYAXPAZOBABACBAXBBYBDXBFBKBLLBPBSXCCAGCAHCATCCLCICINFCLXCMACMICMSCNPCOFCOLMCOSTCSCOCSXCTASCTLCTXSCVXDALDDDISDOVDUKDVNECLEFXEMNEMREOGEQRFDXFISVGEGISGLGPCGPSGRAGSGWWHASHDHESHONHPQHRBHWMIBMIFFINTCINTUIPGITWJNJJPMJWNKEYKOLBLINLLYLMTLNCLOWLUVMASMCDMDTMMCMMMMRKMROMSFTMUNEENEMNKENOCNTAPNUENVDANWLORCLOXYPCARPFEPGPGRPHPHMPKIPNCRCLRDSBRFRHIRTXRYAAYSAVESCHWSHWSLBSNASPGSWKSYKSYYTTAPTFCTGTTIFTJXTOTTRVTXTUNHUNMUPSVVFCVMCVZWBAWFCWMBWMTXLNXXOMYUMZBHZION_30_20100101000000.pickle'
# # df=pd.DataFrame([1,2,3])
# # df.to_pickle(pts[:258])
tt = {"train": [], "test": []}

for train_index, test_index in tss.split(df):
    print(f"test_index {len(train_index)}, test_index {len(test_index)}")
    # if len(train_index)<int(tss.max_train_size*3/4):
    #     continue
    # if np.random.rand()>0.5:
    #     break
    # print(len(train_index))
    # print(test_index)
    df_train = df.loc[train_index]
    df_test = df.loc[test_index]

    portfolio_train = Portfolio(df_train)
    upon_portf = BeyoundPortolios(portfolio_train)
    df_optimized = upon_portf.optimize()
    # priors=df_optimized[df_optimized['tag']=='sharpe'][upon_portf.portfolio.assets_names].values[0]

    df_simmulates = upon_portf.simmulate(100)
    optim_df = upon_portf.prepare_df_of_best(df_simmulates, df_optimized)
    # upon_portf.visualize(df_simmulates, optim_df)
    max_sharpe_sim_x = optim_df[portfolio_train.assets_names].T['sharpe_sim']

    portfolio_optimization = PortfolioOptimization(portfolio_train)
    weights_best = portfolio_optimization.max_sharpe_ratio().x
    # portfolio_optimization.min_VaR()
    performance_train = PortfolioPerformance(portfolio_train, weights_best, 'perf_train')

    portfolio_test = Portfolio(df_test)
    performance_test = PortfolioPerformance(portfolio_test, weights_best, 'perf_test')
    perf.append([performance_train.__dict__, performance_test.__dict__])
    tt['train'].append(portfolio_train.returns * weights_best)
    tt['test'].append(portfolio_test.returns * weights_best)
CovarianceShrinkage(portfolio_train.returns.dropna().astype('float32')).oracle_approximating()
portfolio_train.cov
# qs.plots.snapshot(pd.concat(tt['train']).dropna().sum(axis=1), title="portfolio_train")
# sum(np.unique(pd.concat(tt['test']).dropna().sum(axis=1),return_counts=True)[1]!=1)
qs.plots.snapshot(pd.concat(tt['test']).dropna().sum(axis=1), title="portfolio_test")
pd.DataFrame([i for j in perf for i in j]).drop(list(df.columns), axis=1).groupby('tag').boxplot()
plt.show()
pd.DataFrame([i for j in perf for i in j]).drop(list(df.columns), axis=1).groupby('tag').plot()
pd.DataFrame([i for j in perf for i in j]).drop(list(df.columns), axis=1).groupby('tag').mean()
# tests.append(performance_test.__dict__)
# pd.DataFrame
# upon_portf.prepare_df_of_best(df_simmulates,df_optimized)
#
#
# import matplotlib.pyplot as plt
# max_sharpe_port = df_simmulates.iloc[df_simmulates['sharpe'].idxmax()]
# #locate positon of portfolio with minimum standard deviation
# min_vol_port = df_simmulates.iloc[df_simmulates['stdev'].idxmin()]
# max_diverse_port = df_simmulates.iloc[df_simmulates['diversif'].idxmin()]
# min_VaR_port = df_simmulates.iloc[df_simmulates['VaR'].idxmin()]
# #create scatter plot coloured by Sharpe Ratio
#
# #plot red star to highlight position of portfolio with highest Sharpe Ratio
# plt.scatter(max_sharpe_port[1],max_sharpe_port[0],marker=(5,1,0),color='r',s=200)
# #plot green star to highlight position of minimum variance portfolio
# plt.scatter(min_vol_port[1],min_vol_port[0],marker=(5,1,0),color='g',s=100)
# plt.scatter(max_diverse_port[1],max_diverse_port[0],marker=(5,1,0),color='b',s=100)
# plt.scatter(min_VaR_port[1],min_VaR_port[0],marker=(5,1,0),color='y',s=100)
#
# for idx,row in df_optimized.iterrows():
#     plt.scatter(row.stdev,row.ret,marker=(5,1,0),color='g',s=100)
#     plt.annotate(row.tag,(row.stdev*1.001,row.ret))
#
# plt.show()

# efficient_risk
# LinearRegression
OptPortfolio
