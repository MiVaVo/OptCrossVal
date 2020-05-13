from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

from configs import COMPANIES_LIST, start_date, end_date, OIL_GAZ, CLOTHING, ENTERAINMENT, \
    AIRLINES
from fincrossval.custom_ttt import TimeSeriesSplitLargeTest
from fincrossval.port_custom import Portfolio, PortfolioOptimization
from fincrossval.utils import smartDataReader

df = smartDataReader(name=COMPANIES_LIST, data_source='yahoo', start=start_date, end=end_date, force_reload=False,
                     retry_count=30)


# portfolio_movements=(df_pct_change*optim_portfolio.x).sum(axis=1)
# df_pct_change['portfolio']=portfolio_movements
def get_X_y(label_col, df_pct_change, n_lags=2, quantile=0.1, quantile_values=None, which_quintile="upper"):
    if isinstance(label_col, int):
        label_col = df_pct_change.columns[label_col]

    features_col = [i for i in df_pct_change.columns if i != label_col]
    if which_quintile == "lower":
        if quantile_values is None:
            quantile_values = df_pct_change.quantile(quantile)
        df_pct_change_binary = (df_pct_change < quantile_values).astype("int")
    elif which_quintile == "upper":
        if quantile < 0.5:
            quantile = 1 - quantile
        if quantile_values is None:
            quantile_values = df_pct_change.quantile(quantile)
        df_pct_change_binary = (df_pct_change > quantile_values).astype("int")
    else:
        raise NotImplementedError

    y = df_pct_change_binary[label_col]
    # X_binary=df_pct_change_binary[features_col]
    X_numerical = df_pct_change[features_col]
    X = pd.concat([X_numerical], axis=1)
    laged_features = []
    laged_labels = []
    for i in range(1, n_lags + 1):
        laged_features.append(X.shift(i))
        laged_labels.append(df_pct_change[label_col].shift(i))
    X = pd.concat(laged_features + laged_labels, axis=1).dropna()
    y = y[X.index]
    return X, y, quantile_values


tscv = TimeSeriesSplitLargeTest(n_splits=10)
# X,y=get_X_y("portfolio",df_pct_change,5,quantile=0.01)

# print("Number of black days",np.sum(y))
prob_tests = []
prob_trains = []
quantile = 0.05
COMPANIES_LIST_INTEREST = OIL_GAZ + CLOTHING + ENTERAINMENT + AIRLINES
for train_index, test_index in tqdm(tscv.split(df)):
    # print("TRAIN:", train_index, "TEST:", test_index)
    df_train = df.loc[train_index]
    df_test = df.loc[test_index]

    tickers = df_train['Adj Close'].columns[~df_train['Adj Close'][:3].isna().any()]
    df_train = df_train['Adj Close'][tickers].dropna()
    df_test = df_test['Adj Close'][tickers].dropna()

    # Optimize portfolio
    df_train_interest = df_train[COMPANIES_LIST_INTEREST]
    portfolio_train_interest = Portfolio(df_train_interest)
    optim_portfolio_weights = PortfolioOptimization(portfolio_train_interest).max_sharpe_ratio().x
    labels_train = pd.DataFrame((portfolio_train_interest.returns * optim_portfolio_weights).mean(axis=1),
                                columns=['portfolio'])

    portfolio_train_model = Portfolio(df_train)
    Xy_df_train = pd.concat([labels_train, portfolio_train_model.returns], axis=1)
    X_train, y_train, quantile_values = get_X_y("portfolio", Xy_df_train, n_lags=20, quantile=quantile,
                                                which_quintile='lower')
    print(quantile_values['portfolio'])
    # Prepare independent test set
    df_test_interest = df_test[COMPANIES_LIST_INTEREST]

    portfolio_test_interest = Portfolio(df_test_interest)
    labels_test = pd.DataFrame((portfolio_test_interest.returns * optim_portfolio_weights).mean(axis=1),
                               columns=['portfolio'])

    portfolio_test_model = Portfolio(df_test)
    Xy_df_test = pd.concat([labels_test, portfolio_test_model.returns], axis=1)
    X_test, y_test, _ = get_X_y("portfolio", Xy_df_test, n_lags=20, quantile_values=quantile_values,
                                which_quintile='lower')

    # df_train = (1 + df_train.pct_change()).apply(np.log).dropna()

    if len(np.unique(y_test)) == 1 or len(np.unique(y_train)) == 1:
        print("\n Skip")
        continue
    # TRAIN MODEL
    # clf = LogisticRegression(penalty='l1', solver='liblinear',random_state=0,C=0.5)
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=1000, n_jobs=8)
    clf.fit(X_train, y_train)
    # train_rocauc=roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
    # test_rocauc=roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    preds_test = clf.predict_proba(X_test)[:, 1]
    prob_tests.append(pd.DataFrame({"y_true": y_test.values,
                                    "y_score": preds_test}, index=X_test.index))
    preds_train = clf.predict_proba(X_train)[:, 1]
    prob_trains.append(pd.DataFrame({"y_true": y_train.values,
                                     "y_score": preds_train}, index=X_train.index))

porb_tests_df = pd.concat(prob_tests, axis=0)
print(roc_auc_score(**porb_tests_df))
porb_tests_df[porb_tests_df.index > datetime.strptime('2010-01-01', '%Y-%m-%d')][-80:].plot()
import matplotlib.pyplot as plt

plt.show()

for th in np.linspace(0, 0.1, 100):
    preds = np.asarray(prob_tests[-1]['y_score'] > th).astype("int")
    cmat = confusion_matrix(prob_tests[-1]['y_true'], preds)
    tn, fp, fn, tp = cmat.ravel()
    # FOR=tn/(fn+tn)
    print(cmat, th)
####Post prediction analysis
porb_tests_df['y_score']
import scikitplot as skplt
import matplotlib.pyplot as plt

skplt.metrics.plot_roc_curve(y_true, y_probas)
skplt.metrics.plot_precision_recall_curve

y_probas = np.asarray([1 - porb_tests_df['y_score'], porb_tests_df['y_score']]).T
skplt.metrics.plot_roc(y_true=porb_tests_df['y_true'].values,
                       y_probas=y_probas)

skplt.metrics.plot_precision_recall(y_true=porb_tests_df['y_true'].values,
                                    y_probas=y_probas)
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(porb_tests_df['y_true'].values,
                                                       y_probas[:, 1])
plt.show()

# print(roc_auc_score(**pd.concat(prob_trains,axis=0)))

# roc_auc_score()
#     # prob_tests.append(pd.DataFrame.from_records([preds_test,y_test.values],columns=test_index).T)
#     # prob_tests.append(pd.DataFrame([preds_train,y_train.values],index=test_index))
#
#     # print(f'train rocauc {train_rocauc :.3f} test {test_rocauc :.3f}')
#
# param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)
