import hashlib
import os
import re

import numpy as np
import scipy.stats as st
from pandas_datareader import data
from scipy.optimize import minimize


def get_last_raw(subdf):
    '''
    Aggregation based on last raw of df. Used in conjuction with panda's apply method on aggregated by some interval (week,month,ect.) data.
    :param subdf:
    :return: pd.DataFrame
    '''
    # print(df)
    subdf_cleaned = subdf.dropna()
    if subdf_cleaned.shape[0] != 0:
        return subdf_cleaned.iloc[-1, :]
    else:
        pass


def get_mean(subdf):
    '''
    Aggregation based on mean . Used in conjuction with panda's apply method on aggregated by some interval (week,month,ect.) data.
    :param subdf: pd.DataFrame
    :return: pd.DataFrame
    '''
    subdf_cleaned = subdf.dropna()
    if subdf_cleaned.shape[0] != 0:
        return subdf_cleaned.mean(axis=0)
    else:
        pass


def negative_diversification_coefficient(x, sigmas, V):
    # w=np.asarray([1,2,3,4,5,6])
    return -np.dot(x, sigmas) / np.sqrt(np.matmul(np.matmul(x.reshape(1, -1), V), x.reshape(-1, 1)))


def diversify_portfolio(df_returns, bootstrap=False, **kwargs):
    if bootstrap:
        stds, means = boostraped_diversify_portfolio(df_returns, **kwargs)
        means = means / sum(means)
        return stds, means
    else:
        return _diversify_portfolio(df_returns, **kwargs)


def boostraped_diversify_portfolio(df_returns, n_iters=100, **kwargs):
    weights_accumulated = []
    df_returns = df_returns.dropna()
    for i in range(n_iters):
        sampled_working_data = df_returns.sample(df_returns.shape[0], replace=True)
        _, weights = _diversify_portfolio(sampled_working_data, **kwargs)
        weights_accumulated.append(weights)
    weights_accumulated = pd.concat(weights_accumulated)
    z = (weights_accumulated.mean() / weights_accumulated.std()).dropna()
    p_value = st.norm.cdf(z)
    ratio_of_adequate_weights = sum(p_value > 0.95) / len(p_value)
    print("ratio_of_adequate_weights", ratio_of_adequate_weights)
    return np.std(weights_accumulated, axis=0), np.mean(weights_accumulated, axis=0)


def __create_specific_constraint(list_of_params, columns):
    # list_of_params=['AAPL', 'ineq', [0.1,0.8]]
    print(columns, list_of_params)
    idx_of_variable = np.argwhere(np.asarray(columns) == list_of_params[0])[0][0]
    if len(list_of_params[1]) == 1:
        assert isinstance(list_of_params[1][0], float)
        res = {"type": "eq",
               "fun": lambda x: np.asarray([x[idx_of_variable] - list_of_params[1][0]])}
    elif len(list_of_params[1]) == 2:
        assert list_of_params[1][1] > list_of_params[1][0], "Wrong list_of_params"
        res = {"type": "ineq",
               "fun": lambda x: np.asarray([x[idx_of_variable] - list_of_params[1][0],
                                            list_of_params[1][1] - x[idx_of_variable]])}
    else:
        raise NotImplemented
    return res


def __create_general_constraint(n_variables, magnitude):
    each_weight = 1 / n_variables
    assert n_variables > magnitude
    min_value = each_weight / magnitude
    max_value = each_weight * magnitude
    specific_constraint_low = {"type": "ineq",
                               "fun": lambda x: np.asarray(x - min_value)}

    specific_constraint_high = {"type": "ineq",
                                "fun": lambda x: np.asarray(max_value - x)}
    return [specific_constraint_low, specific_constraint_high]


def deal_with_add_constraints(columns, specific_constraints=None, general_constraints=None):
    '''
    specific_constraints
    :param columns: list of columns
    :param specific_constraints:  None or list of format ['AAPL',  [0.1,0.8]]
    :param general_constraints: int or None , divider  , 1/(len(col)*divivder)<weights<divivder/(len(col))
    :return: list of constraints
    '''
    # specific_constraints = kwargs.get("specific_constraints")
    # general_constraints = kwargs.get("general_constraints")
    # general_constraints=3
    list_of_specific_constraints = []
    list_of_general_constraints = []

    # specific_constraints = [['AAPL', 'eq', [0.1]], ["GRA", "ineq", [0.1, 0.3]]]
    if specific_constraints:
        for constaint in specific_constraints:
            constraint_dict = __create_specific_constraint(constaint, columns)
            list_of_specific_constraints.append(constraint_dict)
    if general_constraints:
        magnitued_of_range = general_constraints
        n_variables = len(columns)
        list_of_general_constraints = __create_general_constraint(n_variables, magnitued_of_range)
    list_of_additional_constraints = list_of_specific_constraints + list_of_general_constraints
    return list_of_additional_constraints


def _diversify_portfolio(df_returns, **kwargs):
    if isinstance(df_returns, np.ndarray):
        V = np.cov(df_returns)
    else:
        V = np.cov(df_returns.dropna().T)

    list_of_additional_constraints = deal_with_add_constraints(df_returns.columns, **kwargs)

    sigmas = np.sqrt(np.diag(V))
    n_assets = sigmas.shape[0]
    eq_cons = {'type': 'eq',
               'fun': lambda x: np.sum(x) - 1}  ## sum(x)=1
    ineq_cons1 = {'type': 'ineq',
                  'fun': lambda x: np.asarray(1 - x)}  # 1-x>=0 => x<=1
    ineq_cons2 = {'type': 'ineq',
                  'fun': lambda x: np.asarray(x)}  # x>0
    x0 = np.asarray([0 for i in range(n_assets - 1)] + [1])
    constraints = [eq_cons, ineq_cons1, ineq_cons2] + list_of_additional_constraints
    res = minimize(negative_diversification_coefficient, x0=x0, args=(sigmas, V),
                   constraints=constraints, options={'disp': False})
    # print(res)
    weights = pd.DataFrame([res.x], columns=list(df_returns.columns)).round(3)
    # weights=res.x
    # import scipy

    # scipy.optimize.show_options(minimize,method="SLSQP")
    # print(weights.round(3))

    return res, weights


from numpy.lib.stride_tricks import as_strided as stride
import pandas as pd


def roll(df, w, **kwargs):
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)


def smartDataReader(force_reload=False, dfs_folder='C:/PycharmProjects/finance/data/dataloader_data', **kwargs):
    '''
    Loads data using dataloader only if the data was not loaded before and user has not forced to reload data
    :param force_reload: boolean, whether or not to reload data if it was already loaded
    :param dfs_folder: folder where to save and from where to load dataframes
    :param kwargs: arguments used for loading (e.g. tickers' names , interval,loading sources, ect.)
    :return: loaded df
    '''
    kwargs = {k: v for k, v in sorted(kwargs.items(), key=lambda item: item[0])}
    filename = "_".join([str(i[1]) for i in kwargs.items()])
    filename = re.sub(r'[^\w\d\.\/]', '', filename)
    print(filename)
    filename = hashlib.sha256(f"{filename}".encode('utf-8')).hexdigest()
    print(filename)
    path_to_csv = f"{dfs_folder}/{filename}.csv"
    path_to_pickle = f"{dfs_folder}/{filename}.pickle"
    if len(path_to_pickle) >= 258:
        path_to_pickle = path_to_pickle[:240] + path_to_pickle[-10:]
        path_to_csv = path_to_csv[:240] + path_to_csv[-10:]

    if os.path.exists(path_to_pickle) and force_reload == False:
        print("Loading data from disk")
        df_from_disk = pd.read_pickle(path_to_pickle)
        df = df_from_disk.copy()
    else:
        print("Loading data from server")

        df_from_server = data.DataReader(**kwargs)

        df_from_server.to_pickle(path_to_pickle)
        df_from_server.to_csv(path_to_csv)

        df_from_disk = pd.read_pickle(path_to_pickle)
        assert df_from_server.equals(df_from_disk)
        df = df_from_server.copy()

    return df


def prepare_FX():
    '''
    Prepares FinEx funds for analysis
    :return:
    '''
    import os
    base_fx_folder = 'c:/PycharmProjects/finance/fx'
    csv_files = os.listdir(base_fx_folder)

    df_lists = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(base_fx_folder, file))
        df = df[['<DATE>', "<CLOSE>"]]
        df.columns = ['date', file.split("_")[0]]
        df['date'] = pd.to_datetime(df['date'].astype("str"), format='%Y%m%d')
        df = df.set_index('date')
        df_lists.append(df)

    prices = pd.concat(df_lists, axis=1)
    return prices


# smartDataReader(c="as",a=1,b=2,d=4,e=5)

def autodetect_frequency(df):
    '''
    :param df: pandas dataframe with index being a datetime object
    :return: frequency
    '''
    delta_time = int(abs(np.mean(
        (pd.DataFrame(df.index[1:]).shift(1).values - pd.DataFrame(df.index[1:])).astype('timedelta64[D]'))))
    print(delta_time)
    if delta_time < 3:
        frequency = 252
    elif delta_time == 7:
        frequency = 52

    elif delta_time > 20 and delta_time < 31:
        frequency = 12
    else:
        raise NotImplementedError
    return frequency
