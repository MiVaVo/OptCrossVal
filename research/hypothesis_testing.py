import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from fincrossval.utils import diversify_portfolio


def Ho_mean_expected_returns_are_equal_for_diversivied_and_random_weights(df):
    statistics = []
    expected_returns = []
    expected_returns_random_weights = []
    if isinstance(df, pd.DataFrame):
        different_type_of_returns = {"default": df}
    else:
        different_type_of_returns = df.copy()
    for key, df in different_type_of_returns.items():
        df = df.dropna()
        for j in tqdm(range(100)):
            train_df, val_df = train_test_split(df)
            for i in range(50):
                sampled_working_data = train_df.sample(700, replace=True)
                sampled_validation_data = val_df.sample(700, replace=True)

                res, weights = diversify_portfolio(sampled_working_data)
                statistics.append(weights)
                expected_return = (np.mean(sampled_validation_data, axis=0) * weights).sum(axis=1)
                expected_returns.append(expected_return)

                random_weights = np.random.dirichlet(np.ones(len(weights)), size=1)[0]
                expected_returns_random_weight = (np.mean(sampled_validation_data, axis=0) * random_weights).sum(axis=1)
                expected_returns_random_weights.append(expected_returns_random_weight)

    # Test hypothesis of H0: mu_opt=mu_random

    mean_expected_return_div = np.mean(expected_returns)
    mean_expected_return_rand = np.mean(expected_returns_random_weights)
    std_expected_return_div = np.std(expected_returns)
    std_expected_return_rand = np.std(expected_returns_random_weights)
    z, p_value = get_z_score_for_diff_in_mean(mean_expected_return_div,
                                              mean_expected_return_rand,
                                              std_expected_return_div,
                                              std_expected_return_rand,
                                              n1=len(expected_returns),
                                              n2=len(expected_returns_random_weights))
    print(z)
    print("P-valie=", round(p_value, 4))


def get_z_score_for_diff_in_mean(mean1, mean2, std1, std2, n1, n2):
    z = (mean1 - mean2) / np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2)
    p_value = st.norm.cdf(z)
    return z, p_value


def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s
