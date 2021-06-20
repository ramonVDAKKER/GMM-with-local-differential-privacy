"""Auxiliary functions to simulate datasets."""

import pandas as pd
import numpy as np

def univariate_linear_regression(mu_X, variance_X, alpha, beta, desired_R_squared, n):
    """This function generated a dataframe with columns Y and X where X consists of i.i.d. draws from a N(mu_X, variance_X)
    distribution and Y = alpha + beta * X + epsilon. Here epsilon consists of i.i.d. draws from a Logistic distribution with
    mean zero and the variance is such that the R-squared of the regression is equal to desired_R_squared (in the population)."""
    # initialize dataframe:
    data_df = pd.DataFrame(columns=["Y", "X"])
    # simulate feature / regressor from N(mu_X, variance_X)
    data_df["X"] = np.random.normal(mu_X, np.sqrt(variance_X), n)
    # calculate variance epsilon and simulate from Logistic distribution:
    desired_variance_epsilon = beta ** 2 * variance_X / desired_R_squared - beta ** 2 * variance_X
    scale = np.sqrt(desired_variance_epsilon * 3) / np.pi
    data_df["epsilon"] = np.random.logistic(0, scale, n)
    data_df["Y"] = alpha + beta * data_df["X"] + data_df["epsilon"]
    data_df.drop(columns="epsilon", inplace=True)
    return data_df
