"""Auxiliary functions to generate Locally Differentially Private data."""

import numpy as np
import pandas as pd


def add_noise_laplace_mechanism(data_df: pd.DataFrame, epsilon: float, 
                                sensitivity: float, seed: int) -> pd.DataFrame:
    """This function takes a (numerical) dataframe as input and yields a new 
    dataframe as output where each element arises by adding noise to the 
    corresponding element from the original dataframe. The Laplace mechanism, 
    with scale = sensitivity / epsilon > 0, is used to generate the noise. 
    A seed can be set for reproducubility."""  
    output_df = data_df.copy()
    scale = sensitivity / epsilon
    noise = np.random.default_rng(seed).laplace(0, scale, output_df.shape)
    output_df = output_df + noise
    # for testing purposes: variance of Laplace(b) is 2b^2
    return output_df


