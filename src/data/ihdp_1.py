# Source: [VCNet] https://github.com/lushleaf/varying-coefficient-net-with-functional-tr

# LOAD MODULES
# Standard library
from typing import Union, Optional, Dict, Any

# Proprietary
from src.data.utils import train_val_test_ids, sample_rows, ContinuousData

# Third party
from sklearn.utils import Bunch
import numpy as np
import pandas as pd


def get_outcome(
        matrix: np.ndarray,
        doses: np.ndarray,
        treatments: np.ndarray,
        cate_idx1=None,
        cate_mean1=None,
        alpha: float = 5.0,
        factor1: float = 1.0,
        factor2: float = 1.0,
        factor3: float = 1.0,
        noise_outcome: float = 0.5,
) -> np.ndarray:
    """
    Calculates the outcome variable for a given dataset and doses.

    The function calculates the outcome variable based on a complex formula involving the doses and several columns of the dataset. The formula involves trigonometric functions, exponential functions, and hyperbolic functions.

    Parameters:
    matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
    doses (np.ndarray): A 1D numpy array representing the doses for each observation.
    treatments (np.ndarray): A 1D numpy array representing the treatments for each observation.
    cate_idx1 (np.ndarray, optional): Indices for the first set of categorical variables. Defaults to None.
    cate_mean1 (float, optional): Mean of the first set of categorical variables. Defaults to None.
    alpha (float, optional): Parameter for the outcome calculation. Defaults to 5.0.
    factor1 (float, optional): First factor for the outcome calculation. Defaults to 1.0.
    factor2 (float, optional): Second factor for the outcome calculation. Defaults to 1.0.
    factor3 (float, optional): Third factor for the outcome calculation. Defaults to 1.0.
    noise_outcome (float, optional): Noise to be added to the outcome. Defaults to 0.5.

    Returns:
    np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
    """
    # Only take continuous variables
    x1 = matrix[:, 0]
    x2 = matrix[:, 1]
    x3 = matrix[:, 2]
    x4 = matrix[:, 4]
    x5 = matrix[:, 5]

    x23 = matrix[:, 23]

    # Calc cate_mean
    cate_idx1 = np.array([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    cate_idx2 = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    cate_mean1 = np.mean(np.mean(matrix[:, cate_idx1], axis=1))
    cate_mean2 = np.mean(np.mean(matrix[:, cate_idx2], axis=1))

    # Calc outcome
    y = 1. / (1.2 - doses) * np.sin(doses * 3. * 3.14159) * (
            factor1 * np.tanh((np.sum(matrix[:, cate_idx1], axis=1) / 10. - cate_mean1) * alpha) + \
            factor2 * np.exp(0.2 * (x1 - x5)) / (0.1 + np.minimum(x2, x3, x4))
            + factor3 * doses * x23
    )

    #scale to [0,1]
    y = (y - (-15)) / (30 - (-15))

    return y



# DATA LOADING FUNCTION
def load_data(
        data_path: str = "data/ihdp_s_1/IHDP-S-1.csv",
        bias: float = 1.0,
        sample_size: Optional[Union[int, float]] = None,
        train_share: float = 0.7,
        val_share: float = 0.1,
        seed: int = 5,
        alpha: float = 5.0,
        factor1: float = 1.0,
        factor2: float = 1.0,
        factor3: float = 0.0,
        noise_dose: float = 0.25, #0.5,
        noise_outcome: float = 0.25,
        rm_confounding: bool = False,
) -> Bunch:
    """
    Loads and preprocesses data from a CSV file.

    The function loads the data, optionally samples a subset of rows, normalizes the data, calculates an outcome variable, adds noise to the outcome, and splits the data into training, validation, and test sets.

    Parameters:
    data_path (str): Path to the CSV file to load. Defaults to "data/ihdp_s_1/IHDP-S-1.csv".
    bias (float): The proportion of the doses that are generated from confounded formula. Rest is randomly sampled from [0,1]. Defaults to 1.
    sample_size (Optional[Union[int, float]], optional): Number or proportion of rows to sample from the data. If None, all rows are used. Defaults to None.
    train_share (float, optional): Proportion of the data to include in the training set. Defaults to 0.7.
    val_share (float, optional): Proportion of the data to include in the validation set. Defaults to 0.1.
    seed (int, optional): Seed for the random number generator. Defaults to 5.
    alpha (float, optional): Parameter for the outcome calculation. Defaults to 5.0.
    factor1 (float, optional): First factor for the outcome calculation. Defaults to 1.0.
    factor2 (float, optional): Second factor for the outcome calculation. Defaults to 1.0.
    factor3 (float, optional): Third factor for the outcome calculation. Defaults to 0.0.
    noise_dose (float, optional): Noise to be added to the doses. Defaults to 0.25.
    noise_outcome (float, optional): Noise to be added to the outcome. Defaults to 0.25.
    rm_confounding (bool, optional): Whether to remove confounding. Defaults to False.

    Returns:
    Bunch: A Bunch object containing the preprocessed data, the outcome variable, the ground truth function, and the indices for the training, validation, and test sets.
    """
    # Set seed
    np.random.seed(seed)

    # Load raw data
    matrix = pd.read_csv(data_path, sep=",", header=0)

    # To numpy
    matrix = matrix.to_numpy()

    # Sample rows if sample_size is specified
    if sample_size is not None:
        matrix = sample_rows(matrix, sample_size, seed=seed)

    # Drop columns
    # NOTE: Only reading 25 columns of dataframe as in original script. 3 more available
    matrix = matrix[:, 2:27]

    # Save info
    num_rows = matrix.shape[0]
    num_cols = matrix.shape[1]

    # Normalize
    for col in range(num_cols):
        minval = min(matrix[:, col]) * 1.
        maxval = max(matrix[:, col]) * 1.
        matrix[:, col] = (1. * (matrix[:, col] - minval)) / maxval

    # Save continuous variables
    x1 = matrix[:, 0]
    x2 = matrix[:, 1]
    x3 = matrix[:, 2]
    x4 = matrix[:, 4]
    x5 = matrix[:, 5]

    # Calc cate_mean
    cate_idx1 = np.array([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    cate_idx2 = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    cate_mean1 = np.mean(np.mean(matrix[:, cate_idx1], axis=1))
    cate_mean2 = np.mean(np.mean(matrix[:, cate_idx1], axis=1)) # Taken from original github, kept unchanged. Should be cate_idx2?

    doses = x1 / (1. + x2) + \
            np.maximum(np.maximum(x3, x4), x5) / (0.2 + np.minimum(np.minimum(x3, x4), x5)) + \
            np.tanh((np.sum(matrix[:, cate_idx2], axis=1) / 10. - cate_mean2) * alpha) \
            - 2.

    # add noise to doses: t = t + N(0, 0.25)
    doses = doses + np.random.randn(matrix.shape[0]) * noise_dose

    # Transform according to link function
    doses = 1. / (1. + np.exp(-2. * doses))

    # Generate a dummy array for treatment
    t = np.zeros(num_rows).astype(int)

    # Sample outcomes
    y = get_outcome(matrix,
                    doses,
                    treatments=t,
                    cate_idx1=cate_idx1,
                    cate_mean1=cate_mean1,
                    alpha=alpha,
                    factor1=factor1,
                    factor2=factor2,
                    factor3=factor3,
                    noise_outcome=noise_outcome)

    # add noise to outcome: y = y + N(0, 0.25) [scaled by 45]
    y_noise = (np.random.randn(matrix.shape[0]) * noise_outcome) / (30 - (-15))
    y = y + y_noise

    # Get train/val/test ids
    train_ids, val_ids, test_ids = train_val_test_ids(num_rows,
                                                      train_share=train_share,
                                                      val_share=val_share, )

    # Generate benefits (cost-sensitivity / profit-driven analytics); this is stored in the last feature; not taken into account for dose or outcome calculation
    benefits = np.random.lognormal(0, 1, num_rows)/np.sqrt(2.718281)

    # Generate bunch
    data = ContinuousData(
        x=matrix,
        t=t,
        d=doses,
        b=benefits,
        y=y,
        ground_truth=get_outcome,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    return data