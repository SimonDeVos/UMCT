from typing import Dict, Any

import numpy as np


def t_x_y(
        doses: np.ndarray,
        matrix: np.ndarray,
        **kwargs: Dict[str, Any]
) -> np.ndarray:


    """
    Calculates the outcome variable for a given dataset and doses.

    The function calculates the outcome variable based on a complex formula involving the doses and several columns of the dataset. The formula involves trigonometric functions, exponential functions, and hyperbolic functions.

    Parameters:
    matrix (np.ndarray): The dataset, represented as a 2D numpy array. Each row represents an observation, and each column represents a variable.
    doses (np.ndarray): A 1D numpy array representing the doses for each observation.
    **kwargs: Dummy to improve compatibility with other functions.

    Returns:
    np.ndarray: A 1D numpy array representing the calculated outcome for each observation.
    """
    # Only take continuous variables
    x1 = matrix[:, 0]
    x2 = matrix[:, 1]
    x3 = matrix[:, 2]
    x4 = matrix[:, 4]
    x5 = matrix[:, 5]

    # todo: hard-coded values, need to be passed as parameters
    # Calc cate_mean
    cate_idx1 = np.array([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    cate_idx2 = np.array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    cate_mean1 = np.mean(np.mean(matrix[:, cate_idx1], axis=1))
    cate_mean2 = np.mean(np.mean(matrix[:, cate_idx1], axis=1))

    # Calc outcome
    y = 1. / (1.2 - doses) * np.sin(doses * 3. * 3.14159) * (
            factor1 * np.tanh((np.sum(matrix[:, cate_idx1], axis=1) / 10. - cate_mean1) * alpha) + \
            factor2 * np.exp(0.2 * (x1 - x5)) / (0.1 + np.minimum(x2, x3, x4))
    )

    return y


def matrix_lin_space(nr_rows, nr_cols):
    # Create an empty ndarray with the specified shape

    arr = np.zeros((nr_rows, nr_cols))
    arr[:, :] = np.linspace(0, 1, nr_cols)

    return arr



