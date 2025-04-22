# LOAD MODULES
# Standard library
from typing import Union, Tuple, Callable

# Third party
from sklearn.model_selection import train_test_split
import numpy as np
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt

# FUNCTIONS

import yaml

def load_config(config_path: str) -> dict:
    """
    Load settings from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Settings.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train_val_test_ids(
        num_obs: int,
        train_share: float = 0.7,
        val_share: float = 0.1,
        seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the range of observations into training, validation, and test sets.

    Parameters:
    num_obs (int): The total number of observations.
    train_share (float, optional): The proportion of the data to include in the training set. Defaults to 0.7.
    val_share (float, optional): The proportion of the data to include in the validation set. Defaults to 0.1.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    tuple: A tuple containing three numpy arrays. The first array contains the indices for the training set, the second contains the indices for the validation set, and the third contains the indices for the test set.
    """
    ids = np.array(range(num_obs))
    num_train = int(train_share * num_obs)
    num_val = int(val_share * num_obs)

    rest, ids_train = train_test_split(ids, test_size=num_train, random_state=seed)
    ids_test, ids_val = train_test_split(rest, test_size=num_val, random_state=seed)

    return ids_train, ids_val, ids_test


def sample_rows(
        arr: np.ndarray,
        num_rows: Union[int, float],
        replace: bool = False,
        seed: int = 42
) -> np.ndarray:
    """
    Samples a specified number of rows from a numpy array.

    Parameters:
    arr (np.ndarray): The input numpy array.
    num_rows (Union[int, float]): The number of rows to sample or the percentage of rows to sample.
    replace (bool, optional): Whether to allow sampling of the same row more than once. Defaults to False.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    np.ndarray: A numpy array containing the sampled rows.
    """
    np.random.seed(seed)

    if isinstance(num_rows, float):
        assert 0.0 <= num_rows <= 1.0, "Percentage must be between 0.0 and 1.0"
        num_rows = int(arr.shape[0] * num_rows)

    row_indices = np.random.choice(arr.shape[0], size=num_rows, replace=replace)

    return arr[row_indices]


def sample_columns(
        arr: np.ndarray,
        num_cols: Union[int, float],
        replace: bool = False,
        seed: int = 42
) -> np.ndarray:
    """
    Samples a specified number of columns from a numpy array.

    Parameters:
    arr (np.ndarray): The input numpy array.
    num_cols (Union[int, float]): The number of columns to sample or the percentage of columns to sample.
    replace (bool, optional): Whether to allow sampling of the same column more than once. Defaults to False.
    seed (int, optional): The seed for the random number generator. Defaults to 42.

    Returns:
    np.ndarray: A numpy array containing the sampled columns.
    """
    np.random.seed(seed)

    if isinstance(num_cols, float):
        assert 0.0 <= num_cols <= 1.0, "Percentage must be between 0.0 and 1.0"
        num_cols = int(arr.shape[1] * num_cols)

    col_indices = np.random.choice(arr.shape[1], size=num_cols, replace=replace)

    return arr[:, col_indices]


def normalize(
        matrix: np.ndarray
) -> np.ndarray:
    """
    Normalizes a 2D numpy array column-wise.

    Parameters:
    matrix (np.ndarray): The 2D numpy array to normalize.

    Returns:
    np.ndarray: The normalized 2D numpy array.
    """
    num_cols = matrix.shape[1]

    for idx in range(num_cols):
        max_value = max(matrix[:, idx])
        matrix[:, idx] = matrix[:, idx] / max_value

    return matrix


def softmax(
        x: np.ndarray,
) -> np.ndarray:
    """
    Computes the softmax of a numpy array.

    Parameters:
    x (np.ndarray): A numpy array for which to compute the softmax.

    Returns:
    np.ndarray: A numpy array representing the softmax of the input.
    """
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def get_beta(
        alpha: float, mode: float
) -> float:
    """
    Computes the beta value based on the provided alpha and mode.

    Parameters:
    alpha (float): The alpha value used in the calculation.
    mode (float): The mode value used in the calculation.

    Returns:
    float: The calculated beta value.
    """
    if (mode <= 0.001 or mode >= 1.0):
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(mode) + (2.0 - alpha)

    return beta


@dataclass
class ContinuousData:
    """
    Creates a dataclass for continuous data.
    """
    x: np.ndarray
    t: np.ndarray
    d: np.ndarray
    y: np.ndarray
    ground_truth: Callable
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    info: str = "No info provided for this dataset."

    # ids
    @property
    def train_val_ids(self):
        return np.concatenate((self.train_ids, self.val_ids))

    # x
    @property
    def x_train(self):
        return self.x[self.train_ids]

    @property
    def x_val(self):
        return self.x[self.val_ids]

    @property
    def x_train_val(self):
        return self.x[self.train_val_ids]

    @property
    def x_test(self):
        return self.x[self.test_ids]

    # t
    @property
    def t_train(self):
        return self.t[self.train_ids]

    @property
    def t_val(self):
        return self.t[self.val_ids]

    @property
    def t_train_val(self):
        return self.t[self.train_val_ids]

    @property
    def t_test(self):
        return self.t[self.test_ids]

    # d
    @property
    def d_train(self):
        return self.d[self.train_ids]

    @property
    def d_val(self):
        return self.d[self.val_ids]

    @property
    def d_train_val(self):
        return self.d[self.train_val_ids]

    @property
    def d_test(self):
        return self.d[self.test_ids]

    # y
    @property
    def y_train(self):
        return self.y[self.train_ids]

    @property
    def y_val(self):
        return self.y[self.val_ids]

    @property
    def y_train_val(self):
        return self.y[self.train_val_ids]

    @property
    def y_test(self):
        return self.y[self.test_ids]

    # xd
    @property
    def xd(self):
        return np.column_stack((self.x, self.d))

    @property
    def xd_train(self):
        return np.column_stack((self.x, self.d))[self.train_ids]

    @property
    def xd_val(self):
        return np.column_stack((self.x, self.d))[self.val_ids]

    @property
    def xd_train_val(self):
        return np.column_stack((self.x, self.d))[self.val_ids][self.train_val_ids]

    @property
    def xd_test(self):
        return np.column_stack((self.x, self.d))[self.test_ids]

    # xt
    @property
    def xt(self):
        return np.column_stack((self.x, self.t))

    @property
    def xt_train(self):
        return self.xt[self.train_ids]

    @property
    def xt_val(self):
        return self.xt[self.val_ids]

    @property
    def xt_train_val(self):
        return self.xt[self.train_val_ids]

    @property
    def xt_test(self):
        return self.xt[self.test_ids]

    # xtd
    @property
    def xtd(self):
        return np.column_stack((self.x, self.t, self.d))

    @property
    def xtd_train(self):
        return np.column_stack((self.x, self.t, self.d))[self.train_ids]

    @property
    def xtd_val(self):
        return np.column_stack((self.x, self.t, self.d))[self.val_ids]

    @property
    def xtd_train_val(self):
        return np.column_stack((self.x, self.t, self.d))[self.train_val_ids]

    @property
    def xtd_test(self):
        return np.column_stack((self.x, self.t, self.d))[self.test_ids]


    def plot_dose_density(self):
        sns.kdeplot(self.d, kernel='gau')
        plt.xlabel('Dose')
        plt.ylabel('Density')
        plt.title('Density Estimate of Dose')
        plt.xlim(0, 1)
        plt.show()

@dataclass
class BinaryData:
    """
    Creates a dataclass for continuous data.
    """
    x: np.ndarray
    t: np.ndarray
    yf: np.ndarray
    ycf: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    subgroup_flag: np.ndarray = None
    ate = float
    att = float
    atc = float
    info: str = "No info provided for this dataset."

    # ids
    @property
    def train_val_ids(self):
        return np.concatenate((self.train_ids, self.val_ids))

    # x
    @property
    def x_train(self):
        return self.x[self.train_ids]

    @property
    def x_val(self):
        return self.x[self.val_ids]

    @property
    def x_train_val(self):
        return self.x[self.train_val_ids]

    @property
    def x_test(self):
        return self.x[self.test_ids]

    # t
    @property
    def t_train(self):
        return self.t[self.train_ids]

    @property
    def t_val(self):
        return self.t[self.val_ids]

    @property
    def t_train_val(self):
        return self.t[self.train_val_ids]

    @property
    def t_test(self):
        return self.t[self.test_ids]

    # xt
    @property
    def xt(self):
        return np.column_stack((self.x, self.t))

    @property
    def xt_train(self):
        return self.xt[self.train_ids]

    @property
    def xt_val(self):
        return self.xt[self.val_ids]

    @property
    def xt_train_val(self):
        return self.xt[self.train_val_ids]

    @property
    def xt_test(self):
        return self.xt[self.test_ids]

    # yf
    @property
    def yf_train(self):
        return self.yf[self.train_ids]

    @property
    def yf_val(self):
        return self.yf[self.val_ids]

    @property
    def yf_train_val(self):
        return self.yf[self.train_val_ids]

    @property
    def yf_test(self):
        return self.yf[self.test_ids]

    # ycf
    @property
    def ycf_test(self):
        return self.ycf[self.test_ids]

        # mu0

    @property
    def mu0_test(self):
        return self.mu0[self.test_ids]

    # mu1
    @property
    def mu1_test(self):
        return self.mu1[self.test_ids]
