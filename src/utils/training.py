# LOAD MODULES
# Standard library
from typing import Callable, Tuple, Dict, Optional, Union
import random
import itertools

# Third party
import numpy as np
from tqdm import tqdm
from sklearn.utils import Bunch

# Proprietary
from src.data.utils import ContinuousData


def train_val_tuner(
        data: ContinuousData,
        model: Callable,
        parameters: dict,
        name: str = "method",
        num_combinations: Optional[int] = None,
) -> Tuple[Callable, dict]:
    """
    Performs training and validation tuning on a given model with specified parameters.

    Parameters:
        data (ContinuousData): The dataset to be used for training and validation.
        model (Callable): The machine learning model to be tuned.
        parameters (dict): The parameters for the model.
        name (str, optional): The name of the method. Defaults to "method".
        num_combinations (int, optional): The number of parameter combinations to consider. If None, all combinations are considered. Defaults to None.

    Returns:
        Tuple[Callable, dict]: The tuned model and the best found settings for the model parameters.
    """
    random.seed(42)
    current_best = np.inf
    combinations = list(itertools.product(*parameters.values()))
    random.shuffle(combinations)

    if (num_combinations is not None) and (num_combinations < len(combinations)):
        combinations = combinations[:num_combinations]

    for combination in tqdm(combinations, leave=False, desc="Tune " + name):
        settings = dict(zip(parameters.keys(), combination))
        estimator = model(**settings)
        estimator.fit(data.x_train, data.y_train, data.d_train, data.t_train)
        score = estimator.score(data.x_val, data.y_val, data.d_val, data.t_val)

        if current_best > score:
            best_parameters = settings
            current_best = score

    final_model = model(**best_parameters)
    final_model.fit(data.x_train, data.y_train, data.d_train, data.t_train)

    return final_model, best_parameters


def cv_tuner(
        data: ContinuousData,
        model: Callable,
        parameters: dict,
        name: str = "method",
        num_combinations: Optional[int] = None,
        num_folds: int = 5,
) -> Tuple[Callable, dict]:
    """
    Performs cross-validation tuning on a given model with specified parameters.

    Parameters:
        data (ContinuousData): The dataset to be used for training and validation.
        model (Callable): The machine learning model to be tuned.
        parameters (dict): The parameters for the model.
        name (str, optional): The name of the method. Defaults to "method".
        num_combinations (int, optional): The number of parameter combinations to consider. If None, all combinations are considered. Defaults to None.
        num_folds (int, optional): The number of folds for cross-validation. Defaults to 5.

    Returns:
        Tuple[Callable, dict]: The tuned model and the best found settings for the model parameters.
    """
    results = []
    random.seed(42)
    train_ids = data.train_val_ids
    fold_ids = np.array_split(data.train_val_ids, num_folds)
    combinations = list(itertools.product(*parameters.values()))
    random.shuffle(combinations)

    if (num_combinations is not None) and (num_combinations < len(combinations)):
        combinations = combinations[:num_combinations]

    for combination in tqdm(combinations, leave=False, desc="Tune " + name):
        settings = dict(zip(parameters.keys(), combination))

        fold_results = []
        for f in tqdm(range(num_folds), desc="Iterate over folds", leave=False):
            x_fold_train = data.x[fold_ids[f]]
            x_fold_val = data.x[np.setdiff1d(train_ids, fold_ids[f])]
            y_fold_train = data.y[fold_ids[f]]
            y_fold_val = data.y[np.setdiff1d(train_ids, fold_ids[f])]
            d_fold_train = data.d[fold_ids[f]]
            d_fold_val = data.d[np.setdiff1d(train_ids, fold_ids[f])]
            t_fold_train = data.t[fold_ids[f]]
            t_fold_val = data.t[np.setdiff1d(train_ids, fold_ids[f])]

            estimator = model(**settings)

            estimator.fit(x_fold_train, y_fold_train, d_fold_train, t_fold_train)
            fold_results.append(estimator.score(x_fold_val, y_fold_val, d_fold_val, t_fold_val))

        results.append(np.mean(fold_results))

    best_option = np.argmin(results)
    best_settings = dict(zip(parameters.keys(), combinations[best_option]))

    final_model = model(**best_settings)
    final_model.fit(data.x[train_ids], data.y[train_ids], data.d[train_ids], data.t[train_ids])

    return final_model, best_settings
