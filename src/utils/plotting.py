# IMPORT MODULES
# Standard library
from typing import Callable, Dict, Optional

# Third party
import numpy as np
import plotly.graph_objects as go
import yaml
from sklearn.utils import Bunch


def get_true_drc(
        data: Bunch,
        num_integration_samples: int = 65,
) -> np.ndarray:
    """
    Calculates the true dose-response curve (DRC) for the given data.

    Parameters:
    data (Bunch): The dataset containing the dose-response information.
    num_integration_samples (int, optional): The number of samples to use for numerical integration. Defaults to 65.

    Returns:
    np.ndarray: The calculated true DRC as a numpy array.
    """
    # Save observation according to quantile
    x = np.quantile(data.x[data.test_ids], q=0.5, axis=0).reshape(1, -1)

    # Multiply observation
    x = x.repeat(num_integration_samples, 0)

    # Get doses
    d = np.linspace(0, 1, num_integration_samples)

    # Define treatment
    t = np.zeros(num_integration_samples)

    return data.ground_truth(x, d, t)


def predict_drc(
        data: Bunch,
        model: Callable,
        num_integration_samples: int = 65,
):
    """
    Predicts the dose-response curve (DRC) for the given data using the provided model.

    Parameters:
    data (Bunch): The dataset containing the dose-response information.
    model (Callable): The machine learning model to be used for prediction.
    num_integration_samples (int, optional): The number of samples to use for numerical integration. Defaults to 65.

    Returns:
    np.ndarray: The predicted DRC as a numpy array.
    """
    # Save observation according to quantile
    x = np.quantile(data.x[data.test_ids], q=0.5, axis=0).reshape(1, -1)

    # Multiply observation
    x = x.repeat(num_integration_samples, 0)

    # Get treatments
    d = np.linspace(0, 1, num_integration_samples)

    # Define treatment
    t = np.zeros(num_integration_samples)

    return model.predict(x, d, t)

def load_plot_settings(config_path: str) -> dict:
    """
    Load plot settings from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Plot settings.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['plot_settings']
