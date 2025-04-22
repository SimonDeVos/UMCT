# LOAD MODULES
# Standard library
from typing import Callable, Tuple

# Third party
import numpy as np
from scipy.integrate import romb
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

def mean_integrated_prediction_error(
    x: np.ndarray,
    response: Callable,
    model: Callable,
    num_integration_samples: int = 65,
) -> float:
    """
    Calculates the Mean Integrated Prediction Error (MIPE) for a given model.

    The MIPE is a measure of the discrepancy between the true response and the model's predicted response,
    integrated over the distribution of the covariates. It is used to evaluate the performance of a model.

    Parameters:
        x (np.ndarray): The covariates.
        response (Callable): A function representing the true response.
        model (Callable): The model to be evaluated.
        num_integration_samples (int, optional): The number of samples to be used for the integration. Defaults to 65.

    Returns:
        float: The Mean Integrated Prediction Error of the model.
    """
    # Get step size
    step_size = 1 / num_integration_samples
    num_obs = x.shape[0]

    # Generate data
    x = np.repeat(x, repeats=num_integration_samples, axis=0)
    d = np.linspace(0, 1, num_integration_samples)
    d = np.tile(d, num_obs)

    # Get true outcomes
    y = response(x, d)
    # Get predictions
    y_hat = model.predict(x, d)

    # Get mise
    mises = []
    y_chunks = y.reshape(-1, num_integration_samples)
    y_hat_chunks = y_hat.reshape(-1, num_integration_samples)

    for y_chunk, y_hat_chunk in zip(y_chunks, y_hat_chunks):
        mise = romb((y_chunk - y_hat_chunk) ** 2, dx=step_size)
        mises.append(mise)

    return np.sqrt(np.mean(mises))