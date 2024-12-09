# LOAD MODULES
# Standard library
from typing import Callable, Tuple, Optional, Type, Dict

# Proprietary
from src.methods.utils.classes import ContinuousCATE
from src.methods.utils.regressors import LinearRegression, LogisticRegression

# Third party
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.stats import norm


class NLearner(ContinuousCATE):
    """
    N-Learner class that inherits from the ContinuousCATE class.

    This class represents an N-Learner, which is a type of causal model. It inherits
    from the ContinuousCATE class, which provides base functionality for continuous
    causal additive treatment effect models.

    For settings with multiple continuous-valued treatments, the N-Learner fits a separate
    model for every treatment.

    Methods:
        fit(X, Y, D, T): Fits the model to the data.
        predict(X): Predicts the treatment effect for the given data.
    """

    def __init__(
            self,
            model_dict: Dict[int, Callable] = {0: LinearRegression},
            **kwargs
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified base model and additional parameters.

        Parameters:
            model_dict: A dictionary with the treatment as key and the model per treatment as value.
            **kwargs: Additional keyword arguments to be passed to the base model.
        """
        # Save model (sklearn estimator)
        self.model_dict = model_dict
        self.kwargs = kwargs
        self.models = {}

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
    ) -> None:
        """
        This is dummy method, as the class only takes pre-trained models.
        """
        print("Not able to fit model. Please use pre-trained models.")

    def predict(
            self,
            x: np.ndarray,
            d: np.ndarray,
            t: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the treatment effect for the given data.
        """
        # Generate predictions based on regressor flag:
        y_hat = np.zeros(x.shape[0])
        for unique_t in np.unique(t):
            mask = t == unique_t
            x_subset = x[mask]
            d_subset = d[mask]
            t_subset = t[mask]

            # Calculate predictions
            y_hat[mask] = self.model_dict[unique_t].predict(x_subset, d_subset, t_subset)

        return y_hat


class SLearner(ContinuousCATE):
    """
    S-Learner class that inherits from the ContinuousCATE class.

    This class represents an S-Learner, which is a type of causal model. It inherits
    from the ContinuousCATE class, which provides base functionality for continuous
    causal additive treatment effect models.

    Methods:
        fit(X, Y, D, T): Fits the model to the data.
        predict(X, D, T): Predicts the treatment effect for the given data.
    """

    def __init__(
            self,
            base_model: BaseEstimator = LinearRegression,
            **kwargs
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified base model and additional parameters.

        Parameters:
            base_model (BaseEstimator, optional): The base model to be used in the class. Defaults to LinearRegression.
            **kwargs: Additional keyword arguments to be passed to the base model.
        """
        # Save model (sklearn estimator)
        self.base_model = base_model(**kwargs)

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
    ) -> None:
        """
        Fits model to the data.
        """
        # Concat data
        xdt = np.column_stack((x, d, t))

        # Fit model
        self.base_model.fit(xdt, y)

    def predict(
            self,
            x: np.ndarray,
            d: np.ndarray,
            t: np.ndarray
    ) -> np.ndarray:
        """
        Predicts the treatment effect for the given data.
        """
        xdt = np.column_stack((x, d, t))

        # Generate predictions based on regressor flag:
        try:
            y_hat = self.base_model.predict_proba(xdt)[:, 1]
        except:
            y_hat = self.base_model.predict(xdt)

        return y_hat


class HIE(ContinuousCATE):
    """
    HIE class that inherits from the ContinuousCATE class.

    This class represents a Hirano Imbens Estimator (HIE), which is a type of causal model. It inherits
    from the ContinuousCATE class, which provides base functionality for continuous
    causal additive treatment effect models.

    Methods:
        fit(X, Y, D, T): Fits the model to the data.
        predict(X): Predicts the treatment effect for the given data.
    """

    def __init__(
            self,
            gps_model: Type[BaseEstimator] = LogisticRegression,
            effect_model: Type[BaseEstimator] = LogisticRegression,
            treatment_interaction_degree: int = 1,
            outcome_interaction_degree: int = 2,
            pca_degree: Optional[int] = None,
            **kwargs
    ) -> None:
        """
        Initializes a new instance of the class.

        This method initializes a new instance of the class with the specified parameters.

        Parameters:
            gps_model (Type[BaseEstimator], optional): The model to be used for the Generalized Propensity Score. Defaults to LogisticRegression.
            effect_model (Type[BaseEstimator], optional): The model to be used for the effect estimation. Defaults to LogisticRegression.
            treatment_interaction_degree (int, optional): The degree of interaction for the treatment. Defaults to 1.
            outcome_interaction_degree (int, optional): The degree of interaction for the outcome. Defaults to 2.
            pca_degree (Optional[int], optional): The degree of PCA to be applied. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base models.
        """
        # Define feature transformer
        self.treatment_feature_transformer = PolynomialFeatures(
            treatment_interaction_degree, include_bias=False
        )
        self.outcome_feature_transformer = PolynomialFeatures(
            outcome_interaction_degree, include_bias=False
        )

        # Initialize treatment and outcome models
        self.treatment_model = gps_model(**kwargs)
        self.outcome_model = effect_model(**kwargs)

        # Save settings
        self.treatment_error_scale = 1
        self.pca_degree = pca_degree

    def fit(
            self,
            x: np.ndarray,
            y: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
    ) -> None:
        """
        Fits the model to the given data.

        This method takes as input the features, target, and treatment indicator, and fits the model to the data.

        Parameters:
            x (np.ndarray): The features.
            y (np.ndarray): The target.
            d (np.ndarray): The dose.
            t (np.ndarray): The treatment indicator.
        """
        # Create x by concatenating x and t
        x = np.column_stack((x, t))

        # Update pca_degree
        if self.pca_degree is not None:
            if x.shape[1] < self.pca_degree:
                self.pca_degree = x.shape[1]

            # Define PCA
            self.pca = PCA(self.pca_degree, random_state=42).fit(x)

            # PCA transform data
            x = self.pca.transform(x)

        # Transform data
        x = self.treatment_feature_transformer.fit_transform(x)

        # Train treatment model
        self.treatment_model.fit(x, d)

        # Estimate errors
        try:
            errors = self.treatment_model.predict_proba(x)[:, 1] - d
        except:
            errors = self.treatment_model.predict(x) - d

        # Update treatment std error
        _, self.treatment_error_scale = norm.fit(errors)

        # Get propensity score and generate prediction data
        propensity_scores = norm.pdf(errors, loc=0, scale=self.treatment_error_scale)

        # Train data (propensity score, t)
        pd = np.column_stack((propensity_scores, d))

        # Transform data
        pd = self.outcome_feature_transformer.fit_transform(pd)

        # Fit outcome model
        self.outcome_model.fit(pd, y)

    def predict(
            self,
            x: np.ndarray,
            d: np.ndarray,
            t: np.ndarray,
    ) -> np.ndarray:
        """
        Predicts the treatment effect for the given features and treatment indicator.

        This method takes as input the features and treatment indicator, and predicts the treatment effect.

        Parameters:
            x (np.ndarray): The features.
            d (np.ndarray): The dose.
            t (np.ndarray): The treatment indicator.

        Returns:
            np.ndarray: The predicted treatment effect.
        """
        # Create x by concatenating x and t
        x = np.column_stack((x, t))

        # PCA transform data
        if self.pca_degree is not None:
            x = self.pca.transform(x)

        # Transform data
        x = self.treatment_feature_transformer.fit_transform(x)

        # Errors
        try:
            errors = self.treatment_model.predict_proba(x)[:, 1] - d
        except:
            errors = self.treatment_model.predict(x) - d

        # Get propensity score and generate prediction data
        propensity_scores = norm.pdf(errors, loc=0, scale=self.treatment_error_scale)

        # Train data (propensity score, t)
        pd = np.column_stack((propensity_scores, d))

        # Transform data
        pd = self.outcome_feature_transformer.fit_transform(pd)

        # Predict
        try:
            y_hat = self.outcome_model.predict_proba(pd)[:, 1]
        except:
            y_hat = self.outcome_model.predict(pd)

        return y_hat