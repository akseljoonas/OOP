import logging
from abc import ABC, abstractmethod

import numpy as np


class ML_model(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._weights = None
        self._n_features = 0
        self._n_samples = 0

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model on the given dataset (usually training data).

        Args:
            X (np.ndarray): The dataset to train the model on. This should
            be in the format of numpy ndarray of features and labels.

            y (np.ndarray): The targets used for training the model.
            This should also be in the format of numpy ndarray to match the
            data being trained on.

        Returns:
            None
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes a prediction based on the input data (usually test data).

        Args:
            X (np.ndarray): The input data to make predictions on. This should
            be in the format of numpy ndarray of features and labels.

        Returns:
            np.ndarray: The prediction result, which should be in the format
            of numpy ndarray of features you want to predict on.
        """
        pass

    @property
    def weights(self) -> np.ndarray:
        """
        Gets a safe copy of the weights from a pre-trained MLR model.

        Returns:
            copy of 'self.weights'
        """
        return self._weights

    @weights.setter
    def weights(self, parameters: np.ndarray = None) -> None:
        """
        Set the weights of the ML model.

        Args:
            parameters (np.ndarray): np.ndarray of bias value and weights.

        Note:
            The 'parameters' array should include the bias value followed by \
                the weights for each feature.
        """
        self._weights = parameters
        self._n_features = len(parameters) if parameters is not None else 0

    def _check_dimensions(self, length_1: int, length_2: int) -> None:
        """
        _check_dimensions() checks if the provided data dimensions match the \
            criteria.

        Args:
            length_1 (int): Dimension 1
            length_2 (int): Dimension 2

        Raises:
            ValueError: If the provided data dimensions do not match.
        """
        try:
            if length_1 != length_2:
                raise ValueError(
                    f"provided data does not match the dimension\
                criteria: size 1 = {length_1}, size 2 = {length_2}"
                )
        except ValueError as e:
            print(f"Caught: ValueError: {e}")

    def _add_bias_row(self, X: np.ndarray) -> np.ndarray:
        """
        _add_bias_row() adds a bias column to the data matrix.
        This is necessary for computing the weights with a matrix\
        multiplication formula.

        Args:
            X: a n*p np.ndarray where n is number of samples and p is number\
            of features features

        Returns:
            modified dataset X
        """
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _log(
        self,
        iter: int,
        model_name: str,
        total_iter: int,
        loss: float,
        mae: float,
    ) -> None:
        """
        _log() logs information about the model.

        Args:
            iter (int): Current iteration.
            model_name (str): Name of the model (used in the log file).
            total_iter (int): Total number of iterations.
            loss (float): Loss value.
            mae (float): Mean Absolute Error value.

        Returns:
            None
        """

        # Config for logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename="weights.log",
        )

        # Log info
        logging.info(
            f"Name:{model_name} \
            Iteration: {iter+1} of {total_iter}, Loss: {loss}, MAE: {mae}"
        )


class ML_model_GD(ML_model):
    def __init__(
        self,
        alpha: float = 0.05,
        n_iter: int = 100,
        dist: str = "uniform",
        strenght: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter
        self.dist = dist
        self.strenght = strenght

    @abstractmethod
    def _gradient_decent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs gradient descent to optimize the regression model's weights.

        Args:
            X (np.ndarray): The input data to train the model on.
            This should be in the format of a numpy ndarray of features and \
            labels.

            y (np.ndarray): The targets used for training the model.
            This should also be in the format of a numpy ndarray to match the
            data being trained on.

        Returns:
            np.ndarray: Optimized model weights.
        """
        pass

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the model on the given dataset using gradient descent.

        Args:
            X (np.ndarray): The dataset to train the model on.
            This should be in the format of a numpy ndarray of features and\
             labels.

            y (np.ndarray): The targets used for training the model.
            This should also be in the format of a numpy ndarray to match the
            data being trained on.

        Returns:
            None
        """
        # adds a bias row to the prediction dataset X
        X = self._add_bias_row(X)

        # check dimensions of the provided sample size of X and y
        self._n_samples, self._n_features = X.shape
        n_samples_y = (y.shape)[0]
        self._check_dimensions(self._n_samples, n_samples_y)

        # Performs gradient decent on training data
        self._weights = self._gradient_decent(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the input data using the trained model.

        Args:
            X (np.ndarray): The input data to make predictions on.
            This should be in the format of a numpy ndarray of features and
            labels.

        Returns:
            np.ndarray: The prediction result.
        """
        # adds a bias row to the prediction dataset X
        X = self._add_bias_row(X)

        try:
            if X is None:
                raise ValueError(
                    "Dataset X was not provided as an argument or\
                it's empty!"
                )
            model_n_features = self._n_features
            _, data_n_features = X.shape
            self._check_dimensions(model_n_features, data_n_features)
        except ValueError as e:
            print(f"Caught a ValueError: {e}")

        return X.dot(self._weights)
