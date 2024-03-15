import numpy as np
from src.ml_model import ML_model


class MultipleLinearRegression(ML_model):
    def __init__(self):
        super().__init__()

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        train() method takes a dataset X and corresponding values y and\
        performs a multiple linear regression
        to fit the weights to the given dataset.

        Args:
            X: a np.ndarray dataset with dimesions n*p where n is number of\
            samples and p is number of features.\n
            y: a np.ndarray of the size n that correspond to the number of\
            samples n in the X.
        """
        # adding a bias row to the training data
        X = self._add_bias_row(X)

        # check dimensions of the provided sample size of X and y
        self._n_samples, self._n_features = X.shape
        n_samples_y = (y.shape)[0]
        self._check_dimensions(self._n_samples, n_samples_y)

        # initialise model weights (including bias row) to zero
        self._weights = np.zeros(self._n_features)

        # fitting the data with a matrix formula
        self._weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """
        predict() method takes a data set as an argument and\
        produces a prediction y

        Args:
            X: a np.ndarray dataset with number of features equal to the\
            number of features of the training data.
        Returns:
            np.ndarray of prediction values for each sample in the input X
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
