# import matplotlib as plt
import numpy as np

# import pandas as pd
from src.ml_model import ML_model_GD


class LassoRegression(ML_model_GD):
    def __init__(
        self,
        alpha: float = 0.05,
        n_iter: int = 100,
        dist: str = "uniform",
        strenght: float = 0.05,
    ) -> None:
        super().__init__(
            alpha=alpha, n_iter=n_iter, dist=dist, strenght=strenght
        )

    def _gradient_decent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs gradient descent to optimize the weights for Lasso Regression.

        Args:
            X (np.ndarray): The input data to train the model on.
            This should be in the format of a numpy ndarray of features and
            labels.

            y (np.ndarray): The targets used for training the model.
            This should also be in the format of a numpy ndarray to match the
            data being trained on.

        Returns:
            np.ndarray: Optimized model weights.
        """
        # We randomly initialize the parameters from a normal
        # or an uniform distribution
        try:
            if self.dist is None:
                raise ValueError(
                    "need to specify dist with as 'normal'\
                or 'uniform'"
                )
            elif self.dist not in ["normal", "uniform"]:
                raise ValueError(
                    "dist needs to be set to either 'normal'\
                or 'uniform"
                )
        except ValueError as e:
            print(f"Caught ValueError: {e}")

        if self.dist == "normal":
            w = np.random.normal(loc=0, scale=1, size=self._n_features)
        else:
            w = np.random.uniform(low=-1, high=1, size=self._n_features)

        for i in range(self.n_iter):
            # Calculate a prediction with current weights
            y_hat = X.dot(w)

            # Compute the gradient ∂L as indicated in the assignment
            lasso = self.strenght * self.__sign(w)

            n = self._n_samples
            gradient = -1 * (2 / n) * X.T.dot((y - y_hat)) + lasso

            # Calculate the loss and MAE and log current weights
            loss = np.mean((y - y_hat) ** 2) + self.strenght * np.sum(
                np.abs(w)
            )
            mae = np.mean(np.abs(y - y_hat))
            self._log(i, "lasso", self.n_iter, loss, mae)

            # We update w according to Equation (4)
            w = w - self.alpha * gradient

        return w

    def __sign(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the sign of each element in the input array.

        Args:
            x (np.ndarray): Input array.

        Returns:
            np.ndarray: Array of the same shape as input with elements
            replaced by their sign.
        """
        output = x.copy()
        for i in range(len(x)):
            if x[i] == 0:
                output[i] = 0
            elif x[i] > 0:
                output[i] = 1
            else:
                output[i] = -1

        return output


class RidgeRegression(ML_model_GD):
    def __init__(
        self,
        alpha: float = 0.05,
        n_iter: int = 100,
        dist: str = "uniform",
        strenght: float = 0.05,
    ) -> None:
        super().__init__(
            alpha=alpha, n_iter=n_iter, dist=dist, strenght=strenght
        )

    def _gradient_decent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Performs gradient descent to optimize the weights for Ridge Regression.

        Args:
            X (np.ndarray): The input data to train the model on.
            This should be in the format of a numpy ndarray of features and
            labels.

            y (np.ndarray): The targets used for training the model.
            This should also be in the format of a numpy ndarray to match the
            data being trained on.

        Returns:
            np.ndarray: Optimized model weights.
        """
        try:
            if self.dist is None:
                raise ValueError(
                    "need to specify dist with as 'normal'\
                or 'uniform'"
                )
            elif self.dist not in ["normal", "uniform"]:
                raise ValueError(
                    "dist needs to be set to either 'normal'\
                or 'uniform"
                )
        except ValueError as e:
            print(f"Caught ValueError: {e}")

        # We randomly initialize the parameters from a normal
        # or an uniform distribution
        if self.dist == "normal":
            w = np.random.normal(loc=0, scale=1, size=self._n_features)
        else:
            w = np.random.uniform(low=-1, high=1, size=self._n_features)

        for i in range(self.n_iter):
            # Calculate a prediction with current weights
            y_hat = X.dot(w)

            # Compute the gradient ∂L as indicated in the assignment
            ridge = 2 * self.strenght * w

            n = self._n_samples
            gradient = -1 * (2 / n) * X.T.dot((y - y_hat)) + ridge

            # Calculate the loss and MAE and log current weights
            loss = np.mean((y - y_hat) ** 2) + self.strenght * np.sqrt(
                np.sum(w**2)
            )
            mae = np.mean(np.abs(y - y_hat))
            self._log(i, "ridge", self.n_iter, loss, mae)

            # We update weights according to Equation (4)
            w = w - self.alpha * gradient

        return w
