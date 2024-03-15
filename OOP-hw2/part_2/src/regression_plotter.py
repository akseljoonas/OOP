from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class RegressionPlotter:
    def plot(
        self,
        model: Any = None,
        X: np.ndarray = None,
        y: np.ndarray = None,
        plot_type="2D",
        feature_index: int = None,
        feature_list: List[int] = None,
    ) -> plt:
        """
        plot() method is used to plot the multiple linear regression line and\
        the data as a scatter plot that have been fitted with a MLR model.

        Args:
            model: a regression model that has a _get_weights() method to\
            retrieve a safe copy of the weights
            plot_type: "2D" (default) to plot one feature against the\
            predicted values , "3D" to plot two features against the predicted\
            values, and "multi" to plot multiple independent features against\
            the predicted values\n
            X: a dataset of the size n*p where n is the number of samples and\
            p is the number of features\n
            y: an np.ndarray of predicted values for the dataset X\n
            feature_index: an int that specifies which feature from the dataX\
            is to be plotted \n
            feature_list: list of feature indexes (int) that are used for\
            plot_type="3D" and plot_type="multi"\n
        Returns:
            visual plots.

        """

        # getting a safe copy of weights from the model
        weights = model.weights

        X = self.__add_bias_row(X)

        if plot_type == "2D":
            # checking if the user has specified mandatory parameter
            # feature_index
            try:
                if feature_index is None:
                    raise ValueError(
                        f"feature_index is a mandatory parameter\
                    for the plot_type: {plot_type}"
                    )
            except ValueError as e:
                print(f"Caught a ValueError: {e}")
                return

            # checking if each specified index in the list is in bounds with
            # the number of features in the model
            try:
                if feature_index > (len(weights) - 2):
                    raise ValueError(
                        "feature_index biggest than the number of\
                                    features in the model"
                    )
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                return

            # setting the weights and intercept
            weight = weights[feature_index + 1]
            intercept = weights[0]
            xx = X[:, feature_index + 1]

            # plotting
            reg_line = weight * xx + intercept
            plt.scatter(X[:, feature_index + 1], y)
            plt.plot(X[:, feature_index + 1], reg_line, color="red")
            plt.show()

        elif plot_type == "3D":
            # checking if the user has specified feature list
            try:
                if feature_list is None:
                    raise ValueError(
                        f"feature_list is a mandatory parameter\
                    for the plot_type: {plot_type}"
                    )
            except ValueError as e:
                print(f"Caught a ValueError: {e}")
                return

            # checking if each specified index in the list is in bounds with
            # the number of features in the model
            try:
                for id in feature_list:
                    if id > (len(weights) - 2):
                        raise ValueError(
                            "feature id in feature_list is\
                        biggest than the number of features in the model"
                        )
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                return

            # checking whether user want to plot two same features against
            # each other and that the feature_list has two features
            try:
                if len(feature_list) != 2:
                    raise ValueError(
                        f"3D plot requires two features, but\
                    feature_list of length {len(feature_list)} was received"
                    )
            except ValueError as e:
                print(f"Caught a ValueError: {e}")
                return

            try:
                if feature_list[0] == feature_list[1]:
                    raise ValueError("same feature was used twice!")
            except ValueError as e:
                print(f"Caught a ValueError: {e}")

            xx = X[:, feature_list[0] + 1]
            yy = X[:, feature_list[1] + 1]
            xx, yy = np.meshgrid(xx, yy)

            # getting the weights and creating the hyperplane
            weight_1 = weights[feature_list[0] + 1]
            weight_2 = weights[feature_list[1] + 1]
            intercept = weights[0]
            hyperplane = weight_1 * xx + weight_2 * yy + intercept

            # plotting the hyperplane
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(xx, yy, hyperplane, cmap=cm.Reds)

            # Plot the data as scatter plot (two features against the y values)
            xx_flat = xx.flatten()
            yy_flat = yy.flatten()
            # Scatter plot of data points
            ax.scatter(
                xx_flat, yy_flat, np.tile(y, len(xx_flat) // len(y)), c="b",
                marker="o"
            )
            ax.set(xticklabels=[], yticklabels=[], zticklabels=[])
            plt.show()

        elif plot_type == "multi":
            try:
                if feature_list is None:
                    raise ValueError(
                        f"feature_list is a mandatory parameter\
                    for the plot_type: {plot_type}"
                    )
            except ValueError as e:
                print(f"Caught a ValueError: {e}")
                return
            try:
                for id in feature_list:
                    if id > (len(weights) - 2):
                        raise ValueError(
                            "feature id in feature_list is\
                        biggest than the number of features in the model"
                        )
            except ValueError as e:
                print(f"Caught ValueError: {e}")
                return

            for feature_id in feature_list:
                # setting the weight and intercept
                weight = weights[feature_id + 1]
                intercept = weights[0]
                xx = X[:, feature_id + 1]

                # plotting the scatter plot and regression line
                reg_line = weight * xx + intercept
                plt.scatter(X[:, feature_id + 1], y)
                plt.plot(X[:, feature_id + 1], reg_line, color="red")
                plt.show()
        else:
            raise ValueError(
                "Please use '2D', '3D' or 'multi' as the\
            plot_type argument"
            )

    def __add_bias_row(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]
