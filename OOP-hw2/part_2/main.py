import sys
import numpy as np

from src.model_saver import ModelSaver
from src.multiple_linear_regression import MultipleLinearRegression
from src.regression import LassoRegression, RidgeRegression
from src.regression_plotter import RegressionPlotter

sys.path.append("..")


def main():
    # Loading training and test data into varibales

    X = np.load("./part_2/train.npy")
    y = np.load("./part_2/targets.npy")

    # creating an instance of Regression models
    model_lr = MultipleLinearRegression()
    model_lasso = LassoRegression(alpha=0.05, n_iter=50)
    model_ridge = RidgeRegression(alpha=0.05, n_iter=50, strenght=0.1)

    # fit the model with multiple linear, lasso and ridge regression
    # with the train data and make a prediction

    model_lr.train(X, y)
    prediction = model_lr.predict(X)

    model_lasso.train(X, y)
    prediction_lasso = model_lasso.predict(X)

    model_ridge.train(X, y)
    prediction_ridge = model_ridge.predict(X)

    # create an instance of the plotter RegressionPlotter class
    plotter = RegressionPlotter()

    # plot lasso and ridge regression
    plotter.plot(model_lr, X, y, plot_type="2D", feature_index=0)
    plotter.plot(model_lasso, X, y, plot_type="2D", feature_index=0)
    plotter.plot(model_ridge, X, y, plot_type="2D", feature_index=0)

    # Saving all the weights of the models
    model_saver_ = ModelSaver()
    model_saver_.save_parameters(model=model_lr, file_format="csv")
    model_saver_.save_parameters(model=model_lasso, file_format="csv")
    model_saver_.save_parameters(model=model_ridge, file_format="csv")

    # Creating new instances
    new_linear = MultipleLinearRegression()
    new_lasso = LassoRegression()
    new_ridge = RidgeRegression()

    # Loading parameters to new instances
    model_saver_.load_parameters(
        file_name=f"./{type(new_linear).__name__}_weights.csv",
        model=new_linear,
        file_format="csv",
    )

    model_saver_.load_parameters(
        file_name=f"./{type(new_lasso).__name__}_weights.csv",
        model=new_lasso,
        file_format="csv",
    )

    model_saver_.load_parameters(
        file_name=f"./{type(new_ridge).__name__}_weights.csv",
        model=new_ridge,
        file_format="csv",
    )

    # Making predictions with loaded models
    new_linear_pred = new_linear.predict(X)
    new_lasso_pred = new_lasso.predict(X)
    new_ridge_pred = new_ridge.predict(X)

    print("LINEAR prediction")
    print(prediction)
    print("LASSO prediction")
    print(prediction_lasso)
    print("RIDGE prediction")
    print(prediction_ridge)
    print("NEW LINEAR prediction")
    print(new_linear_pred)
    print("NEW LASSO prediction")
    print(new_lasso_pred)
    print("NEW RIDGE prediction")
    print(new_ridge_pred)
    print("GROUND TRUTH")
    print(y)


if __name__ == "__main__":
    main()
