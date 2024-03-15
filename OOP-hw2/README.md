# OOP - 2023/24 - Assignment 2

This is the base repository for assignment 2.
Please follow the instructions given in the [PDF](https://brightspace.rug.nl/content/enforced/243046-WBAI045-05.2023-2024.1/assignment%202_v1.1.pdf) for the content of the exercise.

## How to carry out your assignment

**PLEASE FOLLOW THESE STEPS:**

1. Use this template and create a private repository:
   ![](use_template.png)
2. Please add your partner and `oop-otoz` to the collaborators.
3. Create a new branch called `submission` **before adding any files**.
4. Add your code in the `main` branch (**IF YOU DO NOT ADD ANYTHING, THE PULL REQUEST WILL NOT WORK**).
5. Make sure that Actions are allowed: Settings -> Actions -> General -> Allow all actions and workflows.
6. Create a pull request from the `main` branch to your `submission` branch and check that your changes are captured.
7. Now finish your solution.
8. When you are ready to submit, add `oop-otoz` to the reviewers.

**Notes:**

- **Leave the \*\***init\***\*.py files untouched**.
- Do not move the `main.py` files.
- Do not move `requirements.txt`.
- Make the pull request AFTER SUBMITTING SOME CHANGES.

Below this line, you can write your report to motivate your design choices.

## Submission

The code should be submitted on GitHub by opening a Pull Request from the `main` branch on to the `submission` branch. This means that `submission` is the base branch and `main` the compare branch. **Make sure to push your code only to `main`!**

There are automated checks that verify that your submission is correct:

1. Deadline - checks that the last commit in a PR was made before the deadline.
2. Reproducibility - downloads libraries included in `requirements.txt` and runs `python3 main.py`. If your code does not throw any errors, it will be marked as reproducible. **Make sure it is reproducible before submission!**
3. Style - runs `flake8` on your code to ensure adherence to style guides.
4. Tests - runs `unittest` on your tests in `part_1/tests` to make sure all tests succeed.

---

# Your report
Start of report


## Part 1
Design Choices Part 1
The Mastermind game uses four classes: GameMaster, Game, CodeMaker, and CodeBreaker.
### game.py
The Game class manages the overall gameplay. It sets the code_length and game_length during initialization and uses the play() method to start the game. It creates instances of CodeMaker, CodeBreaker, and GameMaster to manage different aspects of the game.

### code_maker.py
CodeMaker generates a random secret code of a specified size. It has _colors (protected), __size, and __code (private) attributes. The create_code() method generates the secret code.

### code_breaker.py
CodeBreaker is designed to get user input for the guess of the correct code. It has _colors (protected), __size (private), and _user_guess (protected) attributes. The make_guess() method gets user input and validates it.

### game_master.py
GameMaster provides feedback to the player based on the correct code and the player's guess. It has four attributes: _correct_code, _user_input (protected), __array_1, and __array_2 (private). The _give_feedback method calculates feedback.

### main.py
In main.py, the Game instance is initialized with a specified code_length and game_length, and the play() method starts the game.

### test_code.py
The test_code.py file contains unit tests for the CodeMaker, CodeBreaker, and GameMaster classes using the unittest module. It verifies the functionality of each class 1.


## Part 2

Design Choices in Part 2

Our work consists of six files: ml_model.py, regression.py, main.py, multiple_linear_regression.py, regression_plotter.py, and model_saver.py.

### ml_model.py
This file defines an abstract base class ML_model for a generic ML model. It includes methods for training and predicting, as well as private helper methods for checking dimensions, adding a bias row to the data and creating a log file. The _weights attribute is private, as it should not be accessed directly by the user. Instead, it can be accessed via the weights property, which returns a safe copy of the weights. The weights setter allows for updating the weights of the model. The attributes _n_features and _n_samples are also private since they are used only inside the methods and shouldnt be accessed by outside of them aswell.

The ML_model_GD class, which inherits from ML_model, adds additional functionality to ML_models that have the need for gradient descent. GD is a mandatory private method that needs to be used in all classes inheriting from it. It is private because users of the resulting classes (e.g. LassoRegression) should not tamper with the gradient decent process. The class also introduces the alpha (learning rate), n_iter (number of iterations in training), dist (distribution of initial weight vector), and strength (regularization strength for Lasso and Ridge regression) attributes, which are used to control the gradient descent process.

### regression.py
This file contains the implementation of the LassoRegression and RidgeRegression classes. Both classes inherit from the ML_model_GD class which itself is a subclass of the ML_model class. This allows the LassoRegression and RidgeRegression classes to utilize the gradient descent algorithm implemented in ML_model_GD and the other base functionalities of an ML model provided by the ML_model class.

The LassoRegression and RidgeRegression classes override the _gradient_decent method from the ML_model_GD class. This is because the gradient descent algorithm needs to be adjusted for the Lasso and Ridge regression, which include a penalty term in the loss function. The penalty term is implemented differently in Lasso and Ridge regression, hence the need for different _gradient_decent implementations.

Both LassoRegression and RidgeRegression classes have alpha, n_iter, dist, and strength as initial parameters, similar to ML_model_GD. These parameters have the same functionaly as above.

The LassoRegression class includes a private method _sign, which returns the sign of the item depending of each element in the input array. This is private since it is only used in LassoRegression.


### multiple_linear_regression.py
This file defines the MultipleLinearRegression class, which is a specific implementation of the ML_model abstract base class. It inherits attributes from ML_model class and overrides the train and predict methods (public) to implement the MLR algorithm without gradient decent.

### model_saver.py
This file defines the ModelSaver class, which is a slightly modified version of ModelSaver from the previous assignment. The modification that we made was adding compatibility with @property decorators that were used in all classes that inherit from ML_model. The reason we do not use @property here is twofold: We need more functionaly than is possible with the standard implementation of @property decorators and from the perspective of the end user having methods to save and load parameters is more intuitive and understandable. 

### regression_plotter.py
This file defines the RegressionPlotter class, which is used for visualizing the regression models. Other than adding comptability with the @property decorators used in models themselves it remains unchanged since the last assignment. 

### main.py
The main.py demonstrates all the functionality developed in this assignment.

The main() function is then called inside if __name__ == "__main__": block. It creates instances of the MultipleLinearRegression, LassoRegression and RidgeRegression classes, loads training and test data from a numpy file, trains regression on the training data, and makes predictions. The results of the predictions are then saved to a file.

Then the weights from the file are loaded into a new instance and these are also predicted on. Finally it prints the predictions for all of the models.