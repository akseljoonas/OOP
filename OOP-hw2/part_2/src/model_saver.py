import json
from typing import Any

import numpy as np


class ModelSaver:
    def save_parameters(
        self, model: Any = None, file_format: str = "csv"
    ) -> None:
        """
        save_parameters() method gets the parameters from a generic model
        and saves them into a specified file type.

        Args:
            model: pre-trained regression model that has  a _get_weights()\
            method to retreive a safe copy of the weights\n
            file_format: 'csv' (default) or 'json'
        Returns:
            saves the parameters into a specified file format\
            (either csv or json)
        """
        try:
            if file_format is None:
                raise ValueError(
                    "need to specify file_format with as 'csv'\
                or 'json'"
                )
            elif file_format not in ["csv", "json"]:
                raise ValueError(
                    "file_format needs to be set to either 'csv'\
                or 'json"
                )
        except ValueError as e:
            print(f"Caught ValueError: {e}")

        parameters = model.weights
        try:
            if parameters is None:
                raise ValueError(
                    "Parameters do not exist (model has not been\
                trained yet)"
                )
        except ValueError as e:
            print(f"Caught ValueError: {e}")

        if file_format == "csv":
            np.savetxt(
                f"./{type(model).__name__}_weights.csv",
                parameters,
                delimiter=",",
            )
        elif file_format == "json":
            with open(f"./{model.__name__}_weights.json", "a") as json_file:
                json.dump(parameters, json_file)

    def load_parameters(
        self,
        file_name: str = None,
        model: Any = None,
        file_format: str = "csv",
    ) -> None:
        """
        load_parameters() method loads the parameters from a csv or json file
        and sets them to a current model.

        Args:
            file_name: a string that specifies the location of the file,\
            e.g., './model_weights.csv'\n
            model: an MultipleLinearRegression model into which the user wants\
            to import the predefined parameters\n
            file_format: 'csv' or 'json'
        """

        # check if the actual file's format matches the defined file_format
        try:
            if not file_name.endswith(f".{file_format}"):
                raise TypeError(
                    f"The file format of {file_name} does not\
                match specified file_format='{file_format}'"
                )
        except TypeError as e:
            print(f"Caught ValueError: {e}")
            return

        # saves the parameters gained from the file (csv or json) to the
        # variable parameters as np.ndarray
        try:
            if file_format == "csv":
                parameters = np.loadtxt(file_name, delimiter=",")
            elif file_format == "json":
                with open(file_name, "r") as json_file:
                    parameters = json.load(json_file)
            else:
                raise ValueError(
                    f"{file_format} is not a supported file\
                format"
                )
        except FileNotFoundError as e:
            print(f"Caught FileNotFoundError: {e}")
            return

        # checks if the loaded file was empty
        try:
            if parameters is None:
                raise ValueError("loaded file is empty")
        except ValueError as e:
            print(f"Caught: ValueError: {e}")

        # calls model method _load_weights() to load the weights from the
        # variable 'parameters' into the model
        model.weights = parameters
