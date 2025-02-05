import numpy as np
import math
from typing import Dict, List
from sklearn.linear_model import LinearRegression
import joblib
import os
import logging

class ScalePredictor:
    """
    A predictor that, given historical trace data (for multiple functions),
    can be trained using linear regression to predict how many instances
    should be scaled at a certain time.

    Methods:
        train(dataset, window_size):
            Train a predictive model using historical data.
        predict(window, index) -> int:
            Predict the number of instances needed based on current usage window.
        clean():
            Clean or reset the current predictive model.
    """

    def __init__(self, debug):
        self.models: Dict[str, LinearRegression] = {}
        self.trained: bool = False
        self.window_size: int = 0
        self.debug = debug
        self.logger = logging.getLogger(__name__)

    def train(self, dataset: Dict[str, List[int]], window_size: int):
        """
        Train a linear regression model for each function based on its historical data.
        For each second we predict next second.
        
        Args:
            dataset: 
                A dict: {functionName: [invocation data per sec, ...]}
            window_size:
                The window size, normally 60 in practice, indicates the sequence
                length of feature vector.
        """
        self.window_size = window_size
        for function_name, call_list in dataset.items():
            if len(call_list) < 2:
                # If not enough data, skip.
                continue

            X, y = [], []
            # Generate training features.
            for i in range(len(call_list) - window_size):
                X.append(call_list[i : i + window_size])
                y.append(call_list[i + window_size])

            # Linear Regression.
            model = LinearRegression()
            model.fit(X, y)

            self.models[function_name] = model

        self.trained = len(self.models) > 0

    def predict(self, function_name: str, window: List[int], index: int) -> int:
        """
        Predict how many instances are needed for a given function based on
        the most recent usage window (e.g., last 60 seconds).

        Args:
            function_name (str): The name of the function to predict for.
            window (List[int]): Invocation data in the last minute (or any timespan).
        
        Returns:
            int: The number of instances needed.
        """
        self.logger.debug("predict called.")
        if not self.trained or function_name not in self.models or self.window_size == 0:
            if self.debug == "0":
                self.logger.error(f"No trained model found for function '{function_name}'. ")
                raise KeyError(
                    f"No trained model found for function '{function_name}'. "
                )
            else:
                self.logger.debug(f"No trained model found for function '{function_name}', but returns 1 in debug mod.")
                return 1

        model = self.models[function_name]

        # Rearrange the windows according to the index to ensure order.
        # Spin and split window.
        sequenced_window = (window[index+1:] + window[:index+1])[-self.window_size:]

        # Predict next second.
        # Round up to an integer.
        predicted_next = math.ceil(model.predict([sequenced_window])[0])

        self.logger.info(f"predicted pod count = {predicted_next}")
        self.logger.debug("predict returns")

        if predicted_next < 0:
            return 0
        
        return math.ceil(predicted_next)

    def clear(self):
        """
        Clean all trained models.
        """
        self.models.clear()
        self.trained = False
        self.window_size = 0

    def export(self, file_path: str):
        """
        Export the current models and state to a file.

        Args:
            file_path (str): The path to the file where the models will be saved.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare the data to serialize
        data = {
            'models': self.models,
            'trained': self.trained,
            'window_size': self.window_size
        }

        # Use joblib to serialize the data
        joblib.dump(data, file_path)
        print(f"ScalePredictor state exported to {file_path}")

    def load(self, file_path: str):
        """
        Load models and state from a file.

        Args:
            file_path (str): The path to the file from which to load the models.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file: '{file_path}'")

        # Load the data using joblib
        data = joblib.load(file_path)

        # Restore the state
        self.models = data.get('models', {})
        self.trained = data.get('trained', False)
        self.window_size = data.get('window_size', 0)

        print(f"ScalePredictor state loaded from {file_path}")