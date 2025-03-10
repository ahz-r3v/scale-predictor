import numpy as np
import math
from typing import Dict, List, Any
from sklearn.linear_model import LinearRegression
import joblib
import os
import logging
import re
import src.scale_predictor.utils as utils
from src.scale_predictor.nhits import NHITSModel
from src.scale_predictor.linear import LinearModel

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

    def __init__(self, model_selector):
        self.models: Dict[str, Any] = {}
        self.trained: bool = False
        self.window_size: int = 0
        self.model_selector = model_selector
        self.smoothing_coeff = 0.6
        self.logger = logging.getLogger(__name__)
        self.nhitsmdl = None
        self.linearmdl = None
        match self.model_selector:
            case "nhits":
                self.nhitsmdl = NHITSModel(60)
                # try loading models
                succ, loaded_func_names = self.nhitsmdl.load_model()
                if succ:
                    for func_name in loaded_func_names:
                        self.models[func_name] = self.nhitsmdl
                        self.logger.info(f"Loaded nhits model: '{func_name}'")
                    self.trained = True
            case "linear":
                self.linearmdl = LinearModel(60)
                # try loading models
                succ, loaded_func_names = self.linearmdl.load_model()
                if succ:  
                    for func_name in loaded_func_names:
                        self.models[func_name] = self.linearmdl
                        self.logger.info(f"Loaded linear model: '{func_name}'")
                    self.trained = True
            case "historical":
                pass
            case "default":
                pass

    def train_by_file(self, path: str, window_size: int):
        self.window_size = window_size
        match self.model_selector:
            case "nhits":
                succ, trained_func_names = self.nhitsmdl.train_from_file(path, window_size)
                if not succ:
                    return False
                for func_name in trained_func_names:
                    self.models[func_name] = self.nhitsmdl
                    self.logger.info(f"Trained nhits model: function '{func_name}'")
                self.trained = True
                return True
            case "linear":
                succ, trained_func_names = self.linearmdl.train_from_file(path, window_size)
                if not succ:
                    return False
                for func_name in trained_func_names:
                    self.models[func_name] = self.linearmdl
                    self.logger.info(f"Trained linear model: function '{func_name}'")
                self.trained = True
                return True
            case "historical":
                return True
            case "default":
                return True



    def train(self, dataset: Dict[str, List[float]], window_size: int) -> bool:
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

    def predict(self, function_name: str, window: List[float], index: int) -> float:
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
        # If not trained, or no model for the function, or window size is 0, use window_average.
        match = re.search(r"trace-func-(\d+)", function_name)
        func_index = match.group(1) if match else None
        self.logger.info(f"function name: '{function_name}'   func_index={func_index}   model_selector={self.model_selector}")
        if func_index is None:
            self.logger.warning(f"Invalid function name '{function_name}', use default window_average.")
            predicted_next = utils.window_average(index, window)
            self.logger.info(f"predicted pod count = {predicted_next}")
            self.logger.debug("predict returns")
            if predicted_next < 0:
                return 0
            return predicted_next
        if not self.trained or func_index not in self.models:
            self.logger.warning(f"No trained model found for function '{function_name}', use default window_average.")
            predicted_next = utils.window_average(index, window)
            self.logger.info(f"predicted pod count = {predicted_next}")
            self.logger.debug("predict returns")
            if predicted_next < 0:
                return 0
            return predicted_next

        # Auto-choose model
        model = self.models[func_index]

        # Rearrange the windows according to the index to ensure order. Spin and split window.
        sequenced_window = list(utils.trim_window(index, window))

        match self.model_selector:
            case "nhits":
                succ, predicted_next = model.predict(func_index, sequenced_window)
                # if failed, fall back to window_average
                if not succ:
                    self.logger.warning(f"NHiTS predict failed for function '{function_name}', use default window_average.")
                    predicted_next = utils.window_average(index, window)
            case "linear":
                succ, predicted_next = model.predict(func_index, sequenced_window)
                # if failed, fall back to window_average
                if not succ:
                    self.logger.warning(f"Linear predict failed for function '{function_name}', use default window_average.")
                    predicted_next = utils.window_average(index, window)
            case "default":
                # Predict next second. Round up to an integer.
                predicted_next = utils.window_average(index, window)

        self.logger.info(f"predicted pod count = {predicted_next}")
        self.logger.debug("predict returns")

        if predicted_next < 0:
            return 0
        
        return predicted_next

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