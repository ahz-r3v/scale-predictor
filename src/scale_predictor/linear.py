from sklearn.linear_model import LinearRegression
import joblib
from typing import Dict, List, Any
import logging
import os
import threading
import pandas as pd
import numpy as np
from collections import defaultdict

class LinearModel:
    def __init__(self, window_size):
        self.models: Dict[str, Any] = {}
        self.window_size = window_size
        self.model_path = 'models/'
        self.output_path = 'data/'
        os.makedirs(self.model_path, exist_ok=True) 
        self.logger = logging.getLogger(__name__)
        self.locks = defaultdict(threading.Lock)

    def save_model(self, func_name):
        if func_name in self.models:
            file_path = os.path.join(self.model_path, f"{func_name}.lpkl")
            joblib.dump(self.models[func_name], file_path)    # linear-pkl
            self.logger.info(f"Model for function '{func_name}' saved successfully!")

    
    def load_model(self) -> (bool, List[str]):
        if not os.path.exists(self.model_path):
            self.logger.warning("No linear model found!")
            return False, []
        if not os.listdir(self.model_path):
            self.logger.warning("No linear model found!")
            return False, []
        for filename in os.listdir(self.model_path):
            if filename.endswith(".lpkl"):
                func_name = filename.replace(".lpkl", "")
                model_filename = os.path.join(self.model_path, filename)
                self.models[func_name] = joblib.load(model_filename)
                self.logger.info(f"Model for function '{func_name}' loaded successfully!")
        return True, list(self.models.keys())


    def generate_dataframe(self, file_path, train_set_filename, test_set_filename):
        self.logger.info(f"generate_dataframe: file_path={file_path}, train_set_filename={train_set_filename}, test_set_filename={test_set_filename}")
        df = pd.read_csv(file_path, names=['timestamp', 'function', 'cpu'], dtype={'timestamp': int, 'function': str, 'cpu': float}, skiprows=1)

        # trim timestamps to [min_time, max_time]
        min_time = df['timestamp'].min()
        max_time = df['timestamp'].max()
        all_timestamps = pd.DataFrame({'timestamp': range(min_time, max_time + 1)})

        unique_ids = df['function'].unique()
        full_data = pd.DataFrame()

        for uid in unique_ids:
            sub_df = df[df['function'] == uid]
            uid_timestamps = all_timestamps.copy()
            uid_timestamps['function'] = uid 
            # fill missing 'cpu' to 0
            merged_df = uid_timestamps.merge(sub_df, on=['timestamp', 'function'], how='left').fillna({'cpu': 0})
            full_data = pd.concat([full_data, merged_df], ignore_index=True)

        full_data = full_data.sort_values(['function', 'timestamp']).reset_index(drop=True)
        full_data.rename(columns={'function': 'unique_id', 'cpu': 'y'}, inplace=True)
        full_data['ds'] = pd.to_datetime(full_data['timestamp'], unit='s')

        # 80% train, 20% test
        train = pd.DataFrame()
        test = pd.DataFrame()

        for uid in unique_ids:
            sub_df = full_data[full_data['unique_id'] == uid]
            unique_timestamps = sub_df['timestamp'].unique()
            train_size = int(0.8 * len(unique_timestamps))

            train = pd.concat([train, sub_df[sub_df['timestamp'] < unique_timestamps[train_size]]], ignore_index=True)
            test = pd.concat([test, sub_df[sub_df['timestamp'] >= unique_timestamps[train_size]]], ignore_index=True)

        self.logger.info(f"train set size: {train.shape}")
        self.logger.info(f"test set size: {test.shape}")

        train.to_csv(train_set_filename, index=False)
        test.to_csv(test_set_filename, index=False)

        self.logger.info(f"generate_dataframe successfully!")
        return train, test

    def train_from_file(self, filepath: str, window_size: int) -> (bool, List[str]):
        train_set_filename = 'data/train.csv'
        test_set_filename = 'data/test.csv'
        self.window_size = window_size

        self.generate_dataframe(filepath, train_set_filename, test_set_filename)
        train = pd.read_csv(train_set_filename)
        train['ds'] = pd.to_datetime(train['ds'])
        
        required_columns = {'unique_id', 'ds', 'y'}
        assert required_columns.issubset(train.columns)
        
        trained_func_index = train["unique_id"].astype(str).unique()
        func_num = len(trained_func_index)
        
        self.logger.info(f"Linear Regression model start training! Function total num= {func_num}")

        for func_name in trained_func_index:
            func_data = train[train["unique_id"].astype(str) == func_name]
            self.train_one_model(func_name, func_data)
        
        self.logger.info(f"Linear Regression model trained successfully! Function total num= {func_num}")
        return True, list(trained_func_index)

    def train_one_model(self, func_name: str, train: pd.DataFrame):
        self.logger.info(f"Linear Regression model start training! Function name= {func_name}")
        
        call_list = train["y"].values.tolist()
        if len(call_list) < self.window_size + 1:
            self.logger.warning(f"Not enough data to train model for function '{func_name}', skipping...")
            return
        
        X, y = [], []
        for i in range(len(call_list) - self.window_size):
            X.append(call_list[i : i + self.window_size])
            y.append(call_list[i + self.window_size])
        
        model = LinearRegression()
        model.fit(X, y)
        
        self.models[func_name] = model
        self.logger.info(f"Linear Regression model trained successfully! Function name= {func_name}")
        self.save_model(func_name)

    def predict(self, func_name, window):
        if self.models is None or self.window_size is None:
            self.logger.warning("linear predict fail: no model found!")
            return False, 0
        
        if func_name not in self.models:
            self.logger.warning(f"linear predict fail: no model found for function {func_name}")
            return False, 0

        if len(window) != self.window_size:
            self.logger.error(f"linear predict fail: input window length should equals to window_size:{self.window_size}, but got {len(window)}")
            return False, 0

        with self.locks[func_name]:
            model = self.models[func_name]
            prediction = model.predict([window])
        pid = os.getpid()
        self.logger.info(f"[PID: {pid}] linear predict success: function {func_name}, prediction={prediction[0]}")
        return True, prediction[0]
