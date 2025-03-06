import pandas as pd
import pickle
import logging
import os
import shutil
import threading
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime, timedelta, date
from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast
from neuralforecast.utils import AirPassengersDF
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

class NHITSModel:
    def __init__(self, window_size):
        self.model: Dict[str, Any] = {}
        self.window_size = window_size
        self.output_path = 'data/'
        self.model_path = 'models/' 
        os.makedirs(self.model_path, exist_ok=True) 
        self.logger = logging.getLogger(__name__)
        self.locks = defaultdict(threading.Lock)


    def save_model(self, func_name):
        with self.locks[func_name]:
            if func_name in self.model:
                model_filename = os.path.join(self.model_path, f"{func_name}.pkl")
                with open(model_filename, "wb") as f:
                    pickle.dump(self.model[func_name], f)
                self.logger.info(f"Model for function '{func_name}' saved successfully!")

    def load_model(self) -> (bool, List[str]):
        if not os.path.exists(self.model_path):
            self.logger.warning("No model found!")
            return False, []
        if not os.listdir(self.model_path):
            self.logger.warning("No model found!")
            return False, []
        for filename in os.listdir(self.model_path):
            if filename.endswith(".pkl"):
                func_name = filename.replace(".pkl", "")
                model_filename = os.path.join(self.model_path, filename)
                with open(model_filename, "rb") as f:
                    self.model[func_name] = pickle.load(f)
                self.logger.info(f"Model for function '{func_name}' loaded successfully!")
        return True, list(self.model.keys())


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

    # returns (success: bool, trained_func_index)
    def train_from_file(self, filepath: str, window_size: int) -> (bool, List[str]):
        train_set_filename = 'data/train.csv'
        test_set_filename = 'data/test.csv'
        self.window_size = window_size

        self.generate_dataframe(filepath, train_set_filename, test_set_filename)
        train = pd.read_csv(train_set_filename)
        train['ds'] = pd.to_datetime(train['ds'])

        required_columns = {'unique_id', 'ds', 'y'}
        assert required_columns.issubset(train.columns)

        trained_func_index = unique_ids = train["unique_id"].astype(str).unique()
        func_num = len(trained_func_index)

        self.logger.info(f"NHITS model start training! Function total num= {func_num}")

        for func_name in trained_func_index:
            func_data = train[train["unique_id"].astype(str) == func_name]
            self.train_one_model(func_name, func_data)

        self.logger.info(f"NHITS model trained succesfully! Function total num= {func_num}")

        log_dir = "lightning_logs"
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        return True, trained_func_index
    
    def train_one_model(self, func_name, train):
        self.logger.info(f"NHITS model start training! Function name= {func_name}")
        model = NHITS(h=1, input_size=self.window_size, loss=MAE(), max_steps=100)
        nf = NeuralForecast(models=[model], freq='s')
        nf.fit(train)

        with self.locks[func_name]:
            self.model[func_name] = nf

        self.logger.info(f"NHITS model trained succesfully! Function name= {func_name}")
        self.save_model(func_name)


    def predict(self, func_name, window):
        if self.model is None or self.window_size is None:
            self.logger.warning("nhits predict fail: no model found!")
            return False, 0
        
        if func_name not in self.model:
            self.logger.warning(f"nhits predict fail: no model found for function {func_name}")
            return False, 0

        if len(window) != self.window_size:
            self.logger.error(f"nhits predict fail: input window length should equals to window_size:{self.window_size}, but got {len(window)}")
            return False, 0

        # only take now().minute and now().second
        now = datetime.now()
        mod_now = now.minute * 60 + now.second
        mod_time = datetime(1970, 1, 1) + timedelta(seconds=mod_now)
        mod_time_pd = pd.Timestamp(mod_time)
        ds_series = pd.date_range(end=mod_time_pd, periods=len(window), freq='s')

        # 构造 DataFrame
        input_df = pd.DataFrame({
            'unique_id': [func_name] * len(window),
            'ds': ds_series,
            'y': window
        })

        with self.locks[func_name]:
            model = self.model[func_name]
            forecast = model.predict(input_df)
        pid = os.getpid()
        self.logger.info(f"[PID: {pid}] nhits predict successfully!")
        return True, forecast.iloc[0]['NHITS']


    