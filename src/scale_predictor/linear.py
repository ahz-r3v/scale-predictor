from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import joblib
from typing import Dict, List, Any
import logging
import os
import threading
import pandas as pd
import numpy as np
from collections import defaultdict
import polars as pl
import time
# from src.scale_predictor.utils import root_mean_squared_error

class LinearModel:
    def __init__(self, window_size):
        self.models: Dict[str, Any] = {}
        self.window_size = window_size
        self.model_path = 'models/'
        self.output_path = 'data/'
        self.cutoff = 0.005
        os.makedirs(self.model_path, exist_ok=True) 
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()

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

    def evaluate_model(self, model_name, test_set_filename):
        test_df = pl.read_csv(test_set_filename)
        trained_func_index = test_df['unique_id'].unique().to_list()
        self.logger.info("Start evaluating model accuracy on test set...")

        accuracy_log = []
        total_predict_times = []

        for func_name in trained_func_index:
            df_func_test = test_df.filter(pl.col("unique_id") == func_name)
            if df_func_test.height < self.window_size + 100:
                self.logger.warning(f"Not enough data for function '{func_name}' to predict 100 steps, skipping...")
                continue

            df_func_pd = df_func_test.to_pandas()
            y_series = df_func_pd["y"].values
            y_series = y_series[:-21] # remove tail dirty data

            window = y_series[:self.window_size:].tolist()
            true_values = y_series[self.window_size:self.window_size+100].tolist()
            true_values_duplication = true_values.copy()

            predicted = []
            
            total_predict_time = 0.0
            for _ in range(100):
                model = self.models[model_name]
                start_time = time.perf_counter()
                prediction = model.predict([window])[0]
                predicted.append(prediction)
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) * 1000  # to ms
                total_predict_time += elapsed_time
                
                window.pop(0)
                window.append(true_values_duplication.pop(0))

            if len(predicted) != 100:
                self.logger.warning(f"Function '{func_name}' prediction failed, unexpected length: {len(predicted)}")
                continue

            rmse = root_mean_squared_error(true_values, predicted)
            accuracy_log.append((func_name, rmse))
            self.logger.info(f"Function '{func_name}' RMSE: {rmse:.2f}")
            print(f"Function '{func_name}' RMSE: {rmse:.2f} inference_time: {total_predict_time/100:.4f} ms")

            total_predict_times.append(total_predict_time/100)
            total_predict_time = 0.0


        if accuracy_log:
            avg_rmse = sum([score for _, score in accuracy_log]) / len(accuracy_log)
            avg_inference_time = sum(total_predict_times) / len(total_predict_times)
            print(f"Average RMSE over {len(accuracy_log)} functions: {avg_rmse:.2f}")
            print(f"max RMSE: {max([score for _, score in accuracy_log])}")
            print(f"midian RMSE: {np.median([score for _, score in accuracy_log])}")
            print(f"Average inference time: {avg_inference_time:.4f} ms")
        else:
            self.logger.warning("No functions evaluated successfully.")

    def evaluate_model_batching(self, model_name, test_set_filename):
        test_df = pl.read_csv(test_set_filename)
        trained_func_index = test_df['unique_id'].unique().to_list()

        print("Start evaluating Linear model accuracy on test set...")
        accuracy_log = []
        total_predict_times = []
        pred_list = {}
        test_series_list = []

        for func_name in trained_func_index:
            df_func_test = test_df.filter(pl.col("unique_id") == func_name)
            if df_func_test.height < self.window_size + 60:
                print(f"[WARN] {func_name} skipped due to insufficient data")
                continue

            df_func_pd = df_func_test.to_pandas()
            y_series = df_func_pd["y"].values
            time_index = pd.to_datetime(df_func_pd["ds"])

            window = y_series[:self.window_size].tolist()
            true_values = y_series[self.window_size:self.window_size + 60].tolist()
            true_values_dup = true_values.copy()

            predicted = []
            total_predict_time = 0.0

            for _ in range(6):  # 6 * 10s = 60s
                start = time.perf_counter()
                prediction = self.models[model_name].predict([window])[0]  # shape: (10,)
                end = time.perf_counter()
                elapsed = (end - start) * 1000  # ms
                total_predict_time += elapsed

                predicted.extend(prediction)
                window = window[10:] + true_values_dup[:10]
                true_values_dup = true_values_dup[10:]

            if len(predicted) != len(true_values):
                print(f"[ERROR] {func_name} prediction mismatch. Skipping...")
                continue

            rmse = root_mean_squared_error(true_values, predicted)
            accuracy_log.append((func_name, rmse))
            total_predict_times.append(total_predict_time / 6)
            pred_list[func_name] = predicted
            test_series_list.append((func_name, true_values))

            print(f"Function {func_name} RMSE: {rmse:.2f}, Avg Inference Time: {total_predict_time/6:.2f} ms")

        if accuracy_log:
            avg_rmse = np.mean([score for _, score in accuracy_log])
            avg_time = np.mean(total_predict_times)
            print(f"Average RMSE over {len(accuracy_log)} functions: {avg_rmse:.2f}")
            print(f"Max RMSE: {np.max([score for _, score in accuracy_log]):.2f}")
            print(f"Median RMSE: {np.median([score for _, score in accuracy_log]):.2f}")
            print(f"Average Inference Time: {avg_time:.2f} ms")
        else:
            print("No functions evaluated successfully.")

        # Plot
        import matplotlib.pyplot as plt
        for func_name, true_values in test_series_list:
            try:
                forecast_values = pred_list[func_name]
                plt.figure(figsize=(10, 4))
                plt.plot(true_values, label="True")
                plt.plot(forecast_values, label="Forecast")
                plt.title(f"Function {func_name} Prediction vs True")
                plt.xlabel("Time Index")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{model_name}_{func_name}_linear_output.png")
                plt.close()
            except Exception as e:
                print(f"Function {func_name} plot failed: {e}")

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

    def train_from_file(self, train_set_filename: str, window_size: int) -> (bool, List[str]):
        # train_set_filename = 'data/train.csv'
        # test_set_filename = 'data/test.csv'
        self.window_size = window_size

        # self.generate_dataframe(filepath, train_set_filename, test_set_filename)
        print("handling data...")
        train_df = pd.read_csv(train_set_filename)
        train_df['ds'] = pd.to_datetime(train_df['ds'])
        # test_df = pd.read_csv(test_set_filename)
        # test_df['ds'] = pd.to_datetime(test_df['ds'])
        
        required_columns = {'unique_id', 'ds', 'y'}
        assert required_columns.issubset(train_df.columns)
        
        trained_func_index = train_df["unique_id"].astype(str).unique()
        func_num = len(trained_func_index)
        
        self.logger.info(f"Linear Regression model start training! Function total num= {func_num}")

        print("generating series list...")
        series_list = []
        # for func_name in trained_func_index:
        #     df_func = train_df[train_df["unique_id"].astype(str) == func_name].copy()
        #     series_list.append(df_func)
            # self.train_one_model(func_name, func_data)
        func_name_set = set(map(str, trained_func_index))
        filtered_df = train_df[train_df["unique_id"].astype(str).isin(func_name_set)].copy()
        series_list = [group for _, group in filtered_df.groupby("unique_id")]
        print("training global model...")
        self.train_global_model(series_list)

        self.logger.info(f"Linear Regression model trained successfully! Function total num= {func_num}")
        
        return True, ['global',]
    
    def train_global_model(self, series_list: List[pd.DataFrame]) -> bool:
        self.logger.info(f"Linear Regression model start training! Function name= global")

        all_X, all_y = [], []

        for df_func in series_list:
            call_list = df_func["y"].values.tolist()
            func_name = df_func["unique_id"].values[0]
            
            if len(call_list) < self.window_size + 1:
                self.logger.warning(f"Not enough data to train model for function '{func_name}', skipping...")
                continue

            X, y = [], []
            for i in range(len(call_list) - self.window_size - 9):
                X.append(call_list[i : i + self.window_size])
                y.append(call_list[i + self.window_size : i + self.window_size + 10])
            all_X.extend(X)
            all_y.extend(y)

        if not all_X or not all_y:
            self.logger.warning(f"Not enough data to train global model, skipping...")
            return False
        
        model = LinearRegression()
        model.fit(all_X, all_y)

        self.models['global'] = model
        self.save_model('global')
        self.logger.info(f"Linear Regression model trained successfully! Function name= global")
        return True

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

        if len(window) != self.window_size:
            self.logger.error(f"linear predict fail: input window length should equals to window_size:{self.window_size}, but got {len(window)}")
            return False, 0

        # with self.locks[func_name]:
        model = self.models['global_400_b10_lr']
        start_time = time.perf_counter()
        prediction = model.predict([window])[0]
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000  # to ms
        pid = os.getpid()
        self.logger.info(f"[PID: {pid}] linear predict success: function {func_name}, prediction={prediction[0]}, elapsed_time={elapsed_time} ms")
        # predict 10s ahead
        result = max(prediction)
        # handle randomness with cutoff
        if result < self.cutoff and window[-1] == 0:
            result = 0
        return True, result


    
   