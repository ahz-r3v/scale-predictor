import pandas as pd
import time
import logging
import os
import shutil
from collections import defaultdict
from typing import Dict, List, Any
from datetime import datetime, timedelta, date
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
import polars as pl


from src.scale_predictor.inverse_loss import MAELogMSEPenalizeLoss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss", 
    patience=10,
    mode="min", 
    verbose=True,
)

class NHITSModel:
    def __init__(self, window_size):
        self.models: Dict[str, Any] = {}
        self.window_size = window_size
        self.output_path = 'data/'
        self.model_path = 'models/' 
        self.cutoff = 0.02
        os.makedirs(self.model_path, exist_ok=True) 
        self.logger = logging.getLogger(__name__)
        # from 1970-01-01 01:00:00
        mod_time = datetime(1970, 1, 1, 1, 0, 0)
        mod_time_pd = pd.Timestamp(mod_time)
        self.ds_series = pd.date_range(start=mod_time_pd, periods=self.window_size, freq='s')


    def save_model(self, func_name):
        if func_name in self.models:
            model_filename = os.path.join(self.model_path, f"{func_name}.pkl")
            # with open(model_filename, "wb") as f:
            #     pickle.dump(self.models[func_name], f)
            self.models[func_name].save(model_filename)
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
                model = NHiTSModel.load(model_filename, map_location="cpu", pl_trainer_kwargs={"accelerator": "cpu", "logger": False})
                model.model.to("cpu")
                assert isinstance(model, NHiTSModel), "Loaded object is not a NHiTSModel!"
                self.models[func_name] = model
                self.logger.info(f"Model for function '{func_name}' loaded successfully!")
        return True, list(self.models.keys())

    def evaluate_model_batching(self, model_name, test_set_filename):
        test_df = pl.read_csv(test_set_filename)
        trained_func_index = test_df['unique_id'].unique().to_list()

        print("Start evaluating model accuracy on test set...")
        accuracy_log = []
        test_series_list = []
        
        for func_name in trained_func_index:
            df_func_test = test_df.filter(pl.col("unique_id") == func_name)
            if df_func_test.height < self.window_size:
                print(df_func_test)
                continue

            df_func_pd = df_func_test.to_pandas()
            ts_test = TimeSeries.from_dataframe(df_func_pd, time_col='ds', value_cols='y')
            # ts_test = ts_test[:-21]

            test_series_list.append((func_name, ts_test))

        print(trained_func_index)

        accuracy_log = []
        total_predict_times = []
        pred_list = {}
        for func_name, ts_test in test_series_list:
            predictions = []
            elapsed_time = 0
            for i in range(30):
                try:
                    start_time = time.perf_counter()
                    forecast = self.models[model_name].predict(n=10, series=ts_test[3+(i*10):603+(i*10)], verbose=False)
                    end_time = time.perf_counter()
                    elapsed_time += (end_time - start_time) * 1000  # to ms
                    # print(forecast.values())
                    predictions.extend(forecast.values().flatten())
                except Exception as e:
                    print(f"Function {func_name} evaluation failed: {e}")
            
            test_slice = ts_test[603:903]
            time_index = test_slice.time_index
            df_pred = pd.DataFrame({'value': predictions}, index=time_index)
            pred_list[func_name] = predictions
            pred_ts = TimeSeries.from_dataframe(df_pred)
            score = rmse(ts_test[603:903], pred_ts)
            accuracy_log.append((func_name, score))
            print(f"Function {func_name} RMSE: {score:.2f} inference_time: {elapsed_time/100:.4f} ms")
            total_predict_times.append(elapsed_time/100)

        if accuracy_log:
            avg_rmse = sum([score for _, score in accuracy_log]) / len(accuracy_log)
            avg_inference_time = sum(total_predict_times) / len(total_predict_times)
            print(f"Average RMSE over {len(accuracy_log)} functions: {avg_rmse:.2f}")
            print(f"max RMSE: {max([score for _, score in accuracy_log])}")
            print(f"Average inference time: {avg_inference_time:.4f} ms")
        else:
            print("No functions evaluated successfully.")

        import matplotlib.pyplot as plt
        for func_name, ts_test in test_series_list:
            try:
                forecast_values = pred_list[func_name]
                true_values = ts_test[603:903].values()
                # forecast_values = forecast.values()
                plt.figure(figsize=(10, 4))
                plt.plot(true_values, label="True")
                plt.plot(forecast_values, label="Forecast")
                plt.title(f"Function {func_name} Prediction vs True")
                plt.xlabel("Time Index")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.show()
                plt.savefig(f"tmp/{model_name} {func_name} output.png")
            except Exception as e:
                print(f"Function {func_name} plot failed: {e}")

    def evaluate_model(self, model_name, test_set_filename):
        test_df = pl.read_csv(test_set_filename)
        trained_func_index = test_df['unique_id'].unique().to_list()

        self.logger.info("Start evaluating model accuracy on test set...")
        accuracy_log = []
        test_series_list = []
        
        for func_name in trained_func_index:
            df_func_test = test_df.filter(pl.col("unique_id") == func_name)
            if df_func_test.height < self.window_size:
                print(df_func_test)
                continue

            df_func_pd = df_func_test.to_pandas()
            ts_test = TimeSeries.from_dataframe(df_func_test, time_col='ds', value_cols='y')
            ts_test = ts_test[:-21]

            test_series_list.append((func_name, ts_test))

        print(trained_func_index)

        accuracy_log = []
        total_predict_times = []
        for func_name, ts_test in test_series_list:
            predictions = []
            elapsed_time = 0
            for i in range(100):
                try:
                    start_time = time.perf_counter()
                    forecast = self.models[model_name].predict(n=5, series=ts_test[0+i:600+i], verbose=False)
                    end_time = time.perf_counter()
                    elapsed_time += (end_time - start_time) * 1000  # to ms
                    
                    prediction = forecast.values()[0][0]
                    predictions.append(prediction)
                except Exception as e:
                    self.logger.warning(f"Function {func_name} evaluation failed: {e}")
            
            test_slice = ts_test[600:700]
            time_index = test_slice.time_index
            df_pred = pd.DataFrame({'value': predictions}, index=time_index)
            pred_ts = TimeSeries.from_dataframe(df_pred)
            score = rmse(ts_test[600:700], pred_ts)
            accuracy_log.append((func_name, score))
            print(f"Function {func_name} RMSE: {score:.2f} inference_time: {elapsed_time/100:.4f} ms")
            total_predict_times.append(elapsed_time/100)

        if accuracy_log:
            avg_rmse = sum([score for _, score in accuracy_log]) / len(accuracy_log)
            avg_inference_time = sum(total_predict_times) / len(total_predict_times)
            print(f"Average RMSE over {len(accuracy_log)} functions: {avg_rmse:.2f}")
            print(f"max RMSE: {max([score for _, score in accuracy_log])}")
            print(f"Average inference time: {avg_inference_time:.4f} ms")
        else:
            self.logger.warning("No functions evaluated successfully.")

    # returns (success: bool, trained_func_index)
    def train_from_file(self, train_set_filename: str, val_set_filename: str, window_size: int) -> (bool, List[str]):
            self.window_size = window_size

            train_df = pd.read_csv(train_set_filename)
            train_df['ds'] = pd.to_datetime(train_df['ds'])
            val_df = pd.read_csv(val_set_filename)
            val_df['ds'] = pd.to_datetime(val_df['ds'])

            required_columns = {'unique_id', 'ds', 'y'}
            assert required_columns.issubset(train_df.columns)
            assert required_columns.issubset(val_df.columns)

            trained_func_index = train_df["unique_id"].astype(str).unique()
            func_num = len(trained_func_index)

            print(f"NHITS model start training! Function total num= {func_num}")

            train_series_list = []
            val_series_list = []

            for func_name in trained_func_index:
                df_train_func = train_df[train_df["unique_id"].astype(str) == func_name].copy()
                df_val_func = val_df[val_df["unique_id"].astype(str) == func_name].copy()
                # instruction of Darts TimeSeries
                if len(df_train_func) < window_size or len(df_val_func) < 10:
                    print(f"[Skipping] Not enough data for function {func_name}")
                    continue
                
                ts_train = TimeSeries.from_dataframe(df_train_func, time_col='ds', value_cols='y', freq='s')
                ts_val = TimeSeries.from_dataframe(df_val_func, time_col='ds', value_cols='y', freq='s')

                train_series_list.append(ts_train)
                val_series_list.append(ts_val)

            if not train_series_list:
                print("No valid series to train.")
                return False, []

            self.train_global_model(train_series_list, val_series_list)

            print(f"NHITS model trained succesfully! Function total num= {func_num}")
                    
            return True, ['global',]

    def train_global_model(self, train_series_list, val_series_list=None):
        print(f"NHITS model start training! Function name= global")

        custom_loss = MAELogMSEPenalizeLoss(a=10, np=2.0, up=1.2)

        model = NHiTSModel(
            input_chunk_length=self.window_size,
            output_chunk_length=10, # output = 10
            n_epochs=50,
            random_state=42,
            dropout=0.1,
            optimizer_kwargs={'weight_decay': 1e-5},
            batch_size=128,
            pl_trainer_kwargs={
                'accelerator': 'gpu',
                'enable_progress_bar': False,
                'logger': True,
                'callbacks': [early_stopping]
            },
            num_stacks=3,    
            num_blocks=2,   
            num_layers=2,
            layer_widths=512,

            loss_fn=custom_loss
        )

        if val_series_list:
            model.fit(train_series_list, val_series=val_series_list, verbose=True)
        else:
            model.fit(train_series_list, verbose=True)

        self.models['global'] = model
        print(f"NHITS model trained succesfully! Function name= global")
        self.save_model("global")


    def predict(self, func_name, window):
        if self.models is None or self.window_size is None:
            self.logger.warning("nhits predict fail: no model found!")
            return False, 0
        
        # if func_name not in self.models:
        #     self.logger.warning(f"nhits predict fail: no model found for function {func_name}")
        #     return False, 0

        if len(window) != self.window_size:
            self.logger.error(f"nhits predict fail: input window length should equals to window_size:{self.window_size}, but got {len(window)}")
            return False, 0

        # only take now().minute and now().second
        # ds_series = self.ds_series

        # input_df = pd.DataFrame({
        #     'unique_id': [func_name] * len(window),
        #     'ds': ds_series,
        #     'y': window
        # })
        start_time = time.perf_counter() 
        series = TimeSeries.from_series(pd.Series(window, index=self.ds_series))
        # with self.lock:
            # model = self.models[func_name]
            # forecast = model.predict(input_df)
        forecast = self.models['global_mixed_loss'].predict(n=10, series=series, verbose=False)
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000  # ms
        pid = os.getpid()
        self.logger.info(f"[PID: {pid}] nhits predict success: function {func_name}, prediction={forecast.values()[0][0]}, elapsed_time={elapsed_time_ms} ms")
        # result equals to the largest predicted value in the following 10s
        result =  max(forecast.values()[0])
        # deal with randomness
        if result < self.cutoff:
            result = 0
        # if have concurrency, should not return 0
        if window[-1] > 0.0 and result <= 0.0:
            result = 1.0
        return True, result


    