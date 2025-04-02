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
from torch.optim.lr_scheduler import StepLR

from inverse_loss import InverseValueMSELoss, InverseLogValueMSELoss, LogMSELoss, GaussianWeightedMSELoss, RelativeDiffLoss, MAELogMSEPenalizeLoss
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        
early_stopping = EarlyStopping(
    monitor="val_loss",       # 监控验证集 loss
    patience=10,               # 多少 epoch 没有提升就停止
    mode="min",               # 寻找最小 val_loss
    verbose=True,
)

class NHITSModel:
    def __init__(self, window_size):
        self.models: Dict[str, Any] = {}
        self.window_size = window_size
        self.output_path = 'data/'
        self.model_path = 'models/' 
        self.cutoff = 0
        os.makedirs(self.model_path, exist_ok=True) 
        mod_time = datetime(1970, 1, 1)
        mod_time_pd = pd.Timestamp(mod_time)
        self.ds_series = pd.date_range(end=mod_time_pd, periods=self.window_size, freq='s')

    def save_model(self, func_name):
        if func_name in self.models:
            model_filename = os.path.join(self.model_path, f"{func_name}.pkl")
            # with open(model_filename, "wb") as f:
            #     pickle.dump(self.models[func_name], f)
            self.models[func_name].save(model_filename)
            print(f"Model for function '{func_name}' saved successfully!")

    def train_global_model(self, train_series_list, val_series_list=None):
        print(f"NHITS model start training! Function name= global")
        # use reverse MSE Loss
        # custom_loss = InverseValueMSELoss(c=0.01)
        # custom_loss = InverseLogValueMSELoss(a=100.0, c=3.0)
        # custom_loss = LogMSELoss(a=100)
        # custom_loss = RelativeDiffLoss(a=100, c=0.1)
        # custom_loss = GaussianWeightedMSELoss(a=100, sigma=50, mu=0)
        custom_loss = MAELogMSEPenalizeLoss(a=10.0, np=2.0, up=2.0)
        # custom_loss = InversedMSEandLogMSEwithPenalizeLoss(a=100, c=100, np=2.0, up=1.2)

        model = NHiTSModel(
            input_chunk_length=self.window_size,
            output_chunk_length=10, # output = 10
            n_epochs=50,
            random_state=42,
            dropout=0.1,
            optimizer_kwargs={
                'lr': 1e-3,
                'weight_decay': 1e-5
            },
            lr_scheduler_cls=StepLR,
            lr_scheduler_kwargs={"step_size": 1, "gamma": 0.8},
            batch_size=128,
            pl_trainer_kwargs={
                'accelerator': 'gpu',
                'enable_progress_bar': False,
                'logger': True,
                'callbacks': [early_stopping]
            },
            num_stacks=3,       # 3 stacks
            num_blocks=2,       # each stack 1 block
            num_layers=2,       # each block 2 layers of FC
            layer_widths=256,    # layer width 128,64

            # loss_fn=custom_loss
        )

        if val_series_list:
            model.fit(train_series_list, val_series=val_series_list, verbose=True)
        else:
            model.fit(train_series_list, verbose=True)

        self.models['global'] = model
        print(f"NHITS model trained succesfully! Function name= global")
        self.save_model("global")

    def train_from_file(self, train_set_filename: str, val_set_filename: str, window_size: int) -> (bool, List[str]):
            # train_set_filename = 'data/train.csv'
            # train_set_filename = 'data/train_latter_half.csv'
            # test_set_filename = 'data/test.csv'
            self.window_size = window_size

            # self.generate_dataframe(filepath, train_set_filename, test_set_filename)
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

            # log_dir = "lightning_logs"
            # if os.path.exists(log_dir):
            #     shutil.rmtree(log_dir)
                    
            return True, ['global',]
    
    def load_model(self) -> (bool, List[str]):
        if not os.path.exists(self.model_path):
            print("No model found!")
            return False, []
        if not os.listdir(self.model_path):
            print("No model found!")
            return False, []
        for filename in os.listdir(self.model_path):
            if filename.endswith(".pkl"):
                func_name = filename.replace(".pkl", "")
                model_filename = os.path.join(self.model_path, filename)
                model = NHiTSModel.load(model_filename, map_location="cpu", pl_trainer_kwargs={"accelerator": "cpu", "logger": False})
                model.model.to("cpu")
                assert isinstance(model, NHiTSModel), "Loaded object is not a NHiTSModel!"
                self.models[func_name] = model
                print(f"Model for function '{func_name}' loaded successfully!")
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
                # 计算每个 10 秒窗口的最大值
                window_size = 10
                max_values = [max(forecast_values[i:i+window_size]) for i in range(0, len(forecast_values), window_size)]

                # 展开成等长的“阶梯线”
                max_line = []
                for val in max_values:
                    max_line.extend([val] * window_size)
                max_line = max_line[:len(forecast_values)]  # 截断多余部分（最后一段可能不足10）

                plt.plot(max_line, '--', label="Max (10s window)")

                plt.title(f"Function {func_name} Prediction vs True")
                plt.xlabel("Time Index")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                # plt.show()
                plt.savefig(f"{model_name} {func_name} output.png")


            except Exception as e:
                print(f"Function {func_name} plot failed: {e}")

    def generate_dataframe(self, file_path, train_set_filename, test_set_filename):
        self.logger.info(f"generate_dataframe (Polars): {file_path}")

        df = pl.read_csv(file_path, has_header=True)
        df = df.rename({'function': 'unique_id', 'cpu': 'y'})
        
        min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
        unique_ids = df['unique_id'].unique().to_list()
        all_timestamps = pl.DataFrame({'timestamp': range(min_time, max_time + 1)})

        full_index = all_timestamps.join(pl.DataFrame({'unique_id': unique_ids}), how='cross')
        merged = full_index.join(df, on=['timestamp', 'unique_id'], how='left').fill_null(0.0)
        merged = merged.with_columns([
            pl.col('timestamp').cast(pl.Int64),
            pl.col('y').cast(pl.Float64),
            pl.col('timestamp').cast(pl.Datetime(time_unit='ms')).alias('ds')
        ])

        # divide training / validation set
        train_dfs, test_dfs = [], []
        for uid in unique_ids:
            sub_df = merged.filter(pl.col('unique_id') == uid).sort('timestamp')
            timestamps = sub_df['timestamp'].unique()
            split_idx = int(0.8 * len(timestamps))
            split_time = timestamps[split_idx]
            train_dfs.append(sub_df.filter(pl.col('timestamp') < split_time))
            test_dfs.append(sub_df.filter(pl.col('timestamp') >= split_time))

        train_df = pl.concat(train_dfs)
        test_df = pl.concat(test_dfs)

        # save
        train_df.write_csv(train_set_filename)
        test_df.write_csv(test_set_filename)
        self.logger.info(f"Train size: {train_df.shape}, Test size: {test_df.shape}")
        return train_df, test_df
    
    # def auto_search_hyperparams(self, train_series_list, val_series_list):
    #     custom_loss = LogMSELoss(a=100)
    #     print("Start NHiTS hyperparameter search...")

    #     # 合并训练 + 验证，用于切分
    #     combined_series = train_series_list + val_series_list

    #     # 构造一个简单的 Splitter（用于内部交叉验证）
    #     splitter = Splitter(train_fraction=0.8)

    #     # 定义参数搜索空间
    #     param_space = {
    #         "model__num_stacks": [2, 3],
    #         "model__num_blocks": [1, 2, 3],
    #         "model__layer_widths": [64, 128, 256, 512],
    #         "model__dropout": [0.0, 0.1],
    #         "model__n_epochs": [30],
    #         "model__batch_size": [64],
    #         "model__loss_fn": [custom_loss],
    #     }

    #     # 搜索器：基于验证集性能（这里用 MAPE）选出最优模型
    #     gridsearch = GridSearch(
    #         model_class=NHiTSModel,
    #         parameters=param_space,
    #         forecast_horizon=10,
    #         splitter=splitter,
    #         metric=mape,
    #         verbose=True
    #     )

    #     best_model, best_params = gridsearch.fit(combined_series)

    #     print("🏆 最优超参数：", best_params)
    #     print("✅ 自动搜索完成，模型已训练完毕。")

    #     # 存下模型
    #     self.models['global'] = best_model
    #     self.save_model("global")
