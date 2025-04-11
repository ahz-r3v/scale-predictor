from src.scale_predictor.utils import window_average, root_mean_squared_error
import unittest
import polars as pl
import numpy as np
import time
import os

class TestEvaluateWindowAverage(unittest.TestCase):
    def setUp(self):
        self.window_size = 600
        self.test_dir = os.path.dirname(__file__)
        self.warm10_test = os.path.join(self.test_dir, "data/warm10_test.csv")

    def test_window_average(self):
        # evaluate model accuracy on test set
        test_set_filename = self.warm10_test
        test_df = pl.read_csv(test_set_filename)

        trained_func_index = test_df['unique_id'].unique().to_list()
        print("Start evaluating model accuracy on test set...")

        accuracy_log = []
        total_predict_times = []

        for func_name in trained_func_index:
            df_func_test = test_df.filter(pl.col("unique_id") == func_name)
            if df_func_test.height < self.window_size + 100:
                print(f"Not enough data for function '{func_name}' to predict 100 steps, skipping...")
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
                start_time = time.perf_counter()
                prediction = window_average(600, window)
                desired_pod = prediction / 1
                predicted.append(desired_pod)
                end_time = time.perf_counter()

                elapsed_time = (end_time - start_time) * 1000  # to ms
                total_predict_time += elapsed_time

                window.pop(0)
                window.append(true_values_duplication.pop(0))

            if len(predicted) != 100:
                print(f"Function '{func_name}' prediction failed, unexpected length: {len(predicted)}")
                continue

            rmse = root_mean_squared_error(true_values, predicted)
            accuracy_log.append((func_name, rmse))
            print(f"Function '{func_name}' RMSE: {rmse:.2f} inference_time: {total_predict_time/100:.4f} ms")
            total_predict_times.append(total_predict_time/100)
            total_predict_time = 0.0
        # print(trained_func_index)

        if accuracy_log:
            avg_rmse = sum([score for _, score in accuracy_log]) / len(accuracy_log)
            avg_inference_time = sum(total_predict_times) / len(total_predict_times)
            print(f"Average RMSE over {len(accuracy_log)} functions: {avg_rmse:.2f}")
            print(f"max RMSE: {max([score for _, score in accuracy_log])}")
            print(f"midian RMSE: {np.median([score for _, score in accuracy_log])}")
            print(f"Average inference time: {avg_inference_time:.4f} ms")
        else:
            print("No functions evaluated successfully.")

        return 