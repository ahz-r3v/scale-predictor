import pandas as pd
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse
import polars as pl

train_val_preprocessed = "../preprocessed/full_0-60.csv"
test_preprocessed = "../preprocessed/full_60-80.csv"
train_out = "../out/full_train.csv"
val_out = "./out/full_val.csv"
test_out = "./out/full_test.csv"

def generate_train_val(file_path, train_set_filename, val_set_filename):
    print(f"generate_dataframe (Polars): {file_path}")

    df = pl.read_csv(file_path, has_header=True)
    df = df.rename({'function': 'unique_id', 'cpu': 'y'})    
    
    # min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
    min_time = 60
    max_time = 3600
    unique_ids = df['unique_id'].unique().to_list()
    all_timestamps = pl.DataFrame({'timestamp': range(min_time, max_time)})

    full_index = all_timestamps.join(pl.DataFrame({'unique_id': unique_ids}), how='cross')
    merged = full_index.join(df, on=['timestamp', 'unique_id'], how='left').fill_null(0.0)
    merged = merged.with_columns([
        pl.col('timestamp').cast(pl.Int64),
        pl.col('y').cast(pl.Float64),
        (pl.col('timestamp') * 1000).cast(pl.Datetime(time_unit='ms')).alias('ds')
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
    val_df = pl.concat(test_dfs)

    # save
    train_df.write_csv(train_set_filename)
    val_df.write_csv(val_set_filename)
    print(f"Train size: {train_df.shape}, Validation size: {val_df.shape}")
    return train_df, val_df

def generate_test(file_path, test_set_filename):
    print(f"generate_test_set (Polars): {file_path}")

    df = pl.read_csv(file_path, has_header=True)
    df = df.rename({'function': 'unique_id', 'cpu': 'y'})
    
    min_time = 3600
    max_time = 4800
    unique_ids = df['unique_id'].unique().to_list()
    all_timestamps = pl.DataFrame({'timestamp': range(min_time, max_time + 1)})

    full_index = all_timestamps.join(pl.DataFrame({'unique_id': unique_ids}), how='cross')
    merged = full_index.join(df, on=['timestamp', 'unique_id'], how='left').fill_null(0.0)
    merged = merged.with_columns([
        pl.col('timestamp').cast(pl.Int64),
        pl.col('y').cast(pl.Float64),
        (pl.col('timestamp') * 1000).cast(pl.Datetime(time_unit='ms')).alias('ds')
    ])

    # divide training / validation set
    test_dfs = []
    for uid in unique_ids:
        sub_df = merged.filter(pl.col('unique_id') == uid).sort('timestamp')
        timestamps = sub_df['timestamp'].unique()
        split_idx = 0
        split_time = timestamps[split_idx]
        test_dfs.append(sub_df.filter(pl.col('timestamp') >= split_time))

    test_df = pl.concat(test_dfs)

    # save
    test_df.write_csv(test_set_filename)
    print(f"Test size: {test_df.shape}")
    return test_df

generate_train_val(train_val_preprocessed, train_out, val_out)
generate_test(test_preprocessed, test_out)