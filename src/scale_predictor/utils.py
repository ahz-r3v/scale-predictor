import logging
from typing import List
import numpy as np

def window_average(idx: int, buckets: list):
    """
    Calculates the average of consecutive non-zero intervals in the bucket array,
    Ignore leading and trailing zeros.
    If all buckets are zero, 0.0 is returned
    """
    if idx < 0 or idx >= len(buckets) - 1:
        new_buckets = buckets
    else: 
        new_buckets = buckets[idx+1:] + buckets[:idx+1]

    first_notneg = -1
    last_notneg = -1
    total = 0.0

    for i, bucket in enumerate(new_buckets):
        if bucket >= 0:
            total += bucket
            if first_notneg == -1:
                first_notneg = i
            last_notneg = i

    if first_notneg == -1:
        return 0.0

    valid_count = last_notneg - first_notneg + 1
    avg = total / valid_count
    return round(avg, 6)


def trim_window(idx: int, buckets: List[float]) -> List[float]:
    if idx < 0 or idx >= len(buckets) - 1:
        new_buckets = buckets
    else:
        new_buckets = buckets[idx+1:] + buckets[:idx+1]
    last_idx = -1
    # find the last valid (non-negtive) element
    for i in range(len(new_buckets)-1, -1, -1):
        if new_buckets[i] >= 0 and last_idx == -1:
            last_idx = i
        if new_buckets[i] < 0:
            new_buckets[i] = 0
    # rearrange the array so that the last non-negtive element is at the end
    if last_idx == len(new_buckets)-1:
        return new_buckets
    ret_buckets = new_buckets[last_idx+1:] + new_buckets[:last_idx+1]
    return ret_buckets

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))