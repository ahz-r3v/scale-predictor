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
    # rearrange the array so that idx is at the end
    if idx < 0 or idx >= len(buckets) - 1:
        new_buckets = buckets
    else:
        new_buckets = buckets[idx+1:] + buckets[:idx+1]

    # if the last is invalid, check one element before
    for offset in range(1, 4):  # check -1, -2, -3
        if new_buckets[-offset] >= 0:
            # 旋转，使 new_buckets[-offset] 成为最后一个元素
            new_buckets = new_buckets[-offset:] + new_buckets[:-offset]
            break
    
    # fill the negative elements with 0
    for i in range(len(new_buckets)-1, -1, -1):
        if new_buckets[i] < 0:
            new_buckets[i] = 0

    return new_buckets

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))