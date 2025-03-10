import logging
from typing import List

def window_average(idx: int, buckets: list):
    """
    Calculates the average of consecutive non-zero intervals in the bucket array,
    Ignore leading and trailing zeros.
    If all buckets are zero, 0.0 is returned
    """
    if idx < 0 or idx >= len(buckets) - 1:
        new_buckets = buckets
    new_buckets = buckets[idx+1:] + buckets[:idx+1]

    first_nonzero = -1
    last_nonzero = -1
    total = 0.0

    for i, bucket in enumerate(new_buckets):
        if bucket != 0:
            total += bucket
            if first_nonzero == -1:
                first_nonzero = i
            last_nonzero = i

    if first_nonzero == -1:
        return 0.0

    valid_count = last_nonzero - first_nonzero + 1
    avg = total / valid_count
    return round(avg, 6)


def trim_window(idx: int, buckets: List[float]) -> List[float]:
    # if all 0, return directly
    # if all(bucket == 0 for bucket in buckets):
    #     return buckets
    if idx < 0 or idx >= len(buckets) - 1:
        new_buckets = buckets
    new_buckets = buckets[idx+1:] + buckets[:idx+1]
    last_idx = len(buckets)-1
    # find the last non-zero element
    for i in range(len(new_buckets)-1, -1, -1):
        if new_buckets[i] != 0:
            last_idx = i
            break
    # rotate the array so that the last non-zero element is at the end
    if last_idx == len(new_buckets)-1:
        return new_buckets
    ret_buckets = new_buckets[last_idx+1:] + new_buckets[:last_idx+1]
    # check if the window is outdated, +- 1 seconds' shift is allowed
    if last_idx < idx and idx - last_idx > 2:
        logger = logging.getLogger(__name__)
        logger.warning(f"window may be outdated: received_idx={idx}, last_valid_idx={last_idx}")
    
    return ret_buckets