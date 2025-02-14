
def window_average(start_idx: int, buckets: list, smoothing_coeff: float) -> float:
    total_b = len(buckets)
    num_b = len(buckets)
    multiplier = smoothing_coeff
    
    result = 0.0
    
    for i in range(num_b):
        effective_idx = (start_idx - i) % total_b
        v = buckets[effective_idx] * multiplier
        result += v
        multiplier *= (1 - smoothing_coeff)
    
    return result

def window_average(idx: int, buckets: list):
    """
    Calculates the average of consecutive non-zero intervals in the bucket array,
    Ignore leading and trailing zeros.
    If all buckets are zero, 0.0 is returned
    """
    first_nonzero = -1
    last_nonzero = -1
    total = 0.0

    for i, bucket in enumerate(buckets):
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