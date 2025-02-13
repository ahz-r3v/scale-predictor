
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