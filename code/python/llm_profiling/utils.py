from constants import *

def _latency_to_string(latency_in_s, precision=2):
    if latency_in_s is None:
        return "None"
    day = 24 * 60 * 60
    hour = 60 * 60
    minute = 60
    ms = 1 / 1000
    us = 1 / 1000000
    if latency_in_s // day > 0:
        return str(round(latency_in_s / day, precision)) + " days"
    elif latency_in_s // hour > 0:
        return str(round(latency_in_s / hour, precision)) + " hours"
    elif latency_in_s // minute > 0:
        return str(round(latency_in_s / minute, precision)) + " minutes"
    elif latency_in_s > 1:
        return str(round(latency_in_s, precision)) + " s"
    elif latency_in_s > ms:
        return str(round(latency_in_s / ms, precision)) + " ms"
    else:
        return str(round(latency_in_s / us, precision)) + " us"
    
def _num_to_string(num, precision=2):
    if num is None:
        return "None"
    if num // 10**12 > 0:
        return str(round(num / 10.0**12, precision)) + " T"
    elif num // 10**9 > 0:
        return str(round(num / 10.0**9, precision)) + " G"
    elif num // 10**6 > 0:
        return str(round(num / 10.0**6, precision)) + " M"
    elif num // 10**3 > 0:
        return str(round(num / 10.0**3, precision)) + " K"
    else:
        return str(num)

def get_readable_summary_dict(summary_dict: dict, title="Summary") -> str:
    log_str = f"\n{title.center(PRINT_LINE_WIDTH, '-')}\n"
    for key, value in summary_dict.items():
        if "num_tokens" in key or "num_params" in key or "flops" in key:
            log_str += f"{key}: {_num_to_string(value)}\n"
        elif "gpu_hours" == key:
            log_str += f"{key}: {int(value)}\n"
        elif "memory" in key and "efficiency" not in key:
            log_str += f"{key}: {_num_to_string(value)}B\n"
        elif "latency" in key:
            log_str += f"{key}: {_latency_to_string(value)}\n"
        else:
            log_str += f"{key}: {value}\n"
    log_str += f"{'-' * PRINT_LINE_WIDTH}\n"
    return log_str

def within_range(val, target, tolerance):
    return abs(val - target) / target < tolerance

def average(lst):
    if not lst:
        return None
    return sum(lst) / len(lst)

def max_value(lst):
    if not lst:
        return None
    return max(lst)
