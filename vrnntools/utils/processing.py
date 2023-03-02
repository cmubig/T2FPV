import numpy as np
import torch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_winters_smoothing(series, alpha=0.5, beta=0.5, simple=False):
    '''
    Holt-Winters double exponential smoothing
    Assume the time steps are regular (with same interval)
    If simple=True, it becomes basic (simple) exponential smoothing (Holt linear)
    '''
    level = series[0]
    # trend = series[1] - series[0]
    trend = 0 # This initialization is used for delta_x smoothing
    if simple==True:
        trend = 0
        beta = 0
    results = torch.zeros(series.shape).to(series.device)
    results[0] = level
    for i in range(1, len(series)):
        previous_level = level
        previous_trend = trend
        level = alpha * series[i] + (1 - alpha) * (previous_level + previous_trend)
        trend = beta * (level - previous_level) + (1 - beta) * previous_trend
        results[i] = level
    return results