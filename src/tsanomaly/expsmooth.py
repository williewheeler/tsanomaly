import numpy as np
import pandas as pd

from math import *


def _to_result(x, m, s, k):
    """Produces a data frame containing the anomaly detection results.
    
    Parameters
    ----------
    x : pd.Series
        Time series data
    m : pd.Series
        Mean estimate
    s : pd.Series
        Standard deviation estimate
    k : float
        Band multiplier
    
    Returns
    -------
    pd.DataFrame
        Data frame containing anomaly detection results
    """
    
    upper = m + k * s
    lower = m - k * s
    anom = (x < lower) | (x > upper)
    return pd.DataFrame({
        "x" : x,
        "mean" : m,
        "stdev" : s,
        "upper" : upper,
        "lower" : lower,
        "anomaly" : anom,
    })


# Shouldn't alpha=0.0 be permissible? May have to implement it myself.
def ewma(x, alpha, k, T=5):
    """
    Anomaly detection based on the exponentially weighted moving average.
    
    Parameters
    ----------
    x : pd.Series
        Time series data
    alpha : float
        Current weight in (0.0, 1.0]
    k : float
        Band multiplier
    T : int
        Training period
    
    Returns
    -------
    pd.DataFrame
        Data frame containing anomaly detection results
    """

    ewm = x.shift().ewm(alpha=alpha, min_periods=T)
    return _to_result(x, ewm.mean(), ewm.std(bias=True), k)


# FIXME Not correctly incorporating the training period.
# Notice that the stdev starts tight even with training.
def pewma(x, alpha, beta, k, T=5):
    """
    Anomaly detection based on the probabilistic exponentially weighted moving average.

    Parameters
    ----------
    x : pd.Series
        Time series data
    alpha : float
        Current weight in [0.0, 1.0]. Note that this is the reverse of what's in the
        paper (i.e., they use alpha for the history weight)
    beta : float
        Probability weight, applied to alpha
    k : float
        Band multiplier
    T : int
        Training period
    
    Returns
    -------
    pd.DataFrame
        Data frame containing anomaly detection results
    """
    PR_DENOM = sqrt(2.0 * pi)
    
    # Flip this to match pandas EWMA.
    # So alpha is the weight for the most recent value, and gamma
    # is the weight for the most recent estimate (i.e., history).
    gamma = 1.0 - alpha
    
    n = len(x)
    s1 = x[0]
    s2 = x[0] * x[0]
    m = np.repeat(x[0], n)
    s = np.repeat(0.0, n)
    
    for t in range(1, n):
        if t < T:
            gamma_t = 1.0 - 1.0 / t
        elif s[t-1] == 0:
            gamma_t = gamma
        else:
            # Use z-score to calculate probability
            z = (x[t-1] - m[t-1]) / s[t-1]
            p = np.exp(-0.5 * z * z) / PR_DENOM
            gamma_t = (1.0 - beta * p) * gamma
        
        s1 = gamma_t * s1 + (1.0 - gamma_t) * x[t-1]
        s2 = gamma_t * s2 + (1.0 - gamma_t) * x[t-1] * x[t-1]
        m[t] = s1
        s[t] = sqrt(s2 - s1 * s1)
        
    return _to_result(x, m, s, k)
