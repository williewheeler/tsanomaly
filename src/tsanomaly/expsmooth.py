import numpy as np
import pandas as pd

from math import *


def _to_result(x, mu, sigma, m):
    upper = mu + m * sigma
    lower = mu - m * sigma
    anom = (x < lower) | (x > upper)
    return pd.DataFrame({
        "x" : x,
        "mean" : mu,
        "upper" : upper,
        "lower" : lower,
        "anomaly" : anom,
    })


def ewma(x, alpha, m, T=5):
    """
    Anomaly detection based on the exponentially-weighted moving average.
    
    Parameters
    ----------
    x : a pandas series
    alpha : current weight in (0.0, 1.0]
    m : band multiplier
    T : training period
    
    Returns
    -------
    Pandas DataFrame containing EWMA anomaly results
    """

    ewm = x.shift().ewm(alpha=alpha, min_periods=T)
    return _to_result(x, ewm.mean(), ewm.std(), m)


def pewma(x, alpha, beta, m, T=5):
    """
    Anomaly detection based on the probabilistic exponentially-weighted moving average.

    Parameters
    ----------
    X : data
    alpha : current weight in [0.0, 1.0]. Note that this is the reverse of what's in the paper
            (i.e., they use alpha for the history weight)
    beta : probability weight, applied to alpha
    m : band multiplier
    T : training period
    
    Returns
    -------
    Pandas DataFrame containing PEWMA anomaly results
    """
    PR_DENOM = sqrt(2.0 * pi)
    
    # Flip this to match pandas EWMA
    gamma = 1.0 - alpha
    
    n = len(x)
    s1 = x[0]
    s2 = x[0] * x[0]
    mu = np.repeat(x[0], n)
    sigma = np.repeat(0.0, n)
    
    for t in range(1, n):
        if t < T:
            gamma_t = 1.0 - 1.0 / t
        elif sigma[t-1] == 0:
            gamma_t = gamma
        else:
            # Use z-score to calculate probability
            z = (x[t-1] - mu[t-1]) / sigma[t-1]
            p = np.exp(-0.5 * z * z) / PR_DENOM
            gamma_t = (1.0 - beta * p) * gamma
        
        s1 = gamma_t * s1 + (1.0 - gamma_t) * x[t-1]
        s2 = gamma_t * s2 + (1.0 - gamma_t) * x[t-1] * x[t-1]
        mu[t] = s1
        sigma[t] = sqrt(s2 - s1 * s1)
        
    return _to_result(x, mu, sigma, m)