import numpy as np
import pandas as pd

from math import *

def ewma(x, alpha, m, T=5):
    """
    Anomaly detection based on the exponentially-weighted moving average.
    
    Parameters
    ----------
    x : a pandas series
    alpha : current vs history param [0.0, 1.0]
    m : band multiplier
    
    Returns
    -------
    Pandas DataFrame containing EWMA anomaly results
    """
    
    # Init
    n = len(x)    
    xhat = np.repeat(x[0], n)
    upper = np.repeat(x[0], n)
    lower = np.repeat(x[0], n)
    anom = np.repeat(False, n)    
    s1 = x[0]
    s2 = x[0] * x[0]
    
    for t in range(1, n):
        s1 = alpha * s1 + (1.0 - alpha) * x[t-1]
        s2 = alpha * s2 + (1.0 - alpha) * x[t-1] * x[t-1]
        xhat[t] = s1
        sigma_hat = sqrt(s2 - s1 * s1)
        upper[t] = xhat[t] + m * sigma_hat
        lower[t] = xhat[t] - m * sigma_hat
        if t > T:
            anom[t] = (x[t] > upper[t] or x[t] < lower[t])
            
    return pd.concat([
        pd.Series(x, name="x"),
        pd.Series(xhat, name="mean", index=x.index),
        pd.Series(upper, name="upper", index=x.index),
        pd.Series(lower, name="lower", index=x.index),
        pd.Series(anom, name="anomaly", index=x.index),
    ], axis=1)


def pewma(x, alpha, beta, m, T=5):
    """
    Anomaly detection based on the probabilistic exponentially-weighted moving average.

    Parameters
    ----------
    X : data
    alpha : maximal current vs history param [0.0, 1.0]
    beta : probability weight, applied to alpha
    m : band multiplier
    T : training period
    
    Returns
    -------
    Pandas DataFrame containing PEWMA anomaly results
    """
    PR_DENOM = sqrt(2.0 * pi)

    # Init
    n = len(x)
    xhat = np.repeat(x[0], n)
    upper = np.repeat(x[0], n)
    lower = np.repeat(x[0], n)
    anom = np.repeat(False, n)    
    s1 = x[0]
    s2 = x[0] * x[0]
    sigma_hat = 1.0
        
    for t in range(1, n):
        
        # Probabilistic alpha adjustment, during and after training period
        if t < T:
            alpha_t = 1.0 - 1.0 / t
        else:
            # Use z-score to calculate probability
            z = (x[t-1] - xhat[t-1]) / sigma_hat
            pr = np.exp(-0.5 * z * z) / PR_DENOM
            alpha_t = (1.0 - beta * pr) * alpha
        
        s1 = alpha_t * s1 + (1.0 - alpha_t) * x[t-1]
        s2 = alpha_t * s2 + (1.0 - alpha_t) * x[t-1] * x[t-1]        
        xhat[t] = s1
        sigma_hat = sqrt(s2 - s1 * s1)
        upper[t] = xhat[t] + m * sigma_hat
        lower[t] = xhat[t] - m * sigma_hat
        
        if t > T:
            anom[t] = (x[t] > upper[t] or x[t] < lower[t])
            
    return pd.concat([
        pd.Series(x, name="x"),
        pd.Series(xhat, name="mean", index=x.index),
        pd.Series(upper, name="upper", index=x.index),
        pd.Series(lower, name="lower", index=x.index),
        pd.Series(anom, name="anomaly", index=x.index),
    ], axis=1)


def plot_anomalies(ax, df, title, show_mean=True, band_color="k", band_alpha=0.2):
    anom = df.loc[df["anomaly"], "x"]
    ax.set_title(title)
    ax.plot(df["upper"], color=band_color, alpha=band_alpha, linestyle="--")
    ax.plot(df["lower"], color=band_color, alpha=band_alpha, linestyle="--")
    if show_mean:
        ax.plot(df["mean"], color=band_color, alpha=band_alpha, linestyle="--")
    ax.plot(df["x"], linestyle="-")
    ax.scatter(x=anom.index, y=anom.values, color="r")
