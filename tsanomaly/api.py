import numpy as np
import pandas as pd

from math import *


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
    mu = ewm.mean()
    sigma = ewm.std()
    upper = mu + m * sigma
    lower = mu - m * sigma
    anom = (x <= lower) | (x >= upper)
    return pd.concat([x, mu, upper, lower, anom], axis=1) \
        .set_axis(["x", "mean", "upper", "lower", "anomaly"], axis=1)


def pewma(x, alpha, beta, m, T=5):
    """
    Anomaly detection based on the probabilistic exponentially-weighted moving average.

    Parameters
    ----------
    X : data
    alpha : current weight in [0.0, 1.0]
    beta : probability weight, applied to alpha
    m : band multiplier
    T : training period
    
    Returns
    -------
    Pandas DataFrame containing PEWMA anomaly results
    """
    PR_DENOM = sqrt(2.0 * pi)
    
    # Flip alpha to match the paper.
    # So now alpha is the history weight in [0.0, 1.0].
    alpha = 1.0 - alpha

    # Init
    n = len(x)
    mu = np.repeat(x[0], n)
    upper = np.repeat(x[0], n)
    lower = np.repeat(x[0], n)
    anom = np.repeat(False, n)    
    s1 = x[0]
    s2 = x[0] * x[0]
    sigma = 1.0
        
    for t in range(1, n):
        
        # Probabilistic alpha adjustment, during and after training period
        if t < T:
            alpha_t = 1.0 - 1.0 / t
        else:
            # Use z-score to calculate probability
            z = (x[t-1] - mu[t-1]) / sigma
            pr = np.exp(-0.5 * z * z) / PR_DENOM
            alpha_t = (1.0 - beta * pr) * alpha
        
        s1 = alpha_t * s1 + (1.0 - alpha_t) * x[t-1]
        s2 = alpha_t * s2 + (1.0 - alpha_t) * x[t-1] * x[t-1]        
        mu[t] = s1
        sigma = sqrt(s2 - s1 * s1)
        upper[t] = mu[t] + m * sigma
        lower[t] = mu[t] - m * sigma
        
        if t > T:
            anom[t] = (x[t] > upper[t] or x[t] < lower[t])
            
    return pd.DataFrame({
        "x" : x,
        "mean" : mu,
        "upper" : upper,
        "lower" : lower,
        "anomaly" : anom,
    })


def plot_anomalies(ax, df, title, show_mean=True, band_color="k", band_alpha=0.2):
    anom = df.loc[df["anomaly"], "x"]
    ax.set_title(title)
    ax.plot(df["upper"], color=band_color, alpha=band_alpha, linestyle="--")
    ax.plot(df["lower"], color=band_color, alpha=band_alpha, linestyle="--")
    if show_mean:
        ax.plot(df["mean"], color=band_color, alpha=band_alpha, linestyle="--")
    ax.plot(df["x"], linestyle="-")
    ax.scatter(x=anom.index, y=anom.values, color="r")
