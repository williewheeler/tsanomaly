import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import *

def ewma(x, alpha, m):
    """
    Anomaly detection based on the exponentially-weighted moving average.
    
    Parameters
    ----------
    x : a pandas series
    alpha : current vs history param [0.0, 1.0]
    m : band width param
    
    Returns
    -------
    TODO
    """
    
    n = len(x)    
    xhat = np.repeat(None, n)
    upper = np.repeat(None, n)
    lower = np.repeat(None, n)
    anom = np.repeat(False, n)

    xhat[0] = x[0]
    s1 = x[0]
    s2 = x[0] * x[0]
    
    for i in range(1, n):
        xhat[i] = alpha * xhat[i-1] + (1.0-alpha) * x[i]
        sigma_hat = sqrt(s2 - s1 * s1)
        
        upper[i] = xhat[i] + m * sigma_hat
        lower[i] = xhat[i] - m * sigma_hat
        anom[i] = (x[i] > upper[i] or x[i] < lower[i])
        
        s1 = alpha * s1 + (1.0-alpha) * x[i]
        s2 = alpha * s2 + (1.0-alpha) * x[i] * x[i]
    
    return pd.concat([
        pd.Series(x, name="x"),
        pd.Series(xhat, name="mean", index=x.index),
        pd.Series(upper, name="upper", index=x.index),
        pd.Series(lower, name="lower", index=x.index),
        pd.Series(anom, name="anomaly", index=x.index),
    ], axis=1)


def pewma():
    return None


def plot_anomalies(df):
    anom = df.loc[df["anomaly"], "x"]
    fig = plt.figure(figsize=(14, 2))
    ax = fig.add_subplot()
    ax.plot(df["x"], linestyle="-")
    ax.plot(df["mean"], color="k", alpha=0.2, linestyle="--")
    ax.plot(df["upper"], color="k", alpha=0.2, linestyle="--")
    ax.plot(df["lower"], color="k", alpha=0.2, linestyle="--")
    ax.scatter(x=anom.index, y=anom.values, color="r")
    plt.show()
