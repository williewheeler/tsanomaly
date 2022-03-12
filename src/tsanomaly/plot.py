def plot_anomalies(
    ax,
    df,
    title=None,
    show_x=True,
    show_band=True,
    x_color="k",
    band_color="gray",
    band_alpha=0.5):
    
    """Plots mean, bounds, observations and anomalies.
    """
    
    anom = df.loc[df["anomaly"], "x"]
    if title:
        ax.set_title(title)
    if show_band:
        ax.plot(df["upper"], color=band_color, alpha=band_alpha, linestyle="--")
        ax.plot(df["lower"], color=band_color, alpha=band_alpha, linestyle="--")
        ax.plot(df["mean"], color=band_color, alpha=band_alpha, linestyle="--")
    if show_x:
        ax.plot(df["x"], linestyle="-")
    ax.scatter(x=anom.index, y=anom.values, color="r")

