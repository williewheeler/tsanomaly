def plot_anomalies(ax, df, title, show_mean=True, band_color="k", band_alpha=0.2):
    anom = df.loc[df["anomaly"], "x"]
    ax.set_title(title)
    ax.plot(df["upper"], color=band_color, alpha=band_alpha, linestyle="--")
    ax.plot(df["lower"], color=band_color, alpha=band_alpha, linestyle="--")
    if show_mean:
        ax.plot(df["mean"], color=band_color, alpha=band_alpha, linestyle="--")
    ax.plot(df["x"], linestyle="-")
    ax.scatter(x=anom.index, y=anom.values, color="r")

