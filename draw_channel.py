import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def import_geometry(path):
    df = pd.read_excel(path, skiprows=1)

    chainages = df.iloc[:, 1]
    widths    = df.iloc[:, 2]
    levels    = df.iloc[:, 3]

    width_df = pd.DataFrame({
        "chainage": chainages,
        "width": widths
    }).dropna()
    width_df = width_df.astype(float).sort_values(by="chainage")

    level_df = pd.DataFrame({
        "chainage": chainages,
        "level": levels
    }).dropna()
    level_df = level_df.astype(float).sort_values(by="chainage")

    width_ch  = width_df["chainage"].tolist()
    widths    = width_df["width"].tolist()
    level_ch  = level_df["chainage"].tolist()
    levels    = level_df["level"].tolist()

    return widths, levels, width_ch, level_ch

def plot_channel_topview(chainages, widths):
    chainages = np.array(chainages, dtype=float)
    widths = np.array(widths, dtype=float)

    # half-widths
    half_w = widths / 2.0

    # left and right bank coordinates
    left_x = chainages
    left_y = -half_w
    right_x = chainages
    right_y = +half_w

    # build polygon path (downstream along right bank, upstream along left bank)
    X = np.concatenate([right_x, left_x[::-1]])
    Y = np.concatenate([right_y, left_y[::-1]])

    plt.figure(figsize=(10, 3))
    plt.fill(X, Y, color="lightblue", edgecolor="k", linewidth=0.5)
    plt.plot([chainages[0], chainages[-1]], [0, 0], "k--", lw=0.5)  # centerline
    plt.xlabel("Chainage (m)")
    plt.ylabel("Cross-channel (m)")
    plt.title("Channel top view (planform width variation)")
    plt.axis("equal")
    plt.show()

widths, levels, width_ch, level_ch = import_geometry("geometry.xlsx")
# Example usage
plot_channel_topview(width_ch, widths)
