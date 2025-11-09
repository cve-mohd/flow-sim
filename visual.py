import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cross_section_from_index(index, folder='cases\\gerd_roseires\\data\\cross_sections_base\\', results_csv='composite_trapezoids.csv',
                                  overlay=True, save=False, show=True):
    """
    Plot an irregular cross-section and its trapezoidal approximations
    using 'composite_trapezoids.csv' and the original cross-section file.

    Parameters
    ----------
    index : int
        Index (row number or ID) of the cross-section to plot.
    folder : str
        Folder containing the original cross-section .csv files (x,z columns).
    results_csv : str
        Path to composite_trapezoids.csv file with trapezoid parameters.
    overlay : bool
        Overlay trapezoid(s) on top of the original cross-section.
    save : bool
        If True, saves the plot as a .png next to the source file.
    show : bool
        If True, displays the plot interactively.
    """

    # --- Read trapezoid parameters ---
    results = pd.read_csv(results_csv)

    # Locate the target row
    row = results.iloc[[index]]

    if row.empty:
        raise ValueError(f"No cross-section found for index {index}")

    row = row.squeeze()  # convert to Series

    # --- Identify corresponding cross-section file ---
    xs_file = os.path.join(folder, row['file'])

    # --- Read original cross-section data ---
    xs_data = pd.read_csv(xs_file, header=0, names=['x', 'z'])
    x, z = xs_data['x'].values, xs_data['z'].values
    z_min = np.min(z)

    # --- Extract parameters ---
    b_main = float(row['b_main'])
    m_main = float(row['m_main'])
    b_fp   = float(row['b_fp'])
    m_fp   = float(row['m_fp'])
    h_bankfull = float(row['h_bankfull'])
    h_max = float(row['h_max'])

    # Build trapezoid definitions    
    main_ch_trapezoid = {'b': b_main, 'm': m_main, 'zb': z_min, 'hb': h_bankfull}  # main channel
    floodplain_trapezoid = {'b': b_fp, 'm': m_fp, 'zb': z_min + h_bankfull, 'hb': h_max - h_bankfull}  # floodplain

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    
    if overlay:
        ax.plot(x, z, 'k-', lw=1.5, label='Original cross-section')

    colors = ['tab:blue', 'tab:orange']
    
    # Floodplain
    bottom_width = floodplain_trapezoid['b']
    side_slope = floodplain_trapezoid['m']
    bed_level = floodplain_trapezoid['zb']
    bankful_depth = floodplain_trapezoid['hb']
    center = x[0] + 0.5 * (x[-1] - x[0])
    
    left = center - 0.5 * bottom_width - side_slope * bankful_depth
    right = left + bottom_width + 2*side_slope*bankful_depth
    xs = np.array([left, left + side_slope*bankful_depth, left + side_slope*bankful_depth + bottom_width, right])
    zs = np.array([bed_level+bankful_depth, bed_level, bed_level, bed_level+bankful_depth])
    label = "Floodplain"
    ax.plot(xs, zs, color=colors[1], lw=2, label=label)
    ax.fill_between(xs, zs, bed_level, color=colors[1], alpha=0.25)
    
    # Main channel
    bottom_width = main_ch_trapezoid['b']
    side_slope = main_ch_trapezoid['m']
    bed_level = main_ch_trapezoid['zb']
    bankful_depth = main_ch_trapezoid['hb']
    
    left = center - 0.5 * bottom_width - side_slope * bankful_depth
    right = left + bottom_width + 2*side_slope*bankful_depth
    xs = np.array([left, left + side_slope*bankful_depth, left + side_slope*bankful_depth + bottom_width, right])
    zs = np.array([bed_level+bankful_depth, bed_level, bed_level, bed_level+bankful_depth])
    label = "Main channel"
    ax.plot(xs, zs, color=colors[0], lw=2, label=label)
    ax.fill_between(xs, zs, bed_level - 0.3*bankful_depth, color=colors[0], alpha=0.25)

    # --- Add bankfull line ---
    ax.axhline(z_min + h_bankfull, color='gray', ls='--', lw=1, label='Bankfull elevation')

    # --- Formatting ---
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"Cross-section {index} — Trapezoidal approximation")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    #ax.invert_yaxis()  # water depth increases downward
    plt.tight_layout()

    # --- Save or show ---
    if save:
        out_name = os.path.splitext(xs_file)[0] + "_approx.png"
        plt.savefig(out_name, dpi=300)
        print(f"Saved figure: {out_name}")

    if show:
        plt.show()
    else:
        plt.close()

for i in range(22):
    plot_cross_section_from_index(index=i)