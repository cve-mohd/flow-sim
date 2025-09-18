import numpy as np
import matplotlib.pyplot as plt

def reconstruct_centerline(chainages, curvature, x0, y0, theta0):
    """
    chainages: 1D array (monotonic arc-lengths)
    curvature: 1D array same length (kappa = 1/R). use 0 for straight (R=inf)
    x0, y0: coordinates at chainages[0]
    theta0: heading angle at chainages[0] in radians (0 = +x axis)
    Returns: x, y, theta arrays same length
    """
    s = np.asarray(chainages, dtype=float)
    k = np.asarray(curvature, dtype=float)

    if s.ndim != 1 or k.ndim != 1 or s.size != k.size:
        raise ValueError("chainages and curvature must be 1D arrays of same length")

    # segment lengths
    ds = np.diff(s, prepend=s[0])
    ds[0] = 0.0  # no step at first point

    # integrate theta using trapezoidal rule: theta_i = theta_{i-1} + 0.5*(k_{i-1}+k_i)*ds_i
    theta = np.empty_like(k)
    theta[0] = theta0
    for i in range(1, len(s)):
        theta[i] = theta[i-1] + 0.5*(k[i-1] + k[i]) * (s[i] - s[i-1])

    # integrate x,y: x_i = x_{i-1} + 0.5*(cos(theta_{i-1})+cos(theta_i))*ds_i
    x = np.empty_like(k)
    y = np.empty_like(k)
    x[0], y[0] = x0, y0
    for i in range(1, len(s)):
        ds_i = s[i] - s[i-1]
        x[i] = x[i-1] + 0.5*(np.cos(theta[i-1]) + np.cos(theta[i])) * ds_i
        y[i] = y[i-1] + 0.5*(np.sin(theta[i-1]) + np.sin(theta[i])) * ds_i

    return x, y, theta

def plot_channel_outline(x, y, theta, widths):
    # normals
    widths = np.asarray(widths, dtype=float)
    nx = -np.sin(theta)
    ny = np.cos(theta)

    left_x  = x + 0.5 * widths * nx
    left_y  = y + 0.5 * widths * ny
    right_x = x - 0.5 * widths * nx
    right_y = y - 0.5 * widths * ny

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'k-', label="Centerline")
    plt.plot(left_x, left_y, 'b-', label="Left bank")
    plt.plot(right_x, right_y, 'g-', label="Right bank")
    
    plt.axis("equal")
    plt.legend()
    plt.title("Channel outline")
    plt.show()

    return left_x, left_y, right_x, right_y

def draw(chainages, widths, curvature, x0, y0, theta0):
    x, y, theta = reconstruct_centerline(chainages, curvature, x0, y0, theta0)
    return plot_channel_outline(x, y, theta, widths)
    
def import_geometry(path):
    from pandas import read_excel, DataFrame
    df = read_excel(path, skiprows=1)

    chainages    = df.iloc[:, 1]
    widths       = df.iloc[:, 2]
    levels       = df.iloc[:, 3]
    eastings     = df.iloc[:, 4]
    northings    = df.iloc[:, 5]

    width_df = DataFrame({
        "chainage": chainages,
        "width": widths
    }).dropna()
    width_df = width_df.astype(float).sort_values(by="chainage")

    level_df = DataFrame({
        "chainage": chainages,
        "level": levels
    }).dropna()
    level_df = level_df.astype(float).sort_values(by="chainage")
    
    coords_df = DataFrame({
        "chainage": chainages,
        "eastings": eastings,
        "northings": northings
    }).dropna()
    coords_df = coords_df.astype(float).sort_values(by="chainage")

    width_ch     = width_df["chainage"].tolist()
    widths       = width_df["width"].tolist()
    level_ch     = level_df["chainage"].tolist()
    levels       = level_df["level"].tolist()
    coords_ch    = coords_df["chainage"].tolist()
    coords_x     = coords_df["eastings"].tolist()
    coords_y     = coords_df["northings"].tolist()

    return widths, width_ch, levels, level_ch, coords_x, coords_y, coords_ch

def export_banks(left_x, left_y, right_x, right_y,
                 crs="EPSG:20136",
                 outfile="banks.shp"):
    """
    Export left and right bank polylines to Shapefile.

    Parameters
    ----------
    left_x, left_y : arrays
        Left bank coordinates.
    right_x, right_y : arrays
        Right bank coordinates.
    crs : str
        Coordinate reference system (default: EPSG:32636).
    outfile : str
        Path to save file (GeoJSON or Shapefile).
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    left_line  = LineString(list(zip(left_x, left_y)))
    right_line = LineString(list(zip(right_x, right_y)))

    gdf = gpd.GeoDataFrame(
        {"bank": ["left", "right"]},
        geometry=[left_line, right_line],
        crs=crs
    )

    gdf.to_file(outfile, driver="ESRI Shapefile")
    return gdf

def import_area_curve(path: str) -> np.ndarray:
    """
    Imports stage and area data from an Excel file and returns it as a 2D NumPy array.

    Args:
        path (str): The file path to the Excel file.

    Returns:
        np.ndarray: A 2D NumPy array where the first column contains stages 
                    and the second column contains corresponding areas.
    """
    from pandas import read_excel
    
    df = read_excel(path, skiprows=1)
    stages = df.iloc[:, 0].to_numpy()
    areas = df.iloc[:, 1].to_numpy()
    
    area_curve = np.vstack((stages, areas)).T
    
    return area_curve
