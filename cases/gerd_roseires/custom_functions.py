import numpy as np
import matplotlib.pyplot as plt
import os
from src.hydromodel.cross_section import CrossSection
from pandas import read_csv

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
    table = read_csv(path, skiprows=[1])
    table = table.astype(np.float64).sort_values(by="stage")
    
    area_curve = table.to_numpy()[:, :2]
    area_curve[:, 1] *= 1e6
    
    return area_curve

def import_hydrograph(path: str) -> np.ndarray:  
    table = read_csv(path, skiprows=[1])
    table = table.astype(np.float64).sort_values(by="time")
    
    area_curve = table.to_numpy()
    area_curve[:, 0] *= 3600
    
    return area_curve

def import_table(path: str, header: bool = True, sort_by: str = None):
    table = read_csv(path, header=(0 if header else None)).dropna(axis=1, how="all").dropna()
    
    if sort_by is not None:
        table = table.astype(np.float64).sort_values(by=sort_by)
    
    return table.to_numpy(dtype=np.float64)

def load_cross_sections(xs_folder: str, info_csv: str):
    """
    Loads cross-section geometry and properties, returning:
        chainages  -> numpy array of chainage values [m]
        sections   -> list of CrossSection objects

    Parameters
    ----------
    xs_folder : str
        Path to folder containing .csv files named by xs_id (e.g., "1.csv", "2.csv", ...).
    info_csv : str
        Path to CSV file containing metadata with columns:
        xs_id, chainage, left_fp_limit, right_fp_limit, n_left, n_main, n_right
    """
    # Load the metadata
    info = read_csv(info_csv)
    chainages = []
    sections = []

    for _, row in info.iterrows():
        xs_id = int(row["xs_id"])
        fname = os.path.join(xs_folder, f"{xs_id}.csv")
        if not os.path.exists(fname):
            # Skip missing files but keep warning
            print(f"Warning: cross-section {xs_id} not found at {fname}")
            continue

        # Read x, z pairs
        data = np.loadtxt(fname, delimiter=",", skiprows=1)
        if data.ndim == 1 or data.shape[1] < 2:
            raise ValueError(f"Invalid data in {fname}. Expected two columns (x, z).")
        x, z = data[:, 0], data[:, 1]
        
        # Create the CrossSection object
        cs = CrossSection(x=x, z=z)
        cs.left_fp_limit = float(row["left_fp_limit"])
        cs.right_fp_limit = float(row["right_fp_limit"])
        cs.n_left = float(row["n_left"])
        cs.n_main = float(row["n_main"])
        cs.n_right = float(row["n_right"])

        # Append to lists
        chainages.append(float(row["chainage"]))
        sections.append(cs)

    order = np.argsort(chainages)
    chainages = np.array(chainages)[order]
    sections = [sections[i] for i in order]

    return chainages.tolist(), sections
