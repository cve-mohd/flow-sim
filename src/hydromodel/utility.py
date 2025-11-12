import os
import numpy as np
    
def create_directory_if_not_exists(directory):
    """
    Checks if a directory exists and creates it if it doesn't.

    Attributes
    ----------
    directory : str
        The path to the directory to check.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def manhattan_norm(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return np.sum(np.abs(vector))

def euclidean_norm(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return np.sum(np.square(vector))**0.5

def seconds_to_hms(seconds: int):
    if seconds < 0:
        return "0:00:00"
    
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    
    return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

def compute_curv(x_coords, y_coords):
    x_coords, y_coords = np.asarray(x_coords, dtype=np.float64), np.asarray(y_coords, dtype=np.float64)
    # Arc length parameterization
    ds = np.hypot(np.diff(x_coords), np.diff(y_coords))
    s = np.insert(np.cumsum(ds), 0, 0.0)

    dx = np.gradient(x_coords, s)
    dy = np.gradient(y_coords, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    # Curvature κ = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    
    return kappa
    