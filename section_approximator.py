import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path

# === CONFIGURATION ===
FOLDER = Path("cases\\gerd_roseires\\data\\cross_sections_base")   # folder containing x,z CSV files
OUTPUT = "composite_trapezoids.csv"

def cross_section_properties(x, z, h_values):
    """
    Compute A(h), P(h) for an irregular cross-section,
    correctly handling disconnected sub-channels.
    
    h_values = list of depths above min bed elevation.
    Returns dict {h: (A, P)}.
    """
    z_min = np.min(z)
    props = {}
    for h in h_values:
        water_level = z_min + h
        
        # Use a small tolerance for float comparisons
        if h < 1e-9:
            props[h] = (0.0, 0.0)
            continue

        # Find all points on the bed that are submerged
        mask = z < water_level
        x_sub = x[mask]
        z_sub = z[mask]
        
        # Find all intersections with the water surface
        for i in range(len(z)-1):
            z1, z2 = z[i], z[i+1]
            x1, x2 = x[i], x[i+1]
            # Check if segment crosses the water level
            if (z1 - water_level) * (z2 - water_level) < 0:
                # Linearly interpolate the intersection
                x_int = x1 + (x2 - x1) * (water_level - z1) / (z2 - z1)
                x_sub = np.append(x_sub, x_int)
                z_sub = np.append(z_sub, water_level)
                
        if len(x_sub) < 2:
            props[h] = (0.0, 0.0)
            continue
            
        # Sort all submerged points and intersections by x
        order = np.argsort(x_sub)
        x_sub, z_sub = x_sub[order], z_sub[order]
        
        # --- Area Calculation (This was already correct) ---
        A = np.trapezoid(water_level - z_sub, x_sub)
        
        # --- Wetted Perimeter Calculation (This is the fix) ---
        # We must iterate through the segments and only sum the lengths
        # of the segments that are *not* the flat water surface.
        P = 0.0
        dx = np.diff(x_sub)
        dz = np.diff(z_sub)
        
        for j in range(len(dx)):
            # Check if both points of the segment are (within tolerance)
            # at the water level.
            is_water_surface = (np.abs(z_sub[j] - water_level) < 1e-9) and \
                               (np.abs(z_sub[j+1] - water_level) < 1e-9)
            
            if not is_water_surface:
                # If it's not the water surface, it's the bed. Add its length.
                P += np.sqrt(dx[j]**2 + dz[j]**2)
                
        props[h] = (A, P)
        
    return props


def trapezoid_area_perimeter(b, m, h):
    """Return A, P for a symmetric trapezoid at depth h."""
    A = h * (b + m * h)
    P = b + 2 * h * np.sqrt(1 + m**2)
    return A, P

# --- objective ---
def objective(params, h_vals, A_targets, P_targets):
    """
    Least-squares objective: fit trapezoid to minimize error in A and P.
    """
    b, m = params
    # Model trapezoid geometry
    A_model = np.array([trapezoid_area_perimeter(b, m, h)[0] for h in h_vals])
    #P_model = np.array([trapezoid_area_perimeter(b, m, h)[1] for h in h_vals])
    
    # Relative residuals
    residuals_A = (A_model - A_targets) / np.clip(A_targets, 1e-6, None)
    #residuals_P = (P_model - P_targets) / np.clip(P_targets, 1e-6, None)
    
    # FIX: Return a single, concatenated array of all residuals
    return residuals_A
    return np.concatenate((residuals_A, residuals_P))

def fit_trapezoid(h_vals, A_targets, P_targets, b0 = None, m0 = None, bounds = ([0.1, 0.0], [3000000.0, 1000.0])):
    """Fit (b, m) minimizing area error."""
    # Initial guess
    b0 = np.max(A_targets) / np.max(h_vals) if b0 is None else b0
    m0 = 1 if m0 is None else m0
    res = least_squares(objective, [b0, m0],
                        bounds=bounds,
                        args=(h_vals, A_targets, P_targets))
    b, m = res.x
    return b, m, res.cost

import numpy as np
# For the improved smoothing, you would:
# from scipy.signal import savgol_filter 

def determine_bankfull_depth(h, A, window_size=5):
    """
    Determines bankfull depth by finding the "knee" in the
    Area-Depth relationship, which corresponds to the point of
    maximum change in channel width (dA/dh).

    This is found by identifying the peak of the second derivative (d2A/dh2).

    Args:
        h (np.array): 1D array of depths, must be sorted.
        A (np.array): 1D array of corresponding cross-sectional areas.
        window_size (int): The window size for smoothing the derivative. 
                           Must be an odd number.
    
    Returns:
        float: The estimated bankfull depth (h_bf).
    """
    
    if window_size % 2 == 0:
        window_size += 1 # Ensure window size is odd for symmetry
        
    # 1. Compute first derivative (Top Width)
    # Use second-order accurate central differences
    dA_dh = np.gradient(A, h, edge_order=2)

    # 2. Smooth the derivative (Width)
    # A simple moving average is used here.
    # A Savitzky-Golay filter would be a better choice if Scipy is available:
    # dA_dh_smooth = savgol_filter(dA_dh, window_size, 3) # window, poly_order
    dA_dh_smooth = np.convolve(dA_dh, np.ones(window_size)/window_size, mode='valid')
    
    # Note: 'valid' mode shortens the array. We must also shorten 'h' to match.
    # This avoids the edge-effect errors from 'same' mode.
    h_trimmed = h[(window_size//2):-(window_size//2)]

    if len(h_trimmed) == 0:
        # Data is too short for the window size
        return np.max(h)

    # 3. Compute second derivative (Change in Width)
    # This tells us *where* the width is changing fastest.
    d2A_dh2 = np.gradient(dA_dh_smooth, h_trimmed)

    # 4. Find the peak of the second derivative
    # This is the "knee" or "breakpoint" where the floodplain begins.
    try:
        idx_bf = np.argmax(d2A_dh2)
        h_bf = h_trimmed[idx_bf]
    except (ValueError, IndexError):
        # Fallback if something fails (e.g., empty array)
        h_bf = np.max(h)

    return h_bf

def fit_compound_trapezoid(x, z, h_values):
    """Fit a two-part compound trapezoid: main + floodplain."""
    # Step 1: compute measured A(h)
    props = cross_section_properties(x, z, h_values)
    h = np.array(list(props.keys()))
    AP = np.array(list(props.values()))
    A = AP[:, 0]; P = AP[:, 1]

    # Detect bankfull depth
    h_bf = determine_bankfull_depth(h, A)

    # fit main channel (below h_bf)
    mask_main = h <= h_bf
    b_c, m_c, err_c = fit_trapezoid(h[mask_main], A[mask_main], P[mask_main])

    # fit floodplain (above h_bf)
    mask_fp = h > h_bf
    if np.sum(mask_fp) >= 3:
        A_bf = np.interp(h_bf, h, A)
        P_bf = np.interp(h_bf, h, P)
        
        b_min = b_c + 2*m_c*h_bf
        h_arg = h[mask_fp] - h_bf
        A_arg = A[mask_fp] - A_bf
        P_arg = P[mask_fp] - P_bf + b_min

        b_f, m_f, err_f = fit_trapezoid(h_arg, A_arg, P_arg, b0=b_min, bounds=([b_min, 0.0], [1000000.0, 10000.0]))
    else:
        b_f, m_f, err_f = np.nan, np.nan, np.nan

    return {
        "b_main": b_c,
        "m_main": m_c,
        "err_main": err_c,
        "b_fp": b_f,
        "m_fp": m_f,
        "err_fp": err_f,
        "h_bankfull": h_bf,
        "h_max": np.max(h)
    }

# --- Main loop ---
records = []
for file in FOLDER.glob("*.csv"):
    try:
        data = pd.read_csv(file)
    except:
        data = pd.read_csv(file, header=0, names=["x","z"])
    x, z = data.iloc[:,0].values, data.iloc[:,1].values
    max_depth = float(z.max()-z.min())
    min_h = 2.01
    n_steps = int(max(20, (max_depth-min_h)*10))
    depths = np.linspace(min_h, max_depth, n_steps)   # water depths (m) to fit over
    result = fit_compound_trapezoid(x, z, depths)
    result["file"] = file.name
    records.append(result)
    print(f"Processed {file.name}")

# --- Save results ---
df = pd.DataFrame(records)
df.to_csv(OUTPUT, index=False)
print(f"\nResults saved to {OUTPUT}")
