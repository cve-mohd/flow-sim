import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pathlib import Path

# === CONFIGURATION ===
FOLDER = Path("cases\\gerd_roseires\\data\\cross_sections_base")   # folder containing x,z CSV files
OUTPUT = "cases\\gerd_roseires\\data\\composite_trapezoids.csv"

def cross_section_properties(x, z, h_values):
    """
    Compute A(h) for an irregular cross-section    
    h_values = list of depths above min bed elevation.
    Returns 1D list A
    """
    z_min = np.min(z)
    A = []
    for h in h_values:
        water_level = z_min + h
        
        if h < 1e-9:
            A.append(0)
            continue

        mask = z < water_level
        x_sub = x[mask]
        z_sub = z[mask]
        
        for i in range(len(z)-1):
            z1, z2 = z[i], z[i+1]
            x1, x2 = x[i], x[i+1]
            if (z1 - water_level) * (z2 - water_level) < 0:
                x_int = x1 + (x2 - x1) * (water_level - z1) / (z2 - z1)
                x_sub = np.append(x_sub, x_int)
                z_sub = np.append(z_sub, water_level)
                
        if len(x_sub) < 2:
            A.append(0)
            continue
            
        order = np.argsort(x_sub)
        x_sub, z_sub = x_sub[order], z_sub[order]
        
        a = np.trapezoid(water_level - z_sub, x_sub)
                        
        A.append(a)
        
    return np.array(object=A, dtype=np.float64)

def get_segments_at_level(x, z, level):
    """Finds all (x_start, x_end) segments *below* a given z level."""
    segments = []
    i = 0
    n = len(z)
    while i < n:
        if z[i] < level:
            start_idx = i
            while i < n and z[i] < level:
                i += 1
            end_idx = i - 1
            
            x_start = x[start_idx]
            if start_idx > 0:
                z1, z2 = z[start_idx-1], z[start_idx]
                x1, x2 = x[start_idx-1], x[start_idx]
                if z1 >= level: # Check it crosses from above
                     x_start = x1 + (x2 - x1) * (level - z1) / (z2 - z1)
            
            x_end = x[end_idx]
            if end_idx < n - 1:
                z1, z2 = z[end_idx], z[end_idx+1]
                x1, x2 = x[end_idx], x[end_idx+1]
                if z2 >= level: # Check it crosses from above
                    x_end = x1 + (x2 - x1) * (level - z1) / (z2 - z1)
                    
            segments.append((x_start, x_end))
        else:
            i += 1
    return segments

def trapezoid_area_perimeter(b, m, h):
    """Return A, P for a symmetric trapezoid at depth h."""
    A = h * (b + m * h)
    P = b + 2 * h * np.sqrt(1 + m**2)
    return A, P

def objective(params, h_vals, A_targets):
    """
    Least-squares objective: fit trapezoid to minimize error in A and P.
    """
    b, m = params
    A_model = np.array([trapezoid_area_perimeter(b, m, h)[0] for h in h_vals])
    P_model = np.array([trapezoid_area_perimeter(b, m, h)[1] for h in h_vals])
    
    residuals_A = (A_model - A_targets) / np.clip(A_targets, 1e-6, None)
    #residuals_P = (P_model - P_targets) / np.clip(P_targets, 1e-6, None)
    
    return residuals_A
    #return np.concatenate((residuals_A, residuals_P))

def fit_trapezoid(h_vals, A_targets, bounds, b0 = None, m0 = None):
    """Fit (b, m) minimizing area and perimeter error."""
    # Initial guess
    b0 = np.max(A_targets) / np.max(h_vals) if b0 is None else b0
    m0 = 1 if m0 is None else m0
    res = least_squares(objective, [b0, m0],
                        bounds=bounds,
                        args=(h_vals, A_targets))
    b, m = res.x
    return b, m, res.cost

def determine_bankfull_depth(h, A, window_size=5):
    """
    Determines bankfull depth by finding the "knee" in the
    Area-Depth relationship (peak of the second derivative).
    """
    
    if window_size % 2 == 0:
        window_size += 1
        
    dA_dh = np.gradient(A, h, edge_order=2)
    dA_dh_smooth = np.convolve(dA_dh, np.ones(window_size)/window_size, mode='valid')
    h_trimmed = h[(window_size//2):-(window_size//2)]

    if len(h_trimmed) == 0:
        return np.max(h)

    d2A_dh2 = np.gradient(dA_dh_smooth, h_trimmed)

    try:
        idx_bf = np.argmax(d2A_dh2)
        h_bf = h_trimmed[idx_bf]
    except (ValueError, IndexError):
        h_bf = np.max(h)

    return h_bf

def fit_compound_trapezoid(x, z, h, bank_z=None):
    """Fit a two-part compound trapezoid: main + floodplain."""
    A = cross_section_properties(x, z, h)
    z_min = np.min(z)

    h_bf = determine_bankfull_depth(h, A) if bank_z is None else bank_z - z_min
    z_bank = z_min + h_bf
    
    # Find Bank Locations and Apportion Floodplain
    all_segments = get_segments_at_level(x, z, z_bank)
    
    if not all_segments:
        x_bank_left, x_bank_right = np.min(x), np.max(x)
    else:
        main_channel_seg = max(all_segments, key=lambda s: s[1] - s[0])
        x_bank_left, x_bank_right = main_channel_seg
        
    T_bf = x_bank_right - x_bank_left
    T_max = x[-1] - x[0]

    mask_main = h <= h_bf
    if np.sum(mask_main) < 3:
        return {
            "z_min": z_min, "b_main": np.nan, "m_main": np.nan, "err_main": np.nan,
            "b_fp_left": np.nan, "b_fp_right": np.nan, "m_fp": np.nan, "err_fp": np.nan,
            "h_bankfull": h_bf, "h_max": np.max(h)
        }
    
    min_b = 0.0
    max_T = 0.25 * (3*T_bf + T_max)
    max_m = (max_T - min_b) / (2 * h_bf)
    bounds_c = ([min_b, 0.0], [max_T, max_m])
    
    b_c, m_c, err_c = fit_trapezoid(h[mask_main], A[mask_main], bounds=bounds_c)
    T_bf = b_c + 2 * m_c * h_bf

    width_left_total = x_bank_left - np.min(x)
    width_right_total = np.max(x) - x_bank_right
    total_available_width = width_left_total + width_right_total
    
    mask_fp = h > h_bf
    if np.sum(mask_fp) >= 3:
        A_bf = np.interp(h_bf, h, A)
        
        h_arg = h[mask_fp] - h_bf
        A_arg = A[mask_fp] - A_bf
        
        b0_f = T_bf+0.01# (A_arg[0] / h_arg[0]) if h_arg[0] > 1e-6 else 1.0
        bounds_f = ([T_bf, 0.0], [1000000.0, 10000.0])
        
        b_f, m_f, err_f = fit_trapezoid(h_arg, A_arg, b0=b0_f, bounds=bounds_f)
    else:
        b_f, m_f, err_f = np.nan, np.nan, np.nan

    # Apportion b_f (which is b_fp_left + b_fp_right)
    b_f_left = np.nan
    b_f_right = np.nan
    
    if not np.isnan(b_f - T_bf):
        if total_available_width > 1e-6:
            frac_left = width_left_total / total_available_width
            b_f_left = (b_f - T_bf) * frac_left
            b_f_right = (b_f - T_bf) * (1.0 - frac_left)
        else:
            b_f_left = 0
            b_f_right = 0

    return {
        "z_min": z_min,
        "b_main": b_c,
        "m_main": m_c,
        "err_main": err_c,
        "b_fp_left": b_f_left,
        "b_fp_right": b_f_right,
        "m_fp": m_f,
        "err_fp": err_f,
        "h_bankfull": h_bf,
        "h_max": np.max(h)
    }

# --- Main loop ---
records = [] # 1,   3,   5,   6,   7,    8,   9,    11,  14,   15,    19,    21,   24,  29,  31,   36,  42,   46,  51,   53,   54,   55
bank_z  =   [479, 482, 483, 483, 483, None, 483, 480.9, 483, None, 480.9, 482.5, None, 490, 490, None, 482, None, 490, None, None, None]
for i, file in enumerate(FOLDER.glob("*.csv")):
    try:
        data = pd.read_csv(file)
    except:
        try:
            data = pd.read_csv(file, header=0, names=["x","z"])
        except Exception as e:
            print(f"Failed to read {file.name}: {e}")
            continue
            
    x, z = data.iloc[:,0].values, data.iloc[:,1].values
    
    if len(x) < 3:
        print(f"Skipping {file.name}: insufficient data points.")
        continue
        
    max_depth = float(z.max()-z.min())
    if max_depth < 3.0: # Ensure min_h is not > max_depth
        min_h = max_depth * 0.1
    else:
        min_h = 2.01
        
    if min_h >= max_depth:
        max_depth = min_h + 1.0 # arbitrary small range
        
    n_steps = int(max(20, (max_depth-min_h)*10))
    depths = np.linspace(min_h, max_depth, n_steps)
    
    try:
        result = fit_compound_trapezoid(x, z, depths, bank_z[i])
        result["file"] = file.name
        records.append(result)
        print(f"Processed {file.name}")
    except Exception as e:
        print(f"Failed to process {file.name}: {e}")

# --- Save results ---
df = pd.DataFrame(records)

cols = ["z_min", "file", "b_main", "m_main", "err_main", 
        "b_fp_left", "b_fp_right", "m_fp", "err_fp",
        "h_bankfull", "h_max"]

df_cols = [c for c in cols if c in df.columns]
df = df[df_cols]

df.to_csv(OUTPUT, index=False)
print(f"\nResults saved to {OUTPUT}")