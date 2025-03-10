############                River Attributes                    ############

BED_SLOPE = 0.00061 #12.5/15000
BED_SLOPE_CORRECTION = True
MANNING_COEFF = 0.023# 7
WIDTH = 120 #250
LENGTH = 29000 #15000
APPROX_R = False

############                Simulation Parameters               ############

SCHEME = 'preissmann' # 'preissmann' or 'lax'
LAX_APPROX = 'same' # 'same' or 'mirror'
PREISSMANN_BETA = 0.5
TIME_STEP = 3600
SPATIAL_STEP = 1000
DURATION = 3600 * 20
TOLERANCE = 1e-4
RESULTS_SIZE = (-1, -1) # Default is -1. Means print all data points.

############                Upstream Boundary                   ############

US_INIT_DEPTH = 0.8638 #7.5
US_INIT_DISCHARGE = 100 #1562.5
US_INIT_STAGE = 502.5

CUSTOM_INFLOW = True
# Default wave shape is sine wave. To override, set CUSTOM_INFLOW to True and
# implement custom_hydrograph(time) at the bottom of the file.

PEAK_DISCHARGE = 10000
PEAK_HOUR = 6

US_RATING_CURVE = {"base": 500, "coefficients": [327.23, 318.44, 70.26]}

############                Downstream Boundary                 ############

DS_INIT_DEPTH = 0.8638 #7.5
DS_INIT_DISCHARGE = 100 #1562.5
DS_INIT_STAGE = 490

DS_CONDITION = 'normal_depth' # 'fixed_depth', 'rating_curve' or 'normal_depth'.
DS_RATING_CURVE = {"base": 466.7, "coefficients": [8266.62, 469.31, -2.64]}

############           Akbari & Firoozi's hydrograph            ############

def custom_hydrograph(time: float | int) -> float:
    from math import sin, cos, pi
    
    Qb= 100.
    Qp = 200.
    
    tb = 15.
    tp = 5.
    
    t = time / 3600.
    
    if t <= tp:
        Q = Qp / 2 * sin (pi * t / tp - pi / 2) + Qp / 2 + Qb
    elif tp < t and t <= tb:
        Q = Qp / 2 * cos (pi * (t - tp) / (tb - tp)) + Qp / 2 + Qb
    else:
        Q = Qb
        
    return Q