############                River Parameters                    ############

LENGTH = 15000
WIDTH = 250
MANNING_COEFF = 0.027
FLOW_RATE = 1562.5
USE_GVF = True
APPROX_R = False

############                Simulation Parameters               ############

SCHEME = 'lax' # 'preissmann' or 'lax'

TIME_STEP = 3600
SPATIAL_STEP = 1000
DURATION = 3600 * 48
TOLERANCE = 1e-6

LAX_US_2ND_COND = 'constant' # 'constant', 'mirror', or 'rating_curve'
LAX_DS_2ND_COND = 'constant' # 'constant' or 'mirror'

PREISSMANN_THETA = 0.6

RESULTS_SIZE = (-1, -1) # Default is -1, meaning print all data points. (t, x)

############                Upstream Boundary                   ############

US_INIT_DEPTH = 15.9
US_INIT_STAGE = 490

"""
CUSTOM_INFLOW = False
# Default wave shape is sinus. To override, set CUSTOM_INFLOW to True and
# implement custom_hydrograph(time) at the bottom of the file.

PEAK_DISCHARGE = 50000
PEAK_HOUR = 12
"""

#US_RATING_CURVE = {"base": 500, "coefficients": [327.23, 318.44, 70.26]}

############                Downstream Boundary                 ############

DS_INIT_DEPTH = 7.5
DS_INIT_DISCHARGE = 1562.5
DS_INIT_STAGE = 490
DS_MAX_STAGE = 493

DS_POND_AREA = 478983175

DS_CONDITION = 'fixed_depth' # 'fixed_depth', 'rating_curve' or 'normal_depth'.
#DS_RATING_CURVE = {"base": 466.7, "coefficients": [8266.62, 469.31, -2.64]}

############           Akbari & Firoozi's Hydrograph            ############

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