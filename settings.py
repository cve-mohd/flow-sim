############                River Attributes                    ############

BED_SLOPE = 12.5/15000
MANNING_COEFF = 0.027
WIDTH = 250
LENGTH = 15000

############                Simulation Parameters               ############

SCHEME = 'lax' # 'preissmann' or 'lax'
LAX_APPROX = 'mirror' # 'same' or 'mirror'
PREISSMANN_BETA = 0.6
TIME_STEP = 50
SPATIAL_STEP = 1000
DURATION = 3600 * 24
TOLERANCE = 1e-4
RESULTS_SIZE = (25, -1) # Default is -1. Means print all data points.

############                Upstream Boundary                   ############

US_INIT_DEPTH = 7.5
US_INIT_DISCHARGE = 1562.5
US_INIT_STAGE = 502.5

PEAK_DISCHARGE = 10000
PEAK_HOUR = 6

US_RATING_CURVE = {"base": 500, "coefficients": [327.23, 318.44, 70.26]}

############                Downstream Boundary                 ############

DS_INIT_DEPTH = 7.5
DS_INIT_DISCHARGE = 1562.5
DS_INIT_STAGE = 490

DS_RATING_CURVE = {"base": 466.7, "coefficients": [8266.62, 469.31, -2.64]}
