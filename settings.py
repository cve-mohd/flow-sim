############                River Attributes                    ############

BED_SLOPE = 12.5/15000
MANNING_COEFF = 0.027
WIDTH = 250
LENGTH = 15000

############                Simulation Parameters               ############

PREISSMANN_BETA = 0.6
TIME_STEP = 3600
SPATIAL_STEP = 1000
DURATION = 3600 * 24
TOLERANCE = 1e-4
RESULTS_SIZE = (-1, -1)

############                Upstream Boundary                   ############

US_INIT_DEPTH = 7.5
US_INIT_DISCHARGE = 1562.5
US_INIT_STAGE = 502.5

PEAK_DISCHARGE = 10000
PEAK_HOUR = 6

US_RATING_CURVE_COEFF = [17376701.003616, -69833.35989796935, 70.16036993338325]

############                Downstream Boundary                 ############

DS_INIT_DEPTH = 7.5
DS_INIT_DISCHARGE = 1562.5
DS_INIT_STAGE = 490

DS_RATING_CURVE_COEFF = [-786548.06, 2936.794642859, -2.643543956]
