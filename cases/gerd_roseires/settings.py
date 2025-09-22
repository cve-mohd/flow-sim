############                Channel Geometry                    ############

wet_n = 0.027
dry_n = 0.030

############                Simulation Parameters               ############

enforce_physicality = False
sim_duration = 3600 * 96

dx = 1000

preissmann_dt = 3600
theta = 0.8
tolerance = 1e-4

lax_time_step = 10
lax_secondary_bc = ('constant', 'constant')

############                Hydrologic Parameters               ############

initial_roseires_level = 490
gates_open = True

############                Inflow Hydrograph Functions         ############

lag_time = 0
time_to_peak = 0 * 3600.
peak_time = 1000 * 3600
recession_time = 0 * 3600.

initial_flow = 1562.5

peak_flow = 24000

def trapzoid_hydrograph(t):
    if t <= lag_time:
        return initial_flow
    elif t - lag_time < time_to_peak:
        return initial_flow + (peak_flow - initial_flow) * t / time_to_peak
    elif t - lag_time - time_to_peak < peak_time:
        return peak_flow
    elif t - lag_time - time_to_peak - peak_time < recession_time:
        return peak_flow - (peak_flow - initial_flow) * (t - time_to_peak - peak_time) / recession_time
    else:
        return initial_flow
    
############                Other Global Variables              ############

rs_stages = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493]
spillway_discharges = [7147, 7420, 7686, 7945, 8197, 8449, 8701, 8946, 9184, 9422, 9653, 9891, 10122, 10346]
total_discharges = [14034, 14438, 14830, 15220, 15598, 15971, 16342, 16703, 17062, 17409, 17763, 18110, 18450, 18785]

from src.utility import RatingCurve
rating_curves = []
roseires_spillway_rating_curve = RatingCurve()
roseires_spillway_rating_curve.fit(discharges=spillway_discharges, stages=rs_stages)

roseires_total_rating_curve = RatingCurve()
roseires_total_rating_curve.fit(discharges=total_discharges, stages=rs_stages)

if gates_open:
    used_roseires_rc = roseires_total_rating_curve
else:
    used_roseires_rc = None
