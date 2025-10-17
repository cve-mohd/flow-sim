############                Channel Geometry                    ############

wet_n = 0.027
dry_n = 0.030

############                Simulation Parameters               ############

spatial_step = 1000
time_step = 3600
theta = 0.8

sim_duration = 3600*96
tolerance = 1e-6

############                Hydrologic Parameters               ############

initial_roseires_level = 490
gates_open = True

############                Inflow Hydrograph Functions         ############

lag_time = 0
time_to_peak = 12 * 3600.
peak_time = 28 * 3600
recession_time = 10 * 3600.

initial_flow = 1562.5

peak_flow = 25000

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
    
def constant_flow(t):
    return initial_flow
    
############                Other Global Variables              ############

rs_stages = [i for i in range(480, 494)]
spillway_discharges = [7147, 7420, 7686, 7945, 8197, 8449, 8701, 8946, 9184, 9422, 9653, 9891, 10122, 10346]
total_discharges = [14034, 14438, 14830, 15220, 15598, 15971, 16342, 16703, 17062, 17409, 17763, 18110, 18450, 18785]

from src.hydromodel.rating_curve import RatingCurve
rating_curves = []
roseires_spillway_rating_curve = RatingCurve()
roseires_spillway_rating_curve.fit(discharges=spillway_discharges, stages=rs_stages)

roseires_total_rating_curve = RatingCurve()
roseires_total_rating_curve.fit(discharges=total_discharges, stages=rs_stages)

if gates_open:
    used_roseires_rc = roseires_total_rating_curve
else:
    used_roseires_rc = None
