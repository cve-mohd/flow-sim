############                Channel Geometry                    ############

manning_coefficient = 0.029
lengths        = [16000, 32000, 37000, 8000, 27000]
widths         = [250  , 650  , 1500 , 3000, 6000]

storage_area = 440e6 - sum([w*l for w, l in zip(widths, lengths)][1:])

us_bed_levels  = [495.0, 482.5, 479.6, 476.3, 475.5]
ds_bed_levels  = [482.5, 479.6, 476.3, 475.5, 473.1]

chainages      = [sum(lengths[:i]) for i in range(len(lengths))]

############                Simulation Parameters               ############

sim_duration = 3600 * 96
epochs = 10

preissmann_time_step = 300
theta = 0.8
tolerance = 1e-6

lax_time_step = 10
lax_secondary_bc = ('constant', 'constant')

spatial_resolution = 0.05

results_size = (-1, -1) # (t, x)

############                Hydrologic Parameters               ############

roseires_level = 485

initial_flow = 1562.5
peak_flow = 25000

lag_time = 0
time_to_peak = 12 * 3600.
peak_time = 28 * 3600.
recession_time = 8 * 3600.

############                Inflow Hydrograph Functions         ############

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

from utility import RatingCurve
rating_curves = []
roseires_spillway_rating_curve = RatingCurve()
roseires_spillway_rating_curve.fit(discharges=spillway_discharges, stages=rs_stages)

roseires_total_rating_curve = RatingCurve()
roseires_total_rating_curve.fit(discharges=total_discharges, stages=rs_stages)

used_roseires_rc = roseires_total_rating_curve

reaches = [
    {
        'id': i,
        'length': lengths[i],
        'width': widths[i],
        'us_bed_level': us_bed_levels[i],
        'ds_bed_level': ds_bed_levels[i],
        'chainage': chainages[i]
    }
    for i in range(len(lengths))]
