############                River Parameters                    ############

manning_coefficient = 0.027
initial_flow_rate = 1562.5

############                Simulation Parameters               ############

epochs = 1
scheme = 'lax' # 'preissmann' or 'lax'
sim_duration = 3600 * 48

preissmann_time_step = 3600
theta = 0.6
tolerance = 1e-6

lax_time_step = 5

storage_area = 30e6


results_size = (-1, -1) # Default is -1, meaning print all data points. (t, x)


sec_bc = ('constant', 'constant')

hydrograph_duration = 18 * 3600

def f(t):
    if t <= hydrograph_duration:
        return 1562.5 + (10000 - 1562.5) * t/hydrograph_duration
    else:
        return 10000

lengths        = [16000, 32000, 37000, 8000, 27000]
widths         = [250  , 650  , 1500 , 3000, 6000]

us_water_level = [502.5, 490  , 490  , 490  , 490]
ds_water_level = [490  , 490  , 490  , 490  , 490]

us_bed_levels  = [495.0, 482.5, 479.6, 476.3, 475.5]
ds_bed_levels  = [482.5, 479.6, 476.3, 475.5, 473.1]

chainages      = [sum(lengths[:i]) for i in range(len(lengths))]

rating_curves = []
rs_stages = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493]
rs_discharges = [7147, 7420, 7686, 7945, 8197, 8449, 8701, 8946, 9184, 9422, 9653, 9891, 10122, 10346]

from utility import RatingCurve
roseires_rating_curve = RatingCurve()
roseires_rating_curve.fit(discharges=rs_discharges, stages=rs_stages)

reaches = [
    {
        'id': i,
        'length': lengths[i],
        'width': widths[i],
        'us_water_level': us_water_level[i],
        'ds_water_level': ds_water_level[i],
        'us_bed_level': us_bed_levels[i],
        'ds_bed_level': ds_bed_levels[i],
        'chainage': chainages[i]
    }
    for i in range(len(lengths))]