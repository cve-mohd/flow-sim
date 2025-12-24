############                Simulation Parameters               ############

spatial_step = 1000
time_step = 3600
theta = 0.8

sim_duration = 3600 * 120
tolerance = 1e-6

############                Hydrologic Parameters               ############

initial_roseires_level = 487.
JAMMED_SPILLWAYS = 0
JAMMED_SLUICEGATES = 0
OPEN_TIMING = 3600 * 6
CLOSE_TIMING = 3600 * 55

###########

base_flow = 1562.5
peak_flow = 26000.

lag_time = 0.0
time_to_peak = 3600 * 24
time_at_peak = 3600 * 24

from math import sin, pi
def sin_wave(time: int):
    if time <= lag_time:
        return base_flow
    elif time - lag_time < time_to_peak:
        return base_flow + sin(0.5 * pi * float(time - lag_time) / time_to_peak) * (peak_flow - base_flow)
    elif time - lag_time < time_to_peak + time_at_peak:
        return peak_flow
    elif time - lag_time < 2 * time_to_peak + time_at_peak:
        return base_flow + sin(0.5 * pi * float(time - lag_time - time_at_peak) / time_to_peak) * (peak_flow - base_flow)
    else:
        return base_flow
