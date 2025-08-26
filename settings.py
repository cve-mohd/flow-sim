############                Hydrologic Geometry                 ############

initial_flow = 1562.5
peak_flow = 25000

lag_time = 0 * 3600
time_to_peak = 12 * 3600.
peak_time = 28 * 3600.
recession_time = 8 * 3600.

############                Inflow Hydrograph Functions         ############

def trapzoid_hydrograph(t):
    return initial_flow

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
    