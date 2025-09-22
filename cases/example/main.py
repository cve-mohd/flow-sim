from src.channel import Channel
from src.boundary import Boundary
from src.utility import Hydrograph
from src.preissmann import PreissmannSolver

def trapzoid_hydrograph(t):
    initial_flow = 1000
    peak_flow = 10000
    
    lag_time = 0
    time_to_peak = 3*3600
    peak_time = 6*3600
    recession_time = 4*3600
    
    if t <= lag_time:
        flow = initial_flow
    elif t - lag_time < time_to_peak:
        flow = initial_flow + (peak_flow - initial_flow) * (t - lag_time) / time_to_peak
    elif t - lag_time - time_to_peak < peak_time:
        flow = peak_flow
    elif t - lag_time - time_to_peak - peak_time < recession_time:
        flow = peak_flow - (peak_flow - initial_flow) * (t - lag_time - time_to_peak - peak_time) / recession_time
    else:
        flow = initial_flow
        
    return flow
    
us = Boundary(condition='flow_hydrograph',
              bed_level=5,
              chainage=0,
              hydrograph=Hydrograph(trapzoid_hydrograph))

ds = Boundary(condition='normal_depth',
              bed_level=0,
              chainage=3000)

example_channel = Channel(width = 250,
                        initial_flow = us.hydrograph.get_at(0),
                        roughness = 0.027,
                        upstream_boundary = us,
                        downstream_boundary = ds,
                        interpolation_method='steady-state')

#example_channel.set_intermediate_bed_levels([510], [8000])
#example_channel.set_intermediate_widths([400], [26000])

solver = PreissmannSolver(reach=example_channel,
                          theta=0.8,
                          time_step=3600,
                          spatial_step=1000,
                          simulation_time=24*3600,
                          regularization=False,
                          normalize=True)

solver.run(verbose=0)
solver.save_results(folder_path='cases\\example\\results')
