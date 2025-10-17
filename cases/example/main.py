from src.hydromodel.channel import Channel
from src.hydromodel.boundary import Boundary
from src.hydromodel.hydrograph import Hydrograph
from src.hydromodel.lumped_storage import LumpedStorage
from src.hydromodel.preissmann import PreissmannSolver
from src.hydromodel.lax import LaxSolver

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
              hydrograph=Hydrograph(function=trapzoid_hydrograph))

ds = Boundary(condition='fixed_depth',
              initial_depth=5,
              bed_level=0,
              chainage=20000)

ss = LumpedStorage(surface_area=5000*250, min_stage=5, solution_boundaries=(0, 200))
ds.set_lumped_storage(ss)

example_channel = Channel(width = 250,
                          initial_flow = us.hydrograph.get_at(0),
                          roughness = 0.027,
                          upstream_boundary = us,
                          downstream_boundary = ds)

#example_channel.set_intermediate_bed_levels([510], [8000])
#example_channel.set_intermediate_widths([400], [26000])

p_solver = PreissmannSolver(channel=example_channel,
                            theta=0.8,
                            time_step=3600,
                            spatial_step=1000,
                            simulation_time=24*3600)

p_solver.run(verbose=0)
p_solver.save_results(folder_path='cases\\example\\results\\preissmann')
print('Finished Preissmann.')

example_channel = Channel(width = 250,
                          initial_flow = us.hydrograph.get_at(0),
                          roughness = 0.027,
                          upstream_boundary = us,
                          downstream_boundary = ds,
                          interpolation_method='steady-state')

l_solver = LaxSolver(channel=example_channel,
                     time_step=30,
                     spatial_step=1000,
                     simulation_time=24*3600,
                     secondary_BC=('constant', 'constant'))

l_solver.run(verbose=0)
l_solver.save_results(folder_path='cases\\example\\results\\lax')
print('Finished Lax-Friedrich.')
