from channel import Channel
from boundary import Boundary
from case_study_settings import *
from preissmann import PreissmannSolver
import time
from copy import deepcopy

start = time.perf_counter()

stage_hyd_times = []
stage_hyd_stages = []
            
for e in range(epochs):
            
    new_stage_hyd_times = []
    new_stage_hyd_stages = []
        
    for reach in reaches:        
        h_us = reach['us_water_level'] - reach['us_bed_level']
        h_ds = reach['ds_water_level'] - reach['ds_bed_level']
        
        us = Boundary(h_us, 'flow_hydrograph' , reach['us_bed_level'], chainage=reach['chainage'])
            
        if reach['id'] == 0:
            us.set_flow_hydrograph(trapzoid_hydrograph)
        else:
            times = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
            discharges = solver.get_results('q', spatial_node=-1).tolist()            
            us.build_flow_hydrograph(times, discharges)
            
        
        if reach['id'] == len(reaches) - 1:
            ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
            ds.set_storage_behavior(storage_area=storage_area, storage_exit_rating_curve=used_roseires_rc)
        else:
            if e == 0:
                ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
            else:
                ds = Boundary(h_ds, 'stage_hydrograph', reach['ds_bed_level'])                
                ds.build_stage_hydrograph(stage_hyd_times[ reach['id'] + 1 ], stage_hyd_stages[ reach['id'] + 1 ])
                
            
        channel = Channel(length = reach['length'],
                        width = reach['width'],
                        initial_flow_rate = initial_flow,
                        bed_slope = 'real',
                        manning_co = manning_coefficient,
                        upstream_boundary = us,
                        downstream_boundary = ds)

        solver = PreissmannSolver(channel, theta, preissmann_time_step, spatial_resolution * reach['length'])
        
        if e==8:
            v = 0
        else:
            v = 0
        
        solver.run(sim_duration, auto=False, tolerance=tolerance, verbose=v)
        
        ##############
        
        if e == epochs - 1:
            solver.save_results((solver.total_sim_duration//3600 + 1, -1), 'Results//Preissmann//Reach ' + str( reach['id'] ))
        else:
            timesss = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
            stages = solver.get_results('s', spatial_node=0).tolist()
                        
            new_stage_hyd_times.append(timesss)
            new_stage_hyd_stages.append(stages)
        
        #############
        
        
    stage_hyd_times = deepcopy(new_stage_hyd_times)
    stage_hyd_stages = deepcopy(new_stage_hyd_stages)
    
    print(f"### Finished epoch {e+1} ###")
            
print("Success!")

end = time.perf_counter()

with open('Results//Preissmann//cpu_time.txt', 'w') as file:
    file.write(f'{end - start:.4f} s')
    