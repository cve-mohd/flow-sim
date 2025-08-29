from reach import Reach
from boundary import Boundary
from case_study_settings import *
from preissmann import PreissmannSolver
from copy import deepcopy
import pandas as pd

def run(roseires_level_):

    depth_at_interface = []
    level_at_interface = []

    for reach_number in range(4, 0, -1):
        reach = reaches[reach_number]
        us = Boundary(initial_depth=0, condition='flow_hydrograph' , bed_level=reach['us_bed_level'], chainage=reach['chainage'])
        
        if reach_number == 4:
            downstream_depth = roseires_level_ - reach['ds_bed_level']
        else:
            downstream_depth = upstream_depth
            
        ds = Boundary(initial_depth=downstream_depth, condition='flow_hydrograph' , bed_level=reach['ds_bed_level'], chainage=reach['chainage'])
        
        reach_instance = Reach(length=reach['length'], width=reach['width'], initial_flow_rate=initial_flow,
                                manning_co=manning_coefficient, upstream_boundary=us, downstream_boundary=ds)
        reach_instance.initialize_conditions(20)
        
        upstream_depth, _ = reach_instance.initial_conditions[0]
        upstream_depth = upstream_depth / reach['width']
        
        depth_at_interface.insert(0, upstream_depth)
        level_at_interface.insert(0, upstream_depth + reach['us_bed_level'])

    stage_hyd_times = []
    stage_hyd_stages = []

    hydrograph_file = pd.read_csv('hydrograph.csv', thousands=',')
    hyd_times = [i*3600 for i in hydrograph_file.iloc[:,0].astype(float).tolist()]
    hyd_flows = hydrograph_file.iloc[:,1].astype(float).tolist()

    for e in range(epochs):
                
        new_stage_hyd_times = []
        new_stage_hyd_stages = []
            
        for reach in reaches:
            #h_us = reach['us_water_level'] - reach['us_bed_level']        
            us = Boundary(0, 'flow_hydrograph' , reach['us_bed_level'], chainage=reach['chainage'])
                
            if reach['id'] == 0:
                #us.set_flow_hydrograph(trapzoid_hydrograph)
                us.build_flow_hydrograph(times=hyd_times, discharges=hyd_flows)
            else:
                times = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
                discharges = solver.get_results('flow_rate', spatial_node=-1).tolist()            
                us.build_flow_hydrograph(times, discharges)
                
            
            if reach['id'] == len(reaches) - 1:
                h_ds = roseires_level_ - reach['ds_bed_level']
                ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
                ds.set_storage_behavior(storage_area=storage_area, storage_exit_rating_curve=used_roseires_rc)
            else:
                h_ds = depth_at_interface[reach['id']]
                if e == 0:
                    ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
                else:
                    ds = Boundary(h_ds, 'stage_hydrograph', reach['ds_bed_level'])                
                    ds.build_stage_hydrograph(stage_hyd_times[ reach['id'] + 1 ], stage_hyd_stages[ reach['id'] + 1 ])
                    
                
            channel = Reach(length = reach['length'],
                            width = reach['width'],
                            initial_flow_rate = initial_flow,
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
                solver.save_results(path = f'results//{roseires_level_}//reach_{reach['id']}')
            else:
                timesss = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
                stages = solver.get_results(parameter='level', spatial_node=0).tolist()
                            
                new_stage_hyd_times.append(timesss)
                new_stage_hyd_stages.append(stages)
            
            #############
            
            
        stage_hyd_times = deepcopy(new_stage_hyd_times)
        stage_hyd_stages = deepcopy(new_stage_hyd_stages)
        
        #print(f"### Finished epoch {e+1} ###")
            