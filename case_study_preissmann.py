from reach import Reach
from boundary import Boundary
from case_study_settings import *
from preissmann import PreissmannSolver
from copy import deepcopy
from utility import Hydrograph

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
            
        ds = Boundary(initial_depth=downstream_depth, condition='flow_hydrograph' , bed_level=reach['ds_bed_level'], chainage=reach['chainage']+reach['length'])
        
        reach_instance = Reach(width=reach['width'], initial_flow_rate=initial_flow,
                                channel_roughness=wet_roughness, upstream_boundary=us, downstream_boundary=ds)
        reach_instance.initialize_conditions(20)
        
        upstream_depth, _ = reach_instance.initial_conditions[0]
        upstream_depth = upstream_depth / reach['width']
        
        depth_at_interface.insert(0, upstream_depth)
        level_at_interface.insert(0, upstream_depth + reach['us_bed_level'])

    stage_hydrographs = []

    inflow_hydrograph = Hydrograph()
    inflow_hydrograph.load_csv('hydrograph.csv')

    for e in range(epochs):
                
        new_stage_hydrographs = []
            
        for reach in reaches:
            #h_us = reach['us_water_level'] - reach['us_bed_level']        
            us = Boundary(initial_depth=0, condition='flow_hydrograph' , bed_level=reach['us_bed_level'], chainage=reach['chainage'])
                
            if reach['id'] == 0:
                us.hydrograph = inflow_hydrograph
            else:
                us.hydrograph = Hydrograph()
                us.hydrograph.set_values(times = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)],
                                         values = solver.get_results('flow_rate', spatial_node=-1).tolist())
            
            if reach['id'] == len(reaches) - 1:
                h_ds = roseires_level_ - reach['ds_bed_level']
                ds = Boundary(initial_depth=h_ds, condition='fixed_depth', bed_level=reach['ds_bed_level'], chainage=reach['chainage'] + reach['length'])
                ds.set_storage(storage_area=storage_area, storage_exit_rating_curve=used_roseires_rc)
            else:
                h_ds = depth_at_interface[reach['id']]
                if e == 0:
                    ds = Boundary(initial_depth=h_ds, condition='fixed_depth', bed_level=reach['ds_bed_level'], chainage=reach['chainage'] + reach['length'])
                else:
                    ds = Boundary(initial_depth=h_ds, condition='stage_hydrograph', bed_level=reach['ds_bed_level'], chainage=reach['chainage'] + reach['length'])
                    ds.hydrograph = stage_hydrographs[reach['id'] + 1]
                
            channel = Reach(width = reach['width'],
                            initial_flow_rate = initial_flow,
                            channel_roughness = wet_roughness,
                            upstream_boundary = us,
                            downstream_boundary = ds,
                            floodplain_roughness=dry_roughness,
                            bankful_depth=h_ds)

            solver = PreissmannSolver(channel, theta, preissmann_time_step, spatial_resolution * reach['length'], enforce_physicality)
            
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
                
                stage_hyd = Hydrograph()
                stage_hyd.set_values(timesss, stages)
                
                new_stage_hydrographs.append(stage_hyd)
            
            #############
            
        stage_hydrographs = deepcopy(new_stage_hydrographs)
        
        print(f"### Finished epoch {e+1} ###")
            