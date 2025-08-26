from reach import Channel
from boundary import Boundary
from case_study_settings import *
from preissmann import PreissmannSolver
from copy import deepcopy

def run(peak_inflow, initial_roseires_stage):
    
    ############                Hydrologic Parameters               ############

    us_water_level = [498.1, initial_roseires_stage  , initial_roseires_stage  , initial_roseires_stage  , initial_roseires_stage]
    ds_water_level = [initial_roseires_stage  , initial_roseires_stage  , initial_roseires_stage  , initial_roseires_stage  , initial_roseires_stage]

    initial_flow = 1562.5
    peak_flow = peak_inflow

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
        
    ############################################################################
    
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

    ############################################################################

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
                discharges = solver.get_results('flow_rate', spatial_node=-1).tolist()            
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
                main_folder_name = f'inf{peak_inflow}_stg{initial_roseires_stage}'
                path = f'results//{main_folder_name}//reach_{reach['id']}'
                
                solver.save_results(
                    size = (solver.total_sim_duration//3600 + 1, -1),
                    path = path
                    )
            else:
                timesss = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
                stages = solver.get_results(parameter='level', spatial_node=0).tolist()
                            
                new_stage_hyd_times.append(timesss)
                new_stage_hyd_stages.append(stages)
            
            #############
            
            
        stage_hyd_times = deepcopy(new_stage_hyd_times)
        stage_hyd_stages = deepcopy(new_stage_hyd_stages)
        
        #print(f"### Finished epoch {e+1} ###")
                
    print(f'Finished case {initial_roseires_stage}, {peak_inflow}')
