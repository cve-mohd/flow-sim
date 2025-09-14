from reach import Reach
from boundary import Boundary
from case_study_settings import *
from preissmann import PreissmannSolver
from utility import Hydrograph

def run(roseires_level_, n_epochs = epochs):

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
                                roughness=wet_roughness, upstream_boundary=us, downstream_boundary=ds)
        reach_instance.initialize_conditions(20)
        
        upstream_depth, _ = reach_instance.initial_conditions[0]
        upstream_depth = upstream_depth / reach['width']
        
        depth_at_interface.insert(0, upstream_depth)
        level_at_interface.insert(0, upstream_depth + reach['us_bed_level'])

    inflow_hydrograph = Hydrograph()
    inflow_hydrograph.load_csv('hydrograph.csv')

    # forward pass: compute interface flows
    flow_hydrographs = []

    for i, reach in enumerate(reaches):
        # upstream BC
        us = Boundary(initial_depth=0,
                      condition='flow_hydrograph',
                      bed_level=reach['us_bed_level'],
                      chainage=reach['chainage'])

        if reach['id'] == 0:
            us.hydrograph = inflow_hydrograph
        else:
            # use saved interface hydrograph from previous reach
            us.hydrograph = flow_hydrographs[i-1]

        # downstream BC: always fixed depth in forward pass
        if reach['id'] == 4:
            ds_depth = roseires_level_ - reach['ds_bed_level']
        else:
            ds_depth = depth_at_interface[ reach['id'] ]
            
        ds = Boundary(initial_depth=ds_depth,
                      condition='fixed_depth',
                      bed_level=reach['ds_bed_level'],
                      chainage=reach['chainage'] + reach['length'])

        channel = Reach(upstream_boundary = us,
                        downstream_boundary = ds,
                        width = reach['width'],
                        initial_flow_rate = initial_flow,
                        roughness = wet_roughness,
                        dry_roughness = dry_roughness,
                        bankful_depth = ds_depth)

        solver = PreissmannSolver(channel, theta, preissmann_time_step,
                                  spatial_resolution * reach['length'], enforce_physicality)
        solver.run(sim_duration, auto=False, tolerance=tolerance, verbose=0)

        # save downstream flow hydrograph
        times = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
        flows = solver.get_results('flow_rate', spatial_node=-1).tolist()

        flow_h = Hydrograph()
        flow_h.set_values(times, flows)
        flow_hydrographs.append(flow_h)

    # backward pass: compute interface stages
    stage_hydrographs = [None] * len(reaches)

    for e in range(n_epochs):
        for i in reversed(range(len(reaches))):
            reach = reaches[i]

            # upstream BC: flow hydrograph from forward pass
            us = Boundary(initial_depth=0,
                        condition='flow_hydrograph',
                        bed_level=reach['us_bed_level'],
                        chainage=reach['chainage'])
            
            if reach['id'] == 0:
                us.hydrograph = inflow_hydrograph
            else:
                us.hydrograph = flow_hydrographs[i-1]

            # downstream BC
            if reach['id'] == len(reaches) - 1:
                ds_depth = roseires_level_ - reach['ds_bed_level']
                ds = Boundary(initial_depth=ds_depth,
                            condition='fixed_depth',
                            bed_level=reach['ds_bed_level'],
                            chainage=reach['chainage'] + reach['length'])
                ds.set_storage(storage_area, used_roseires_rc)
                
            else:
                ds_depth = depth_at_interface[i]
                ds = Boundary(initial_depth=ds_depth,
                              condition='stage_hydrograph',
                              bed_level=reach['ds_bed_level'],
                              chainage=reach['chainage'] + reach['length'])
                ds.hydrograph = stage_hydrographs[i+1]

            channel = Reach(upstream_boundary = us,
                            downstream_boundary = ds,
                            width = reach['width'],
                            initial_flow_rate = initial_flow,
                            roughness = wet_roughness,
                            dry_roughness = dry_roughness,
                            bankful_depth = ds_depth)

            solver = PreissmannSolver(channel, theta, preissmann_time_step,
                                    spatial_resolution * reach['length'], enforce_physicality)
            solver.run(sim_duration, auto=False, tolerance=tolerance, verbose=0)

            # save upstream stage hydrograph
            times = [t for t in range(0, solver.total_sim_duration + solver.time_step, solver.time_step)]
            stages = solver.get_results('level', spatial_node=0).tolist()
            flows = solver.get_results('flow_rate', spatial_node=-1).tolist()

            stage_h = Hydrograph()
            stage_h.set_values(times, stages)
            stage_hydrographs[i] = stage_h
            
            flow_h = Hydrograph()
            flow_h.set_values(times, flows)
            flow_hydrographs[i] = flow_h

            if e == n_epochs - 1:
                solver.save_results(path=f'results/{roseires_level_}/reach_{reach["id"]}')
                
        print(f"Epoch {e+1} finished.")

    #print("Two-pass simulation finished.")
