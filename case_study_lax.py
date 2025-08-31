from reach import Reach
from boundary import Boundary
from case_study_settings import *
from utility import RatingCurve
from lax import LaxSolver
import time

start = time.perf_counter()

for reach in reaches:
    ## First pass. Here we build the rating curves ##
    
    h_us = reach['us_water_level'] - reach['us_bed_level']
    h_ds = reach['ds_water_level'] - reach['ds_bed_level']
    
    us = Boundary(h_us, 'flow_hydrograph' , reach['us_bed_level'], flow_hydrograph_function=trapzoid_hydrograph, chainage=reach['chainage'])
    ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])

    channel = Reach(length = reach['length'],
                    width = reach['width'],
                    initial_flow_rate = initial_flow,
                    bed_slope = 'real',
                    manning_co = manning_coefficient,
                    upstream_boundary = us,
                    downstream_boundary = ds)
    
    solver = LaxSolver(channel, lax_time_step, spatial_resolution * channel.total_length, secondary_boundary_conditions=lax_secondary_bc)
    solver.run(int(time_to_peak), verbose=0)
    
    depths = solver.get_results('h', spatial_node=0)
    stages = depths + channel.upstream_boundary.bed_level
    discharges = solver.get_results('q', spatial_node=0)
    
    rating_curve = RatingCurve()
    rating_curve.fit(discharges=discharges[::3600//lax_time_step], stages=stages[::3600//lax_time_step], degree=2)
    
    rating_curves.append(rating_curve)

number_of_iterations = 1
for iteration in range(number_of_iterations):
    
    new_rating_curves = []
    
    for reach in reaches:
        ## Second pass. Use the rating curves as d/s bc ##
        
        h_us = reach['us_water_level'] - reach['us_bed_level']
        h_ds = reach['ds_water_level'] - reach['ds_bed_level']
        
        us = Boundary(h_us, 'flow_hydrograph' , reach['us_bed_level'], chainage=reach['chainage'])
            
        if reach['id'] == 0:
            us.set_flow_hydrograph(trapzoid_hydrograph)
        else:
            times = [t for t in range(0, sim_duration + solver.time_step, solver.time_step)]
            discharges = solver.get_results('q', spatial_node=-1).tolist()
            discharges = discharges[:len(times)]
            us.build_flow_hydrograph(times, discharges)
        
        if reach['id'] < len(reaches) - 1:
            rc = rating_curves[reach['id']+1]
            ds = Boundary(h_ds, 'rating_curve', reach['ds_bed_level'], rating_curve=rc)
        else:
            ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
            ds.set_storage_behavior(storage_area=storage_area, storage_exit_rating_curve=roseires_spillway_rating_curve)
            
        channel = Reach(length = reach['length'], 
                        width = reach['width'],
                        initial_flow_rate = initial_flow,
                        manning_co = manning_coefficient,
                        upstream_boundary = us,
                        downstream_boundary = ds)

        solver = LaxSolver(channel, lax_time_step, spatial_resolution * channel.total_length, secondary_boundary_conditions=lax_secondary_bc)
                    
        solver.run(sim_duration, verbose=0)
        print(f'Finished reach {reach['id']}')
                    
        ##############
        
        if iteration < number_of_iterations - 1:
            depths = solver.get_results('h', spatial_node=0)
            stages = depths + channel.upstream_boundary.bed_level
            discharges = solver.get_results('q', spatial_node=0)
                    
            rating_curve = RatingCurve()
            rating_curve.fit(discharges=discharges[::3600//lax_time_step], stages=stages[::3600//lax_time_step], degree=1)
            
            new_rating_curves.append(rating_curve)
        else:
            solver.save_results((solver.total_sim_duration//3600 + 1, -1), 'Results//Lax//Reach ' + str( reach['id'] ))
            
        
        #############
                
    rating_curves = new_rating_curves
    print(f'Finished epoch {iteration}')
            
print("Success!")

end = time.perf_counter()

with open('Results//Lax//cpu_time.txt', 'w') as file:
    file.write(f'{end - start:.4f} s')
    