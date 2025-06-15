from river import River
from boundary import Boundary
from settings import *
from utility import RatingCurve
from lax import LaxSolver

for reach in reaches:
    ## First pass. Here we build the rating curves ##
    
    h_us = reach['us_water_level'] - reach['us_bed_level']
    h_ds = reach['ds_water_level'] - reach['ds_bed_level']
    
    us = Boundary(h_us, 'hydrograph' , reach['us_bed_level'], hydrograph_function=f, chainage=reach['chainage'])
    ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])

    channel = River(length = reach['length'],
                    width = reach['width'],
                    initial_flow_rate = 1562.5,
                    bed_slope = 'real',
                    manning_co = 0.027,
                    upstream_boundary = us,
                    downstream_boundary = ds)
    
    lax_time_step = 5 * int(float(reaches[-1]['ds_water_level'] - reaches[-1]['ds_bed_level']) // float(h_ds))
    solver = LaxSolver(channel, lax_time_step, 0.1 * channel.total_length, secondary_boundary_conditions=sec_bc)
    solver.run(hydrograph_duration, verbose=0)
    
    depths = solver.get_results('h', spatial_node=0)
    stages = depths + channel.upstream_boundary.bed_level
    discharges = solver.get_results('q', spatial_node=0)
    
    rating_curve = RatingCurve()
    rating_curve.fit(discharges=discharges[::3600//lax_time_step], stages=stages[::3600//lax_time_step])
    
    rating_curves.append(rating_curve)

number_of_iterations = 1
for iteration in range(number_of_iterations):
    
    if iteration == number_of_iterations - 1:
        hydrograph_duration = sim_duration
        
    new_rating_curves = []
    
    for reach in reaches:
        ## Second pass. Use the rating curves as d/s bc ##
        
        h_us = reach['us_water_level'] - reach['us_bed_level']
        h_ds = reach['ds_water_level'] - reach['ds_bed_level']
        
        us = Boundary(h_us, 'hydrograph' , reach['us_bed_level'], chainage=reach['chainage'])
            
        if reach['id'] == 0:
            us.set_hydrograph(f)
        else:
            times = [t for t in range(0, hydrograph_duration + solver.time_step, solver.time_step)]
            discharges = solver.get_results('q', spatial_node=-1)
            us.build_hydrograph(times, discharges.tolist())
        
        if reach['id'] < len(reaches) - 1:
            rc = rating_curves[reach['id']+1]
            ds = Boundary(h_ds, 'rating_curve', reach['ds_bed_level'], rating_curve=rc)
        else:
            ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
            ds.set_storage_behavior(storage_area=storage_area, storage_exit_rating_curve=roseires_rating_curve)
            
        channel = River(length = reach['length'], 
                        width = reach['width'],
                        initial_flow_rate = 1562.5,
                        manning_co = 0.027,
                        upstream_boundary = us,
                        downstream_boundary = ds)

        lax_time_step = 5 * int(float(reaches[-1]['ds_water_level'] - reaches[-1]['ds_bed_level']) // float(h_ds))
        solver = LaxSolver(channel, lax_time_step, 0.01 * channel.total_length, secondary_boundary_conditions=sec_bc)
                    
        solver.run(hydrograph_duration, verbose=0)
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
            solver.save_results((49,-1), 'Results//Lax//Reach ' + str( reach['id'] ))
            
        
        #############
                
    rating_curves = new_rating_curves
    print(f'Finished epoch {iteration}')
            
print("Success!")
