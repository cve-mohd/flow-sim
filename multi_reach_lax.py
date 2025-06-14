from river import River
from boundary import Boundary
from settings import *
from utility import RatingCurve
from lax import LaxSolver

sec_bc = ('constant', 'constant')

duration = 3600 * 18

def f(t):
    if t <= duration:
        return 1562.5 + (10000 - 1562.5) * t/(duration)
    else:
        return 10000

lengths        = [16000, 32000, 37000, 8000, 27000]
widths         = [250  , 650  , 1500 , 3000, 6000]

us_water_level = [502.5, 490  , 490  , 490  , 490]
ds_water_level = [490  , 490  , 490  , 490  , 490]

us_bed_levels  = [495.0, 482.5, 479.6, 476.3, 475.5]
ds_bed_levels  = [482.5, 479.6, 476.3, 475.5, 473.1]

chainages      = [sum(lengths[:i]) for i in range(len(lengths))]

rating_curves = []
rs_stages = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493]
rs_discharges = [7147, 7420, 7686, 7945, 8197, 8449, 8701, 8946, 9184, 9422, 9653, 9891, 10122, 10346]
roseires_rating_curve = RatingCurve()

roseires_rating_curve.fit(discharges=rs_discharges, stages=rs_stages)
reservoir_area = 30e6

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

    solver.run(duration, verbose=0)
    
    depths = solver.get_results('h', spatial_node=0)
    stages = depths + channel.upstream_boundary.bed_level
    discharges = solver.get_results('q', spatial_node=0)
    
    rating_curve = RatingCurve()
    rating_curve.fit(discharges=discharges[::3600//lax_time_step], stages=stages[::3600//lax_time_step])
    
    rating_curves.append(rating_curve)

number_of_iterations = 1
for iteration in range(number_of_iterations):
    
    if iteration == number_of_iterations - 1:
        duration = DURATION
        
    new_rating_curves = []
    
    for reach in reaches:
        ## Second pass. Use the rating curves as d/s bc ##
        
        h_us = reach['us_water_level'] - reach['us_bed_level']
        h_ds = reach['ds_water_level'] - reach['ds_bed_level']
        
        us = Boundary(h_us, 'hydrograph' , reach['us_bed_level'], chainage=reach['chainage'])
            
        if reach['id'] == 0:
            us.set_hydrograph(f)
        else:
            times = [t for t in range(0, duration + solver.time_step, solver.time_step)]
            discharges = solver.get_results('q', spatial_node=-1)
            us.build_hydrograph(times, discharges.tolist())
        
        if reach['id'] < len(reaches) - 1:
            rc = rating_curves[reach['id']+1]
            ds = Boundary(h_ds, 'rating_curve', reach['ds_bed_level'], rating_curve=rc)
        else:
            ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
            ds.set_storage_behavior(reservoir_area=reservoir_area, reservoir_exit_rating_curve=roseires_rating_curve)
            
        channel = River(length = reach['length'], 
                        width = reach['width'],
                        initial_flow_rate = 1562.5,
                        manning_co = 0.027,
                        upstream_boundary = us,
                        downstream_boundary = ds)

        lax_time_step = 5 * int(float(reaches[-1]['ds_water_level'] - reaches[-1]['ds_bed_level']) // float(h_ds))
        solver = LaxSolver(channel, lax_time_step, 0.01 * channel.total_length, secondary_boundary_conditions=sec_bc)
                    
        solver.run(duration, verbose=0)
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
