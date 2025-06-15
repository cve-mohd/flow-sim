from river import River
from boundary import Boundary
from settings import *
from utility import RatingCurve
from preissmann import PreissmannSolver

for reach in reaches:
    ## First pass. Here we build the rating curves ##
    
    h_us = reach['us_water_level'] - reach['us_bed_level']
    h_ds = reach['ds_water_level'] - reach['ds_bed_level']
    
    us = Boundary(h_us, 'hydrograph' , reach['us_bed_level'], hydrograph_function=f, chainage=reach['chainage'])
    ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])

    channel = River(length = reach['length'],
                    width = reach['width'],
                    initial_flow_rate = initial_flow_rate,
                    bed_slope = 'real',
                    manning_co = manning_coefficient,
                    upstream_boundary = us,
                    downstream_boundary = ds)
    
    ###
    
    p_model = PreissmannSolver(channel, theta, preissmann_time_step, 0.05 * channel.total_length)
    p_model.run(hydrograph_duration, auto=False, verbose=0)
    
    ###
    
    depths = p_model.get_results('h', spatial_node=0)
    stages = depths + channel.upstream_boundary.bed_level
    discharges = p_model.get_results('q', spatial_node=0)
    
    if reach['id'] < len(lengths) - 2:
        d = 2
    else:
        d = 1
    
    rating_curve = RatingCurve()
    rating_curve.fit(discharges=discharges, stages=stages, degree=d)
    
    rating_curves.append(rating_curve)

epochs = 1

for e in range(epochs):
    
    new_rating_curves = []
    
    for reach in reaches:
        ## Second pass. Use the rating curves as d/s bc ##
        
        h_us = reach['us_water_level'] - reach['us_bed_level']
        h_ds = reach['ds_water_level'] - reach['ds_bed_level']
        
        us = Boundary(h_us, 'hydrograph' , reach['us_bed_level'], chainage=reach['chainage'])
            
        if reach['id'] == 0:
            us.set_hydrograph(f)
        else:
            times = [t for t in range(0, hydrograph_duration + p_model.time_step, p_model.time_step)]
            discharges = p_model.get_results('q', spatial_node=-1).tolist()
            discharges = discharges[:len(times)]
            us.build_hydrograph(times, discharges)
        
        if reach['id'] == len(lengths) - 1:
            ds = Boundary(h_ds, 'fixed_depth', reach['ds_bed_level'])
            ds.set_storage_behavior(storage_area=storage_area, storage_exit_rating_curve=roseires_rating_curve)
        else:
            rc = rating_curves[reach['id']+1]
            ds = Boundary(h_ds, 'rating_curve', reach['ds_bed_level'], rating_curve=rc)
            
        channel = River(length = reach['length'],
                        width = reach['width'],
                        initial_flow_rate = initial_flow_rate,
                        bed_slope = 'real',
                        manning_co = manning_coefficient,
                        upstream_boundary = us,
                        downstream_boundary = ds)

        p_model = PreissmannSolver(channel, theta, preissmann_time_step, 0.05 * channel.total_length)       
        
        p_model.run(sim_duration, auto=False, tolerance=tolerance, verbose=0)
        
        ##############
        
        if e < epochs - 1:
            depths = p_model.get_results('h', spatial_node=0)
            stages = depths + channel.upstream_boundary.bed_level
            discharges = p_model.get_results('q', spatial_node=0)
            
            if reach['id'] < len(lengths) - 2:
                d = 2
            else:
                d = 1
    
            rating_curve = RatingCurve()
            rating_curve.fit(discharges=discharges, stages=stages, degree=d)
            
            new_rating_curves.append(rating_curve)
        else:
            p_model.save_results(results_size, 'Results//Preissmann//Reach ' + str( reach['id'] ))
        
        #############
                
        
    rating_curves = new_rating_curves
            
print("Success!")
