from river import River
from boundary import Boundary
from settings import *

def f(t):
    if t <= 6*3600:
        return 1562.5 + (10000 - 1562.5) * t/(6*3600)
    else:
        return 10000
    
us = Boundary(7.5, 1562.5, 'hydrograph', 502.5, hydrograph_function=f)
ds = Boundary(7.5, 1562.5, 'fixed_depth', 490, fixed_depth=7.5)

# Declare a 'River' object, specifying the geometric attributes of the river.
blue_nile = River(LENGTH, WIDTH, BED_SLOPE, MANNING_COEFF, us, ds)

if SCHEME == 'preissmann':
    from preissmann_model import PreissmannModel
    
    # Declare a 'PreissmannModel' object, specifying the scheme parameters.
    p_model = PreissmannModel(blue_nile, PREISSMANN_THETA, TIME_STEP, SPATIAL_STEP)

    # Solve the scheme using a specified tolerance.
    p_model.solve(DURATION, TOLERANCE, verbose=3)

    # Save the results.
    p_model.save_results(RESULTS_SIZE)    

elif SCHEME == 'lax':
    from lax_model import LaxModel
    
    # Declare a 'LaxModel' object, specifying the scheme parameters.
    l_model = LaxModel(blue_nile, TIME_STEP, SPATIAL_STEP)

    # Solve the scheme using a specified tolerance.
    l_model.solve(DURATION)

    # Save the results.
    l_model.save_results(RESULTS_SIZE)
    
else:
    raise ValueError("Invalid scheme. Choose 'preissmann' or 'lax'.")  
