from river import River
from boundary import Boundary
from settings import *

def f(t):
    if t <= 24*3600:
        return 1562.5 + (10000 - 1562.5) * t/(24*3600)
    else:
        return 10000

us = Boundary(US_INIT_DEPTH, 'hydrograph', US_INIT_STAGE - US_INIT_DEPTH, hydrograph_function=f)
ds = Boundary(DS_INIT_DEPTH, 'fixed_depth', DS_INIT_STAGE - DS_INIT_DEPTH, fixed_depth=DS_INIT_DEPTH)

# Declare a 'River' object, specifying the geometric attributes of the river.
channel = River(length = LENGTH,
                width = WIDTH,
                initial_flow_rate = FLOW_RATE,
                bed_slope = 'real',
                manning_co = MANNING_COEFF,
                upstream_boundary = us,
                downstream_boundary = ds,
                buffer_length = 0)

if SCHEME == 'preissmann':
    from preissmann_model import PreissmannModel
    
    # Declare a 'PreissmannModel' object, specifying the scheme parameters.
    p_model = PreissmannModel(channel, 0.6, TIME_STEP, SPATIAL_STEP)

    # Solve the scheme using a specified tolerance.
    p_model.solve(DURATION, TOLERANCE, verbose=0)

    # Save the results.
    p_model.save_results((48, -1))               

elif SCHEME == 'lax':
    from lax_model import LaxModel
    
    l_model = LaxModel(channel, TIME_STEP, SPATIAL_STEP)
    l_model.solve(DURATION)
    l_model.save_results(RESULTS_SIZE)
    
else:
    raise ValueError("Invalid scheme. Choose 'preissmann' or 'lax'.")  
