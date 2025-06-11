from river import River
from boundary import Boundary
from settings import *

def f(t):
    hours = 12
    
    if t <= hours*3600:
        return 1562.5 + (10000 - 1562.5) * t/(hours*3600)
    else:
        return 10000

us = Boundary(2.9, 'hydrograph', 495, hydrograph_function=f)
ds = Boundary(7.5, 'fixed_depth', 490 - 7.5, fixed_depth=7.5)

# Declare a 'River' object, specifying the geometric attributes of the river.
channel = River(length = 15000,
                width = 250,
                initial_flow_rate = 1562.5,
                manning_co = 0.027,
                upstream_boundary = us,
                downstream_boundary = ds)

if SCHEME == 'preissmann':
    from preissmann import PreissmannSolver
    
    # Declare a 'PreissmannModel' object, specifying the scheme parameters.
    p_model = PreissmannSolver(channel, 0.6, TIME_STEP, SPATIAL_STEP)

    # Solve the scheme using a specified tolerance.
    p_model.run(3600 * 24, verbose=3)

    # Save the results.
    p_model.save_results((-1, -1))               

elif SCHEME == 'lax':
    from lax import LaxSolver
    
    l_model = LaxSolver(channel, 10, 1000)
    l_model.run(3600 * 24)
    l_model.save_results((25, -1))
    
else:
    raise ValueError("Invalid scheme. Choose 'preissmann' or 'lax'.")  
