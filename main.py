from river import River
from preissmann_model import PreissmannModel
from lax_model import LaxModel
from settings import *


# Declare a 'River' object, specifying the geometric attributes of the river.
blue_nile = River(BED_SLOPE, MANNING_COEFF, WIDTH, LENGTH)

if SCHEME == 'preissmann':
    # Declare a 'PreissmannModel' object, specifying the scheme parameters.
    p_model = PreissmannModel(blue_nile, PREISSMANN_BETA, TIME_STEP, SPATIAL_STEP)

    # Solve the scheme using a specified tolerance.
    p_model.solve(DURATION, TOLERANCE)

    # Save the results.
    p_model.save_results(RESULTS_SIZE)

elif SCHEME == 'lax':
    # Declare a 'LaxModel' object, specifying the scheme parameters.
    l_model = LaxModel(blue_nile, TIME_STEP, SPATIAL_STEP)

    # Solve the scheme using a specified tolerance.
    l_model.solve(DURATION, approximation=LAX_APPROX)

    # Save the results.
    l_model.save_results(RESULTS_SIZE)
    
else:
    raise ValueError("Invalid scheme. Choose 'preissmann' or 'lax'.")  
