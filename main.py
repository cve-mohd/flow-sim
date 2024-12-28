from river import River
from preissmann_model import PreissmannModel
from settings import *


# Declare a 'River' object, specifying the geometric attributes of the river.
blue_nile = River(BED_SLOPE, MANNING_COEFF, WIDTH, LENGTH)

# Declare a 'PreissmannModel' object, specifying the scheme parameters.
p_model = PreissmannModel(blue_nile, PREISSMANN_BETA, TIME_STEP, SPATIAL_STEP)

# Solve the scheme using a specified tolerance.
p_model.solve(DURATION, TOLERANCE)

# Save the results.
p_model.save_results(RESULTS_SIZE)
