from river import River
from preissmann_model import PreissmannModel


# Declare a 'River' object, specifying the geometric attributes of the river.
blue_nile = River(bed_slope=12.5/15000, manning_co=.027, width=250., length=15000.)

# Declare a 'PreissmannModel' object, specifying the scheme parameters.
p_model = PreissmannModel(river=blue_nile, beta=0.6, delta_t=3600, delta_x=1000)

# Set the simulation duration.
duration_hours = 24

# Solve the scheme using a specified tolerance.
p_model.solve(duration=duration_hours*3600, tolerance=1e-4)

# Save the results.
p_model.save_results(time_steps_to_save=duration_hours, space_points_to_save=-1)
