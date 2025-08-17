from channel import Channel
from boundary import Boundary
from settings import trapzoid_hydrograph

us = Boundary(initial_depth=3,
              condition='flow_hydrograph',
              bed_level=495,
              flow_hydrograph_function=trapzoid_hydrograph)

ds = Boundary(initial_depth=3,
              condition='normal_depth',
              bed_level=482.5)

example_channel = Channel(length = 120000,
                          width = 250,
                          initial_flow_rate = 1562.5,
                          manning_co = 0.029,
                          upstream_boundary = us,
                          downstream_boundary = ds)

from preissmann import PreissmannSolver

p_model = PreissmannSolver(channel=example_channel,
                           theta=0.8,
                           time_step=3600,
                           spatial_step=1000)

p_model.run(duration=3600*72, verbose=3)
p_model.save_results()

"""
from lax import LaxSolver

l_model = LaxSolver(channel=example_channel,
                    time_step=50,
                    spatial_step=1000)

l_model.run(duration=3600*72)
l_model.save_results()
"""
