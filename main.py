from reach import Reach
from boundary import Boundary
from settings import trapzoid_hydrograph
from utility import Hydrograph

hyd = Hydrograph(trapzoid_hydrograph)

us = Boundary(initial_depth=3,
              condition='flow_hydrograph',
              bed_level=495,
              chainage=0,
              hydrograph=hyd)

ds = Boundary(initial_depth=3,
              condition='normal_depth',
              bed_level=482.5,
              chainage=15000)

example_channel = Reach(width = 250,
                        initial_flow_rate = 1562.5,
                        channel_roughness = 0.029,
                        upstream_boundary = us,
                        downstream_boundary = ds)

from preissmann import PreissmannSolver

solver = PreissmannSolver(reach=example_channel,
                          theta=0.8,
                          time_step=3600,
                          spatial_step=1000,
                          enforce_physicality=False)

solver.run(duration=3600*72, verbose=2)
solver.save_results()

"""
from lax import LaxSolver

l_model = LaxSolver(channel=example_channel,
                    time_step=50,
                    spatial_step=1000)

l_model.run(duration=3600*72)
l_model.save_results()
"""
