import sys
from pathlib import Path

# add repo root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.reach import Reach
from src.boundary import Boundary
from src.settings import trapzoid_hydrograph
from src.utility import Hydrograph

hyd = Hydrograph(trapzoid_hydrograph)

us = Boundary(initial_depth=3,
              condition='flow_hydrograph',
              bed_level=495,
              chainage=0,
              hydrograph=hyd)

ds = Boundary(initial_depth=5,
              condition='fixed_depth',
              bed_level=495,
              chainage=26000)

example_channel = Reach(width = 250,
                        initial_flow_rate = 1562.5,
                        roughness = 0.029,
                        upstream_boundary = us,
                        downstream_boundary = ds)

example_channel.set_intermediate_bed_levels([510], [8000])
example_channel.set_intermediate_widths([400], [26000])

from preissmann import PreissmannSolver

solver = PreissmannSolver(reach=example_channel,
                          theta=0.8,
                          time_step=3600,
                          spatial_step=1000,
                          enforce_physicality=False)

solver.run(duration=3600*24, verbose=0)
solver.save_results()

"""
from lax import LaxSolver

l_model = LaxSolver(channel=example_channel,
                    time_step=50,
                    spatial_step=1000)

l_model.run(duration=3600*72)
l_model.save_results()
"""
