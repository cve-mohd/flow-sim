from src.hydromodel.channel import Channel
from src.hydromodel.boundary import Boundary
from src.hydromodel.lax import LaxSolver
from src.hydromodel.hydrograph import Hydrograph
from ..akbari_firoozi.settings import *

hyd = Hydrograph(hydrograph)

us = Boundary(condition='flow_hydrograph',
              bed_level=S_0*length,
              chainage=0,
              hydrograph=hyd)

ds = Boundary(condition='normal_depth',
              bed_level=0,
              chainage=length)

example_channel = Channel(width=width,
                        initial_flow=initial_flow,
                        roughness=roughness,
                        upstream_boundary=us,
                        downstream_boundary=ds,
                        interpolation_method='steady-state')

solver = LaxSolver(channel=example_channel,
                   time_step=lax_dt,
                   spatial_step=spatial_step,
                   simulation_time=duration,
                   regularization=False,
                   secondary_BC=lax_secondary_bc)

solver.run(verbose=0)
solver.save_results(folder_path='cases\\akbari_firoozi\\results\\lax')
print('Simulation finished successfuly.')
