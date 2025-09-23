from src.channel import Channel
from src.boundary import Boundary
from src.utility import Hydrograph, LumpedStorage
from src.preissmann import PreissmannSolver
from cases.gerd_roseires.settings import *
from cases.gerd_roseires.custom_functions import import_geometry, import_area_curve

hyd = Hydrograph()
hyd.load_csv('cases\\gerd_roseires\\input_data\\hydrograph.csv')

widths, width_ch, levels, level_ch, coords, coords_ch = import_geometry("cases\\gerd_roseires\\input_data\\geometry.xlsx")

gerd_bed_level = levels[0]
gerd_chainage = min([width_ch.min(), level_ch.min(), coords_ch.min()])

roseires_bed_level = levels[-1]
roseires_chainage = max([width_ch.min(), level_ch.min(), coords_ch.min()])

GERD = Boundary(condition='flow_hydrograph',
                bed_level=gerd_bed_level,
                chainage=gerd_chainage,
                hydrograph=hyd)

ds_depth = initial_roseires_level-roseires_bed_level
Roseires = Boundary(initial_depth=ds_depth,
                    condition='fixed_depth',
                    bed_level=roseires_bed_level,
                    chainage=roseires_chainage)

roseires_storage = LumpedStorage(None, initial_roseires_level, used_roseires_rc)
curve = import_area_curve(path='cases\\gerd_roseires\\input_data\\roseires_geometry.xlsx')
roseires_storage.set_area_curve(curve)
Roseires.set_lumped_storage(roseires_storage)

GERD_Roseires_system = Channel(width=widths[0],
                             initial_flow=hyd.get_at(0),
                             roughness=wet_n,
                             dry_roughness=dry_n,
                             upstream_boundary=GERD,
                             downstream_boundary=Roseires)

GERD_Roseires_system.set_coords(coords=coords, chainages=coords_ch)
GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=levels, chainages=level_ch)
GERD_Roseires_system.set_intermediate_widths(widths=widths, chainages=width_ch)

solver = PreissmannSolver(channel=GERD_Roseires_system,
                          theta=theta,
                          time_step=preissmann_dt,
                          spatial_step=dx,
                          simulation_time=sim_duration,
                          regularization=enforce_physicality)

solver.run(verbose=0, tolerance=tolerance)
solver.save_results(folder_path='cases\\gerd_roseires\\results')

print("Simulation finished successfully.")
