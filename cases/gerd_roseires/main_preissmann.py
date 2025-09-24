from src.channel import Channel
from src.boundary import Boundary
from src.utility import Hydrograph, LumpedStorage
from src.preissmann import PreissmannSolver
from cases.gerd_roseires.settings import *
from cases.gerd_roseires.custom_functions import import_geometry, import_area_curve, import_hydrograph

print("Processing input data...")

hyd = Hydrograph(table=import_hydrograph("cases\\gerd_roseires\\input_data\\inflow_hydrograph.csv"))
bed_profile = import_geometry("cases\\gerd_roseires\\input_data\\bed_profile.csv")
widths = import_geometry("cases\\gerd_roseires\\input_data\\width.csv")
coords = import_geometry("cases\\gerd_roseires\\input_data\\centerline_coords.csv")

gerd_bed_level = bed_profile[0, 1]
gerd_chainage = min([bed_profile[:, 0].min(), widths[:, 0].min(), coords[:, 0].min()])

roseires_bed_level = bed_profile[-1, 1]
roseires_chainage = max([bed_profile[:, 0].max(), widths[:, 0].max(), coords[:, 0].max()])

GERD = Boundary(condition='flow_hydrograph',
                bed_level=gerd_bed_level,
                chainage=gerd_chainage,
                hydrograph=hyd)

ds_depth = initial_roseires_level-roseires_bed_level
Roseires = Boundary(initial_depth=ds_depth,
                    condition='fixed_depth',
                    bed_level=roseires_bed_level,
                    chainage=roseires_chainage)

roseires_storage = LumpedStorage(min_stage=initial_roseires_level, rating_curve=used_roseires_rc, solution_boundaries=None)
roseires_storage.set_area_curve(table=import_area_curve('cases\\gerd_roseires\\input_data\\roseires_storage_curve.csv'))

Roseires.set_lumped_storage(roseires_storage)

GERD_Roseires_system = Channel(width=widths[0],
                             initial_flow=hyd.get_at(0),
                             roughness=wet_n,
                             dry_roughness=dry_n,
                             upstream_boundary=GERD,
                             downstream_boundary=Roseires)

GERD_Roseires_system.set_coords(coords=coords[:, 1:], chainages=coords[:, 0])
GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=bed_profile[:, 1], chainages=bed_profile[:, 0])
GERD_Roseires_system.set_intermediate_widths(widths=widths[:, 1], chainages=widths[:, 0])

solver = PreissmannSolver(channel=GERD_Roseires_system,
                          theta=theta,
                          time_step=preissmann_dt,
                          spatial_step=dx,
                          simulation_time=sim_duration,
                          regularization=regularization)

print("Simulation started.")

solver.run(verbose=0, tolerance=tolerance)

print("Saving results...")

solver.save_results(folder_path='cases\\gerd_roseires\\results\\preissmann')

print("Done.")
