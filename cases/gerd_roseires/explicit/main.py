from src.hydromodel.channel import Channel
from src.hydromodel.boundary import Boundary
from src.hydromodel.hydrograph import Hydrograph
from src.hydromodel.preissmann import PreissmannSolver
from ..settings import initial_roseires_level, wet_n, dry_n, theta, spatial_step, time_step, sim_duration, tolerance
from ..custom_functions import import_table, import_hydrograph
from ..roseires_rating_curve import RoseiresRatingCurve

print("Processing input data...")

input_dir = "cases\\gerd_roseires\\explicit\\data\\"
inflow_hyd = Hydrograph(table=import_hydrograph("cases\\gerd_roseires\\data\\inflow_hydrograph.csv"))
bed_profile = import_table(input_dir + "bed_profile.csv", sort_by='chainage')
widths = import_table(input_dir + "width.csv", sort_by='chainage')
coords = import_table(input_dir + "centerline_coords.csv", sort_by='chainage')

gerd_bed_level = bed_profile[0, 1]
gerd_chainage = min([bed_profile[:, 0].min(), widths[:, 0].min(), coords[:, 0].min()])

ds_bed_level = bed_profile[-1, 1]
ds_chainage = max([bed_profile[:, 0].max(), widths[:, 0].max(), coords[:, 0].max()])

GERD = Boundary(condition='flow_hydrograph',
                bed_level=gerd_bed_level,
                chainage=gerd_chainage,
                hydrograph=inflow_hyd)

Roseires = Boundary(initial_depth=initial_roseires_level-ds_bed_level,
                    condition='rating_curve',
                    bed_level=ds_bed_level,
                    chainage=ds_chainage,
                    rating_curve=RoseiresRatingCurve())

GERD_Roseires_system = Channel(width=widths[0, 1],
                               initial_flow=inflow_hyd.get_at(0),
                               roughness=wet_n,
                               dry_roughness=dry_n,
                               upstream_boundary=GERD,
                               downstream_boundary=Roseires)

GERD_Roseires_system.set_coords(coords=coords[:, 1:], chainages=coords[:, 0])
GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=bed_profile[:, 1], chainages=bed_profile[:, 0])
GERD_Roseires_system.set_intermediate_widths(widths=widths[:, 1], chainages=widths[:, 0])

solver = PreissmannSolver(channel=GERD_Roseires_system,
                          theta=theta,
                          time_step=time_step,
                          spatial_step=spatial_step,
                          simulation_time=sim_duration)

print("Simulation started.")

solver.run(verbose=0, tolerance=tolerance)

print("Saving results...")

solver.save_results(folder_path='cases\\gerd_roseires\\explicit\\results\\')

print("Done.")
