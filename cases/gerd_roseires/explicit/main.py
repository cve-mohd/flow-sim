from src.hydromodel.channel import Channel
from src.hydromodel.boundary import Boundary
from src.hydromodel.hydrograph import Hydrograph
from src.hydromodel.preissmann import PreissmannSolver
from src.hydromodel.cross_section import CrossSection
from ..settings import initial_roseires_level, theta, spatial_step, time_step, sim_duration, tolerance
from ..custom_functions import import_table, import_hydrograph, load_cross_sections
from ..roseires_rating_curve import RoseiresRatingCurve

print("Processing input data...")

input_dir = "cases\\gerd_roseires\\explicit\\data\\"
inflow_hyd = Hydrograph(table=import_hydrograph("cases\\gerd_roseires\\data\\inflow_hydrograph.csv"))
coords = import_table(input_dir + "centerline_coords.csv", sort_by='chainage')

xs_chainages, sections = load_cross_sections(xs_folder='cases\\gerd_roseires\\data\\cross_sections\\', info_csv='cases\\gerd_roseires\\data\\xs_info.csv')

roseires_ch = xs_chainages[-1]
roseires_bed = sections[-1].z_min

#GERD_xs = CrossSection(width=250, bed=495, n=0.3)
GERD_ch = xs_chainages[0]

#sections.insert(0, GERD_xs)
#xs_chainages.insert(0, GERD_ch)

GERD = Boundary(condition='flow_hydrograph',
                hydrograph=inflow_hyd,
                chainage=GERD_ch)

Roseires = Boundary(initial_depth=initial_roseires_level-roseires_bed,
                    condition='rating_curve',
                    bed_level=roseires_bed,
                    rating_curve=RoseiresRatingCurve(),
                    chainage=roseires_ch)

GERD_Roseires_system = Channel(initial_flow=inflow_hyd.get_at(0),
                               upstream_boundary=GERD,
                               downstream_boundary=Roseires)

#GERD_Roseires_system.set_coords(coords=coords[:, 1:], chainages=coords[:, 0])
GERD_Roseires_system.set_cross_sections(chainages=xs_chainages, sections=sections)

solver = PreissmannSolver(channel=GERD_Roseires_system,
                          theta=theta,
                          time_step=time_step,
                          spatial_step=spatial_step,
                          simulation_time=sim_duration)

print("Simulation started.")

#solver.run(verbose=3, tolerance=tolerance)

print("Saving results...")

#solver.save_results(folder_path='cases\\gerd_roseires\\explicit\\results\\')

print("Done.")

# py -m cases.gerd_roseires.explicit.main
import numpy as np
bed_profile = np.array(object=[GERD_Roseires_system.bed_level_at(i=i) for i in range(len(GERD_Roseires_system.xs_at_node))])
print(GERD_Roseires_system.initial_conditions[:, 0] + bed_profile)