from src.reach import Reach
from src.boundary import Boundary
from src.utility import Hydrograph, LumpedStorage
from cases.gerd_roseires.settings import *
from cases.gerd_roseires.custom_functions import import_geometry

hyd = Hydrograph()
hyd.load_csv('cases\\gerd_roseires\\input_data\\hydrograph.csv')
widths, width_ch, levels, level_ch, x, y, coords_ch = import_geometry("cases\\gerd_roseires\\input_data\\geometry.xlsx")

gerd_bed_level = levels[0]
gerd_chainage = min(width_ch+level_ch+coords_ch)

roseires_bed_level = levels[-1]
roseires_chainage = max(width_ch+level_ch+coords_ch)

GERD = Boundary(condition='flow_hydrograph',
                bed_level=gerd_bed_level,
                chainage=gerd_chainage,
                hydrograph=hyd)

ds_depth = roseires_level-roseires_bed_level
Roseires = Boundary(initial_depth=ds_depth,
                    condition='fixed_depth',
                    bed_level=roseires_bed_level,
                    chainage=roseires_chainage)

roseires_storage = LumpedStorage(roseires_area, roseires_level, used_roseires_rc)
Roseires.set_lumped_storage(roseires_storage)

GERD_Roseires_system = Reach(width=widths[0],
                             initial_flow_rate=hyd.get_at(0),
                             roughness=wet_n,
                             dry_roughness=dry_n,
                             upstream_boundary=GERD,
                             downstream_boundary=Roseires)

GERD_Roseires_system.set_coords(coords=zip(x, y), chainages=coords_ch)
GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=levels, chainages=level_ch)
GERD_Roseires_system.set_intermediate_widths(widths=widths, chainages=width_ch)

from src.preissmann import PreissmannSolver

solver = PreissmannSolver(reach=GERD_Roseires_system,
                          theta=theta,
                          time_step=preissmann_dt,
                          spatial_step=dx,
                          enforce_physicality=enforce_physicality)

GERD_Roseires_system.downstream_boundary.lumped_storage.surface_area = roseires_area - GERD_Roseires_system.surface_area

solver.run(duration=sim_duration, verbose=0)
solver.save_results(path='cases\\gerd_roseires\\results')

print('Success.')
"""
from numpy import arctan, pi
from src.custom_functions import draw, import_geometry, export_banks

x1, x2 = GERD_Roseires_system.x_coords[0], GERD_Roseires_system.x_coords[1]
y1, y2 = GERD_Roseires_system.y_coords[0], GERD_Roseires_system.y_coords[1]

theta = pi - arctan(
    abs(y2-y1)/abs(x2-x1)
)

left_x, left_y, right_x, right_y = draw(chainages=GERD_Roseires_system.chainages,
                                        widths=GERD_Roseires_system.widths,
                                        curvature=GERD_Roseires_system.curv,
                                        x0=GERD_Roseires_system.x_coords[0],
                                        y0=GERD_Roseires_system.y_coords[0],
                                        theta0=theta)

export_banks(left_x, left_y, right_x, right_y)
"""