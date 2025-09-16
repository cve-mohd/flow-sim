from reach import Reach
from boundary import Boundary
from utility import Hydrograph
from case_study_settings import *
from settings import trapzoid_hydrograph
from custom_functions import import_geometry

hyd_const = Hydrograph(function=trapzoid_hydrograph)
hyd = Hydrograph()

hyd.load_csv('hydrograph.csv')

GERD = Boundary(initial_depth=0,
                condition='flow_hydrograph',
                bed_level=495,
                chainage=0,
                hydrograph=hyd)

Roseires = Boundary(initial_depth=roseires_level-470.54,
                    condition='fixed_depth',
                    bed_level=470.54,
                    chainage=120000)

Roseires.set_storage(0, used_roseires_rc)

GERD_Roseires_system = Reach(width=250,
                             initial_flow_rate=1562.5,
                             roughness=wet_n,
                             dry_roughness=dry_n,
                             upstream_boundary=GERD,
                             downstream_boundary=Roseires)

widths, width_ch, levels, level_ch, x, y, coords_ch = import_geometry("geometry.xlsx")

GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=levels, chainages=level_ch)
GERD_Roseires_system.set_intermediate_widths(widths=widths, chainages=width_ch)
GERD_Roseires_system.set_coords(coords=zip(x, y), chainages=coords_ch)

from preissmann import PreissmannSolver

solver = PreissmannSolver(reach=GERD_Roseires_system,
                          theta=theta,
                          time_step=preissmann_dt,
                          spatial_step=dx,
                          enforce_physicality=enforce_physicality)

GERD_Roseires_system.downstream_boundary.storage_area = total_channel_area - GERD_Roseires_system.surface_area

solver.run(duration=sim_duration, verbose=0); solver.save_results()
print('Success.')
"""
from numpy import arctan, pi
from custom_functions import draw, import_geometry, export_banks

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