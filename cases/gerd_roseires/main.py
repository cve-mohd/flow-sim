from src.reach import Reach
from src.boundary import Boundary
from src.utility import Hydrograph, LumpedStorage
from cases.gerd_roseires.settings import *
from cases.gerd_roseires.custom_functions import import_geometry, import_area_curve

hyd = Hydrograph(trapzoid_hydrograph)
#hyd.load_csv('cases\\gerd_roseires\\input_data\\hydrograph.csv')
print('Loaded inflow hydrograph.')

widths, width_ch, levels, level_ch, x, y, coords_ch = import_geometry("cases\\gerd_roseires\\input_data\\geometry.xlsx")
print('Imported geometry.')

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

roseires_storage = LumpedStorage(None, roseires_level, used_roseires_rc)
curve = import_area_curve(path='cases\\gerd_roseires\\input_data\\roseires_geometry.xlsx')
roseires_storage.set_area_curve(curve, alpha=1, beta=0)
Roseires.set_lumped_storage(roseires_storage)

print('Created boundaries.')

GERD_Roseires_system = Reach(width=widths[0],
                             initial_flow_rate=hyd.get_at(0),
                             roughness=wet_n,
                             dry_roughness=dry_n,
                             upstream_boundary=GERD,
                             downstream_boundary=Roseires)

GERD_Roseires_system.set_coords(coords=zip(x, y), chainages=coords_ch)
GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=levels, chainages=level_ch)
GERD_Roseires_system.set_intermediate_widths(widths=widths, chainages=width_ch)

print('Created channel.')

from src.preissmann import PreissmannSolver

solver = PreissmannSolver(reach=GERD_Roseires_system,
                          theta=theta,
                          time_step=preissmann_dt,
                          spatial_step=dx,
                          enforce_physicality=enforce_physicality)

print('Initialized solver.')

solver.run(duration=sim_duration, verbose=2, auto=False, tolerance=tolerance)
print('Finished simulation.')
solver.save_results(path='cases\\gerd_roseires\\results')
print('Saved results.')

from numpy import interp

init_vol = interp(roseires_level,
                  [480, 481, 482, 484, 486, 488, 490, 492],
                  [1699, 1970, 2282, 3000, 3847, 4824, 5909, 7114])
new_vol = init_vol + (solver.get_results(parameter='flow_rate', spatial_node=-1).sum() - sum(solver.results['outflow'])) * preissmann_dt * 1e-6
correct_amp = interp(new_vol,
                     [1699, 1970, 2282, 3000, 3847, 4824, 5909, 7114],
                     [480, 481, 482, 484, 486, 488, 490, 492]) - roseires_level

print(f'Increase = {solver.peak_amplitudes[-1]} m')
print(f'Correct increse = {correct_amp} m')
"""
init_vol = interp(roseires_level,
                  [480, 481, 482, 484, 486, 488, 490, 492],
                  [1699, 1970, 2282, 3000, 3847, 4824, 5909, 7114])
new_vol = init_vol + net_added_vol
correct_amp = interp(new_vol,
                     [1699, 1970, 2282, 3000, 3847, 4824, 5909, 7114],
                     [480, 481, 482, 484, 486, 488, 490, 492]) - roseires_level

print(f'Increase = {solver.peak_amplitudes[-1]} m')
print(f'Correct increse = {correct_amp} m')


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
#

from numpy import interp
init_vol = interp(roseires_level,
                  [480, 481, 482, 484, 486, 488, 490, 492],
                  [1699, 1970, 2282, 3000, 3847, 4824, 5909, 7114])
new_vol = init_vol + (solver.get_results(parameter='flow_rate', spatial_node=-1).sum() - sum(solver.results['outflow'])) * preissmann_dt * 1e-6
correct_amp = interp(new_vol,
                     [1699, 1970, 2282, 3000, 3847, 4824, 5909, 7114],
                     [480, 481, 482, 484, 486, 488, 490, 492]) - roseires_level

print(f'Increase = {solver.peak_amplitudes[-1]} m')
print(f'Correct increse = {correct_amp} m')

"""
