from src.hydromodel.channel import Channel
from src.hydromodel.boundary import Boundary
from src.hydromodel.hydrograph import Hydrograph
from src.hydromodel.preissmann import PreissmannSolver
from .custom_functions import import_table, import_hydrograph, load_trapzoid_xs
from . import settings
from .roseires_rating_curve import RoseiresRatingCurve
from .gerd_discharge import GerdHydrograph

n_main = None
n_fp = None
initial_roseires_level = settings.initial_roseires_level
theta = settings.theta
spatial_step = settings.spatial_step
time_step = settings.time_step
sim_duration = settings.sim_duration
verbose = 1
inflow_hyd_path = settings.inflow_hyd_path
inflow_hyd_func = settings.inflow_hyd_func
coords_path = settings.coords_path
cross_sections_path = settings.cross_sections_path
jammed_spillways = settings.JAMMED_SPILLWAYS
jammed_sluice_gates = settings.JAMMED_SLUICEGATES
gerd_level = settings.initial_gerd_level
with_gerd = True

if verbose > 0:
    print("Processing input data...")

if inflow_hyd_func is None:
    gerd_inflow_hyd = Hydrograph(table=import_hydrograph(inflow_hyd_path))
else:
    gerd_inflow_hyd = Hydrograph(function=inflow_hyd_func)
    
if sim_duration is None:
    if gerd_inflow_hyd.table is None:
        raise ValueError("Simulation duration must be specified.")
    else:
        duration = int(gerd_inflow_hyd.table[-1, 0])
else:
    duration = int(sim_duration)
    
gerd_discharge_hyd = GerdHydrograph()
gerd_discharge_hyd.build(inflow_hydrograph=gerd_inflow_hyd, time_step=time_step, duration=duration, initial_stage=gerd_level)
initial_flow = gerd_discharge_hyd.get_at(time=0)

xs_chainages, sections = load_trapzoid_xs(file_path=cross_sections_path, n_fp=n_fp, n_main=n_main)

roseires_ch = xs_chainages[-1]
roseires_bed = sections[-1].z_min

upstream_ch = xs_chainages[0]

upstream_bc = Boundary(condition='flow_hydrograph',
                        hydrograph=gerd_discharge_hyd if with_gerd else gerd_inflow_hyd,
                        chainage=upstream_ch)

Roseires = Boundary(initial_depth=initial_roseires_level-roseires_bed,
                    bed_level=roseires_bed,
                    condition='rating_curve',
                    #condition='fixed_depth',
                    rating_curve=RoseiresRatingCurve(initial_stage=initial_roseires_level, initial_flow=initial_flow,
                                                        jammed_sluice_gates=jammed_sluice_gates,
                                                        jammed_spillways=jammed_spillways),
                    chainage=roseires_ch)

GERD_Roseires_system = Channel(initial_flow=initial_flow,
                                upstream_boundary=upstream_bc,
                                downstream_boundary=Roseires)

if coords_path is not None:
    coords = import_table(coords_path, sort_by='chainage')
    GERD_Roseires_system.set_coords(coords=coords[:, 1:], chainages=coords[:, 0])
    
GERD_Roseires_system.set_cross_sections(chainages=xs_chainages, sections=sections)

solver = PreissmannSolver(channel=GERD_Roseires_system,
                            theta=theta,
                            time_step=time_step,
                            spatial_step=spatial_step,
                            simulation_time=duration)

if True:
    from .custom_functions import draw, export_banks
    from math import atan, pi
    
    tan = (637.154-806.190) / (6802.699-5520.016)
    theta = pi -0.2
    
    left_x, left_y, right_x, right_y = draw(chainages=GERD_Roseires_system.ch_at_node,
                                            widths=[GERD_Roseires_system.top_width(i, hw=GERD_Roseires_system.initial_conditions[i, 0] + GERD_Roseires_system.bed_level_at(i)) for i in range(len(GERD_Roseires_system.ch_at_node))],
                                            curvature=[-xs.curvature for xs in GERD_Roseires_system.xs_at_node],
                                            x0=726833,
                                            y0=1240801,
                                            theta0=theta)
    
    export_banks(left_x, left_y, right_x, right_y)