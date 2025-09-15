from reach import Reach
from boundary import Boundary
from utility import Hydrograph
import pandas as pd
from case_study_settings import used_roseires_rc, storage_area
import pandas as pd
from settings import trapzoid_hydrograph

def import_geometry(path):
    df = pd.read_excel(path, skiprows=1)

    chainages    = df.iloc[:, 1]
    widths       = df.iloc[:, 2]
    levels       = df.iloc[:, 3]
    eastings     = df.iloc[:, 4]
    northings    = df.iloc[:, 5]

    width_df = pd.DataFrame({
        "chainage": chainages,
        "width": widths
    }).dropna()
    width_df = width_df.astype(float).sort_values(by="chainage")

    level_df = pd.DataFrame({
        "chainage": chainages,
        "level": levels
    }).dropna()
    level_df = level_df.astype(float).sort_values(by="chainage")
    
    coords_df = pd.DataFrame({
        "chainage": chainages,
        "eastings": eastings,
        "northings": northings
    }).dropna()
    coords_df = coords_df.astype(float).sort_values(by="chainage")

    width_ch     = width_df["chainage"].tolist()
    widths       = width_df["width"].tolist()
    level_ch     = level_df["chainage"].tolist()
    levels       = level_df["level"].tolist()
    coords_ch    = coords_df["chainage"].tolist()
    coords       = zip(coords_df["eastings"].tolist(), coords_df["northings"].tolist())

    return widths, width_ch, levels, level_ch, coords, coords_ch

hyd_const = Hydrograph(function=trapzoid_hydrograph)

hyd = Hydrograph()
hyd.load_csv('hydrograph.csv')

GERD = Boundary(initial_depth=0,
                condition='flow_hydrograph',
                bed_level=495,
                chainage=0,
                hydrograph=hyd)

Roseires = Boundary(initial_depth=490-470.54,
                    condition='fixed_depth',
                    bed_level=470.54,
                    chainage=120000)

Roseires.set_storage(storage_area, used_roseires_rc)

GERD_Roseires_system = Reach(width = 250,
                             initial_flow_rate = 1562.5,
                             roughness = 0.027,
                             dry_roughness=0.030,
                             upstream_boundary = GERD,
                             downstream_boundary = Roseires)

widths, width_ch, levels, level_ch, coords, coords_ch = import_geometry("geometry.xlsx")

GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=levels, chainages=level_ch)
GERD_Roseires_system.set_intermediate_widths(widths=widths, chainages=width_ch)
#GERD_Roseires_system.set_coords(coords=coords, chainages=coords_ch)

from preissmann import PreissmannSolver

solver = PreissmannSolver(reach=GERD_Roseires_system,
                          theta=0.8,
                          time_step=3600,
                          spatial_step=1000,
                          enforce_physicality=False)

GERD_Roseires_system.downstream_boundary.storage_area = 440e6 - GERD_Roseires_system.surface_area

solver.run(duration=3600*24, verbose=0)
solver.save_results()

print('Success.')
