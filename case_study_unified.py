from reach import Reach
from boundary import Boundary
from utility import Hydrograph
import pandas as pd
from case_study_settings import used_roseires_rc, storage_area

def import_geometry(path):
    # Skip first two rows: header + units
    df = pd.read_excel(path, skiprows=2)

    # Extract columns
    chainages = df.iloc[:, 1].astype(float).tolist()
    widths    = df.iloc[:, 2].astype(float).tolist()
    levels    = df.iloc[:, 3]  # may contain NaNs

    # Width data: all rows
    width_ch  = chainages
    widths    = widths

    # Level data: drop NaNs
    level_ch  = df.iloc[:, 1][~levels.isna()].astype(float).tolist()
    levels    = levels.dropna().astype(float).tolist()

    return widths, levels, width_ch, level_ch

# Example usage
# widths, levels, width_ch, level_ch = import_geometry("geometry.xlsx")


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
                             channel_roughness = 0.029,
                             upstream_boundary = GERD,
                             downstream_boundary = Roseires)

widths, levels, width_ch, level_ch = import_geometry("geometry.xlsx")

GERD_Roseires_system.set_intermediate_bed_levels(bed_levels=levels, chainages=level_ch)
GERD_Roseires_system.set_intermediate_widths(widths=widths, chainages=width_ch)

from preissmann import PreissmannSolver

solver = PreissmannSolver(reach=GERD_Roseires_system,
                          theta=0.8,
                          time_step=3600,
                          spatial_step=1000,
                          enforce_physicality=False)

solver.run(duration=3600*96, verbose=0)
solver.save_results()

print('Success.')
