from src.hydromodel.channel import Channel
from src.hydromodel.boundary import Boundary
from src.hydromodel.hydrograph import Hydrograph
from src.hydromodel.lumped_storage import LumpedStorage
from src.hydromodel.preissmann import PreissmannSolver
from ..settings import initial_roseires_level, wet_n, dry_n, used_roseires_rc, constant_flow
from ..settings import theta, spatial_step, time_step, sim_duration, tolerance
from ..custom_functions import import_table, import_hydrograph, import_area_curve
from ..roseires_rating_curve import RoseiresRatingCurve

input_dir = "cases\\gerd_roseires\\lumped\\data\\"
inflow_hyd = Hydrograph(table=import_hydrograph("cases\\gerd_roseires\\data\\inflow_hydrograph.csv"))
bed_profile = import_table(input_dir + "bed_profile.csv", sort_by='chainage')
widths = import_table(input_dir + "width.csv", sort_by='chainage')
coords = import_table(input_dir + "centerline_coords.csv", sort_by='chainage')

gerd_bed_level = bed_profile[0, 1]
gerd_chainage = min([bed_profile[:, 0].min(), widths[:, 0].min(), coords[:, 0].min()])

ds_bed_level = bed_profile[-1, 1]
ds_chainage = max([bed_profile[:, 0].max(), widths[:, 0].max(), coords[:, 0].max()])
ds_depth = initial_roseires_level - ds_bed_level
total_system_length = 122000
    
def run(
    rep_length_fraction = 0.5,
    wet_n = wet_n,
    dry_n = dry_n,
    n_steepness = 0.15,
    Cc = 0.5,
    K_q = 0,
    verbose = False,
    save_results = False
):      

    roseires_storage = LumpedStorage(min_stage=initial_roseires_level,
                                    #rating_curve=RoseiresRatingCurve(),
                                     rating_curve=used_roseires_rc,
                                     solution_boundaries=None)
    
    roseires_storage.set_area_curve(table=import_area_curve(input_dir + 'roseires_storage_curve.csv'))
    roseires_storage.reservoir_length = rep_length_fraction*(total_system_length - ds_chainage)
    roseires_storage.capture_losses = True
    roseires_storage.Cc = Cc
    roseires_storage.K_q = K_q

    ds_depth_ = recalc_ds_h(width=widths[-1, 1], ds_depth=ds_depth, wet_n=wet_n, roseires_storage=roseires_storage)
        
    GERD = Boundary(condition='flow_hydrograph',
                    bed_level=gerd_bed_level,
                    chainage=gerd_chainage,
                    hydrograph=inflow_hyd)

    Roseires = Boundary(initial_depth=ds_depth_,
                        condition='fixed_depth',
                        bed_level=ds_bed_level,
                        chainage=ds_chainage)

    Roseires.set_lumped_storage(roseires_storage)

    GERD_Roseires_system = Channel(width=widths[0, 1],
                                   initial_flow=inflow_hyd.get_at(0),
                                   roughness=wet_n,
                                   dry_roughness=dry_n,
                                   upstream_boundary=GERD,
                                   downstream_boundary=Roseires,
                                   n_steepness=n_steepness)

    # GERD_Roseires_system.set_coords(coords=coords[:, 1:], chainages=coords[:, 0])
    # INSERT CROSS-SECTIONS HERE

    solver = PreissmannSolver(channel=GERD_Roseires_system,
                              theta=theta,
                              time_step=time_step,
                              spatial_step=spatial_step,
                              simulation_time=sim_duration)

    if verbose:
        print("Simulation started.")

    solver.run(verbose=0, tolerance=tolerance)
    
    
    if save_results:
        if verbose:
            print("Saving results...")
            
        solver.save_results(folder_path='cases\\gerd_roseires\\lumped\\results\\')
        
        if verbose:
            print("Done.")

    from numpy import max as max_
    ds_peak_amp = max_(solver.storage_stage) - initial_roseires_level
    attenuation = 1 - max_(solver.flow[:, -1]) / max_(solver.flow[:, 0])
    us_peak_amp = solver.peak_amplitude[0]
    
    return attenuation, us_peak_amp, ds_peak_amp

def recalc_ds_h(width, ds_depth, wet_n, roseires_storage: LumpedStorage):
    A = width * ds_depth
    P = width + 2 * ds_depth
    R = A/P
    while True:
        new_ds_depth = initial_roseires_level - ds_bed_level + roseires_storage.energy_loss(
            A_ent=A,
            Q=inflow_hyd.get_at(0),
            n=wet_n,
            R=R
        )
        
        diff = ds_depth - new_ds_depth
        ds_depth = new_ds_depth
        
        A = width * ds_depth
        P = width + 2 * ds_depth
        R = A/P
        
        if abs(diff) < 1e-6:
            break
        
    return ds_depth

if __name__ == '__main__':
    run(verbose=True, save_results=True)
    
# Run command: py -m cases.gerd_roseires.lumped.main