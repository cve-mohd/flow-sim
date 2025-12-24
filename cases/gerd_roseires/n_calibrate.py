import numpy as np
from scipy.optimize import minimize
from .model import run

def run_model(n_main, Q):
    Y_values = run(
        n_main=n_main,
        Q=Q,
        verbose=0,
        folder=None,
        inflow_hyd_path="cases\\gerd_roseires\\data\\inflow_hydrograph_small.csv",
        coords_path=None,
        inflow_hyd_func=None,
        sim_duration=None
    )
    
    return np.array(Y_values)

def objective(vars, Q, H_target):
    n_main = vars[0]
    H_sim = run_model(n_main, Q)

    return np.sum((H_sim - H_target)**2)

bounds = [(0.020, 0.050)]

times = np.array([1, 5, 9, 13, 17, 21])  # hours
H_target = np.array([497.5, 500, 502, 505, 507, 510])  # m
Q = np.array([1562.5, 3850, 6000, 10000, 14000, 21000])

initial_guess = [0.028]

"""result = minimize(
    fun=objective,
    x0=initial_guess,
    args=(Q, H_target),
    bounds=bounds,
    method="L-BFGS-B"
)

if result.success:
    n_main_opt = result.x[0]
    print("Optimized n_main:       ", n_main_opt)

    # Evaluate fit quality
    H_sim = run_model(n_main_opt, Q)
    print("\nSimulated water levels:", H_sim)
    print("Target water levels:   ", H_target)
    print("RMSE:", np.sqrt(np.mean((H_sim - H_target)**2)))

else:
    print("Optimization failed:", result.message)"""


def calc_rmse_curve(run_model, Y_target, Q, n_values):
    RMSE = []

    for n in n_values:
        Y_sim = run_model(n_main=n, Q=Q)
        rmse = np.mean( (Y_sim - Y_target)**2 ) ** 0.5
        RMSE.append(float(rmse))
        
    return RMSE


n_values = np.linspace(0.020, 0.060, 10).tolist()
RMSE = calc_rmse_curve(run_model, H_target, Q, n_values)

import csv
with open("calibration_rmse_curve.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["n", "RMSE"])  # header
    for n_val, rmse_val in zip(n_values, RMSE):
        writer.writerow([n_val, rmse_val])


# py -m cases.gerd_roseires.n_calibrate