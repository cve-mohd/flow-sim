import numpy as np
from scipy.optimize import minimize
from .model import run

def run_model(n_main, Q):
    print(f'Main n = {n_main}')
    
    Y_values = run(
        n_main=n_main,
        Q=Q,
        verbose=0,
        save_path=None,
        inflow_hyd_path="cases\\gerd_roseires\\data\\inflow_hydrograph_small.csv"
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

result = minimize(
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
    print("Optimization failed:", result.message)

# py -m cases.gerd_roseires.n_calibrate