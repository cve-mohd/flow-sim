# Calculating Darcy-Weisbach friction factor f for assumed Q values and several roughnesses.
import math
import pandas as pd

def swamee_jain(Re, eps, D):
    return 0.25 / (math.log10(eps/(3.7*D) + 5.74/(Re**0.9))**2)

def colebrook_iterative(Re, eps, D, tol=1e-8, max_iter=200):
    if Re < 2000:
        return 64.0 / Re
    # initial guess: Swamee-Jain
    f = swamee_jain(Re, eps, D)
    for i in range(max_iter):
        # Colebrook residual solved for f_new using fixed-point rearrangement
        # Use Newton-like update via implicit formula:
        lhs = 1.0 / math.sqrt(f)
        rhs = -2.0 * math.log10(eps/(3.7*D) + 2.51/(Re * math.sqrt(f)))
        f_new = 1.0 / (rhs**2)
        if abs(f_new - f) < tol:
            return f_new
        f = f_new
    return f  # return last iterate if not converged

# Parameters
D = 6.0  # m diameter
nu = 1.003e-6  # m2/s at ~20 C (kinematic viscosity)
eps_values = [1e-4, 3e-4, 1e-3]  # m: smooth to rough concrete
Q_list = [50, 200, 500, 1000, 3000, 5000]  # m3/s total through twin barrels

rows = []
for Q in Q_list:
    Qb = Q / 2.0
    A = math.pi * D**2 / 4.0
    V = Qb / A
    Re = V * D / nu
    for eps in eps_values:
        f_sj = swamee_jain(Re, eps, D)
        f_cb = colebrook_iterative(Re, eps, D)
        rows.append({
            'Q_total_m3s': Q,
            'eps_m': eps,
            'V_m_s (per barrel)': round(V,4),
            'Re': int(Re),
            'f_Swamee-Jain': round(f_sj,6),
            'f_Colebrook': round(f_cb,6)
        })

df = pd.DataFrame(rows)
print(df)