width = 120
length = 2000
roughness = 0.023
tolerance = 1e-4
theta = 0.5
time_step = 3600
spatial_step = 1000
S_0 = 0.00061
duration = 20 * 3600

############                Hydrologic Geometry                 ############

initial_flow = 100

############                Inflow Hydrograph Functions         ############

def hydrograph(t):
    from math import sin, cos, pi
    t_b = 15 * 3600
    t_p = 5 * 3600
    Q_p = 200
    Q_b = initial_flow
    
    if t <= t_p:
        return Q_p / 2 * sin(pi*t/t_p - pi/2) + Q_p/2 + Q_b
    elif t_p < t and t <= t_b:
        return Q_p / 2 * cos(pi*(t-t_p)/(t_b-t_p)) + Q_p/2 + Q_b
    else:
        return Q_b
    