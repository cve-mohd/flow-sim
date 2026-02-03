from src.hydromodel.hydrograph import Hydrograph
from scipy.optimize import brentq
from pandas import read_csv
import numpy as np

class GerdHydrograph(Hydrograph):
    def __init__(self):
        super().__init__(function=None, table=None)
        
        self.turbine_flow = 1562.5
        
    def build(self, inflow_hydrograph, time_step, duration, initial_stage):
        self.table = np.empty(shape=(duration//time_step+1, 2), dtype=np.float64)
                
        table = read_csv('cases\\gerd_roseires\\data\\gerd_vol_curve.csv', header=None).to_numpy(dtype=np.float64)
        vols, stages = table[:, 0], table[:, 1]
        
        stage_0 = initial_stage
        inflow_0 = inflow_hydrograph.get_at(0)
        outflow_0 = self.release(inflow=inflow_0, stage=stage_0, initial_stage=initial_stage)
        
        self.table[0, 0] = 0
        self.table[0, 1] = outflow_0

        for t in range(time_step, duration + time_step, time_step):

            # Inflow
            inflow_1 = inflow_hydrograph.get_at(t)
            avg_inflow = 0.5 * (inflow_1 + inflow_0)
            vol_0 = np.interp(x=stage_0, xp=stages, fp=vols)
            
            # Outflow
            Q_req = inflow_1
            
            def mass_balance(stage_1):
                outflow_1 = self.release(Q_req, stage_1, initial_stage)
                avg_outflow = 0.5 * (outflow_1 + outflow_0)
                
                vol_1 = np.interp(x=stage_1, xp=stages, fp=vols)
                
                dVol = vol_1 - vol_0
                
                return dVol - (avg_inflow - avg_outflow) * time_step * 1e-6
                
            stage_1 = brentq(f=mass_balance, a=624.9, b=645)
            outflow_1 = self.release(Q_req, stage_1, initial_stage)

            # Store new values
            time_level = t // time_step
            self.table[time_level, 0] = t
            self.table[time_level, 1] = outflow_1

            # Update old values
            stage_0 = stage_1
            inflow_0 = inflow_1
            outflow_0 = outflow_1

    def release(self, inflow, stage, initial_stage):
        capacity = self.effective_capacity(WL=stage)
        
        if stage > initial_stage:
            discharge = capacity
            
        else:
            discharge = min(inflow, capacity)
            discharge = max(discharge, self.turbine_flow)
        
        return discharge
            
    def effective_capacity(self, WL, use_bottom_outlets = False):
        # 1. Discharge of Spillways

        # 1.1 Gate-controlled ogee crest spillways:
        Q1 = self.gated_spillway(WL) * self.alpha(WL)
        #print(f"Q1 = 196.4017 * (640 - 624.9)^(3/2) = {Q1} m^3/s\n")

        # 1.2 Stepped spillway:
        Q2 = self.stepped_spillway(WL)
        #print(f"Q2 = 447.3594 * (640 - 640.0)^(3/2) = {Q2} m^3/s\n")

        # 1.3 Emergency spillway:
        Q3 = self.emergency_spillway(WL)
        #print(f"Q3 = 654.6723 * (640 - 642.0)^(3/2) = {Q3} m^3/s\n")

        # 2. Discharge of Bottom Outlets
        Q4 = self.bottom_outlets(WL) if use_bottom_outlets else 0
        #print(f"640 - 545 = 9.9125 * 1e-5 Q4^2 + 1.00295 * 1e-3 * {darcey_f} Q4^2")
        #print(f"{640 - 545} = {9.9125 * 1e-5 + 1.00295 * 1e-3 * darcey_f} Q4^2")
        #print(f"Q4 = {Q4} m^3/s")
        
        # 3. Discharge of Turbines
        Q5 = self.turbine_flow

        return Q1 + Q2 + Q3 + Q4 + Q5
    
    def alpha(self, WL):
        spillway_crest = 624.9
        max_operating_level = 640
        
        if WL <= spillway_crest:
            return 0
        elif WL >= max_operating_level:
            return 1
        else:
            return (WL - spillway_crest) / (max_operating_level - spillway_crest)

    def bottom_outlets(self, WL, darcey_f = 0.01): # f range: [0.009, 0.013]
        def f(Q):
            return max(0, WL - 545) - (9.9125 * 1e-5 * Q**2 + 1.00295 * 1e-3 * darcey_f * Q**2)
        
        return brentq(f, a=0, b=1060)

    def emergency_spillway(self, WL):
        Q3 = 654.6723 * max(0, WL - 642.0)**(3/2)
        return Q3

    def stepped_spillway(self, WL):
        Q2 = 447.3594 * max(0, WL - 640.0)**(3/2)
        return Q2

    def gated_spillway(self, WL):
        Q1 = 196.4017 * max(0, WL - 624.9)**(3/2)
        return Q1

if __name__ == '__main__':
    import csv
    def ff(t):
        return 1500
    
    dx = 3600
    T = 3600 * 24 * 4
    g_hyd = GerdHydrograph()
    from .custom_functions import import_hydrograph
    g_hyd.build(
        inflow_hydrograph=Hydrograph(table=import_hydrograph("cases\\gerd_roseires\\data\\inflow_hydrograph.csv")),
        time_step=dx,
        duration=T,
        initial_stage=637
    )
    print(g_hyd.table)
    
    
    raise
            
    with open("gerd_upstream_rating_curve.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Stage", "Discharge"])  # header
                
        for s in range(655, 541, -1):
            writer.writerow([s, g_hyd.outflow_at(s)])
            
    with open("gerd_discharge_hydrograph.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Discharge"])  # header
                
        times = g_hyd.table[:, 0]
        discharge = g_hyd.table[:, 1]
        
        for i in range(T//dx):
            writer.writerow([g_hyd.table[i, 0], g_hyd.table[i, 1]])
            