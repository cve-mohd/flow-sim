from settings import *
import numpy as np
from utility import Utility


class Boundary:
    def __init__(self, initial_depth, initial_discharge, condition, initial_stage = None, rating_curve_eq = None):
        self.initial_depth = initial_depth
        self.initial_discharge = initial_discharge
        
        self.initial_stage = initial_stage
        self.bed_level = None
        
        if initial_stage is not None:
            self.bed_level = initial_stage - initial_depth
        
        self.condition = condition
        
        self.rating_curve_eq = rating_curve_eq
        self.hydrograph = None
        self.hydrograph_Q_func = self.interpolate_hydrograph
    
    
    def rating_curve(self, depth):
        stage = self.bed_level + depth
        q = Utility.rating_curve(stage, self.rating_curve_eq)
        return q
    
    
    def inverse_rating_curve(self, discharge):
        stage_guess = self.initial_stage
        y = Utility.inverse_rating_curve(discharge, self.rating_curve_eq, stage_guess)
        return y
    
    
    def interpolate_hydrograph(self, t):
        if self.hydrograph is None:
            raise ValueError("Hydrograph is not defined.")
        times, flows = zip(*self.hydrograph)
        return float(np.interp(t, times, flows))


    def hydrograph_Q(self, time):
        return self.hydrograph_Q_func(time)
    
    
    def build_hydrograph(self, times, discharges):
        if len(times) != len(discharges):
            raise ValueError("times and discharges must have the same length.")
        
        self.hydrograph = list(zip(times, discharges))
        
        
    def set_hydrograph(self, func):
        self.hydrograph_Q_func = func


    def fit_rating_curve(self, Q_values, Y_values):
        if len(Q_values) < 3:
            raise ValueError("Need at least 3 points.")
        
        if len(Q_values) != len(Y_values):
            raise ValueError("Q and Y lists should have the same lengths.")
        
        Y0 = np.min(Y_values) * 0.9

        Y_shifted = Y_values - Y0

        a, b, c = np.polyfit(Y_shifted, Q_values, deg=2)

        self.rating_curve_eq = {
            "base": Y0,
            "coefficients": [float(c), float(b), float(a)]
        }
        
    
    def condition_residual(self, time = None, depth = None, width = None, discharge = None, bed_slope = None, manning_co = None):
        if self.condition == 'hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph_Q(time)
            return discharge - Q_t
            
        if self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            else:
                return depth - self.fixed_depth
        
        if self.condition == 'normal_depth':
            if width is None or depth is None or discharge is None or bed_slope is None or manning_co is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            P = width + 2 * depth
            A = width * depth
            
            Q = (A ** (5./3) * bed_slope ** 0.5) / (manning_co * P ** (2./3))
            return discharge - Q
        
        if DS_CONDITION == 'rating_curve':
            if depth is None or discharge is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return discharge - self.rating_curve(depth)
        
        
    def condition_derivative_wrt_A(self, time = None, area = None, width = None, bed_slope = None, manning_co = None):
        if self.condition == 'hydrograph':
            return 0
            
        if self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 1./width
        
        if self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or manning_co is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            P = width + 2 * area/width
            R = area/P
            
            return - bed_slope ** 0.5 / manning_co * (
                -4. / (3*width) * R ** (5./3) + 5./3 * R ** (2./3)
                )
        
        if self.condition == 'rating_curve':
            if width is None or area is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage = self.bed_level + area/width
            
            return -(self.rating_curve_eq["coefficients"][1] / width
            + self.rating_curve_eq["coefficients"][2] * 2 * (stage - self.rating_curve_eq["base"]) / width)
        
    
    def condition_derivative_wrt_Q(self):
        if self.condition == 'hydrograph':
            return 1
            
        if self.condition == 'fixed_depth':
            return 0
        
        if self.condition == 'normal_depth':
            return 1
        
        if self.condition == 'rating_curve':
            return 1
