from settings import *
import numpy as np
from utility import Utility, RatingCurve


class Boundary:
    def __init__(self, initial_depth, initial_discharge, condition, initial_stage = None, fixed_depth = None, rating_curve: RatingCurve = None, hydrograph_function = None):
        self.initial_depth = initial_depth
        self.initial_discharge = initial_discharge
        
        self.initial_stage = initial_stage
        self.bed_level = None
        
        if initial_stage is not None:
            self.bed_level = initial_stage - initial_depth
        
        if condition in ['hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve']:
            self.condition = condition
        else:
            raise ValueError("Invalid boundary condition.")
        
        self.rating_curve = rating_curve
        self.hydrograph = None
        self.fixed_depth = fixed_depth
        
        self.reservoir_area = None
        self.reservoir_exit_rating_curve = None
        
        if hydrograph_function is not None:
            self.hydrograph_Q_func = hydrograph_function
        else:
            self.hydrograph_Q_func = self.interpolate_hydrograph
    
    
    def set_storage_behavior(self, reservoir_area: float, reservoir_exit_rating_curve: RatingCurve):
        self.reservoir_area = reservoir_area
        self.reservoir_exit_rating_curve = reservoir_exit_rating_curve
        
        
    def update_fixed_depth(self, inflow, duration, stage):
        if self.reservoir_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
        
        if inflow is None or duration is None or stage is None:
            raise ValueError("Insufficient arguments.")
        
        outflow = self.reservoir_exit_rating_curve.discharge(stage)
        
        if stage <= self.initial_stage and outflow >= inflow:
            return
        
        added_volume = (inflow - outflow) * duration
        added_depth = added_volume / self.reservoir_area
        self.fixed_depth += added_depth
    
    
    def rating_curve_Q(self, stage):
        if self.rating_curve is None:
            raise ValueError("Rating curve is undefined.")
        
        return self.rating_curve.discharge(stage)
    
    
    def rating_curve_stage(self, discharge):
        if self.rating_curve is None:
            raise ValueError("Rating curve is undefined.")
        
        return self.rating_curve.stage(discharge)
    
    
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


    def set_rating_curve(self, rating_curve: RatingCurve):
        self.rating_curve = rating_curve
        
    
    def condition_residual(self, time = None, depth = None, width = None, discharge = None, bed_slope = None, manning_co = None, delta_t = None):
        if self.condition == 'hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph_Q(time)
            return discharge - Q_t
            
        if self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.fixed_depth is None:
                raise ValueError("Fixed depth is not defined.")
            else:
                if self.reservoir_area is not None:
                    self.update_fixed_depth(discharge, delta_t, self.bed_level + depth)
                    
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
            
            return discharge - self.rating_curve_Q(self.bed_level + depth)
        
        
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
            
            return 0 - self.rating_curve.discharge_derivative(stage, width)
        
    
    def condition_derivative_wrt_Q(self):
        if self.condition == 'hydrograph':
            return 1
            
        if self.condition == 'fixed_depth':
            return 0
        
        if self.condition == 'normal_depth':
            return 1
        
        if self.condition == 'rating_curve':
            return 1
