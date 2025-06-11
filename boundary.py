from utility import RatingCurve


class Boundary:
    def __init__(self, initial_depth, condition, bed_level,
                 fixed_depth = None, rating_curve: RatingCurve = None, hydrograph_function = None, chainage: int | float = 0):
        
        self.initial_depth = initial_depth
        
        self.bed_level = bed_level
        self.initial_stage = bed_level + initial_depth
                
        if condition in ['hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve']:
            self.condition = condition
        else:
            raise ValueError("Invalid boundary condition.")
        
        self.rating_curve = rating_curve
        self.hydrograph = None
        self.fixed_depth = fixed_depth
        self.chainage = chainage
        
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
        
        from numpy import interp
        times, flows = zip(*self.hydrograph)
        return float(interp(t, times, flows))


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
            residual = discharge - Q_t
            
        elif self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.fixed_depth is None:
                raise ValueError("Fixed depth is not defined.")
            else:
                if self.reservoir_area is not None:
                    self.update_fixed_depth(discharge, delta_t, self.bed_level + depth)
                    
                residual = depth - self.fixed_depth
        
        elif self.condition == 'normal_depth':
            if width is None or depth is None or discharge is None or bed_slope is None or manning_co is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            P = width + 2 * depth
            A = width * depth
            
            Q = (A ** (5./3) * bed_slope ** 0.5) / (manning_co * P ** (2./3))
            residual = discharge - Q
        
        elif self.condition == 'rating_curve':
            if depth is None or discharge is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            residual = discharge - self.rating_curve_Q(self.bed_level + depth)
        
        return residual
        
        
    def condition_derivative_wrt_A(self, time = None, area = None, width = None, bed_slope = None, manning_co = None):
        dy_dA = 1. / width
        
        if self.condition == 'hydrograph':
            derivative = 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dy_dA - 0
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or manning_co is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            P = width + 2 * area/width
            R = area/P
            
            derivative = - bed_slope ** 0.5 / manning_co * (
                -4. / (3*width) * R ** (5./3) + 5./3 * R ** (2./3)
                )
        
        elif self.condition == 'rating_curve':
            if width is None or area is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage = self.bed_level + area/width
            
            derivative = 0 - self.rating_curve.derivative_wrt_stage(stage) * dy_dA
        
        return derivative
        
    
    def condition_derivative_wrt_Q(self):
        if self.condition == 'hydrograph':
            derivative = 1
            
        if self.condition == 'fixed_depth':
            derivative = 0
        
        if self.condition == 'normal_depth':
            derivative = 1
        
        if self.condition == 'rating_curve':
            derivative = 1
            
        return derivative
