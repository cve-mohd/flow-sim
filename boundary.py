from utility import RatingCurve


class Boundary:
    def __init__(self, initial_depth, condition, bed_level, rating_curve: RatingCurve = None,
                 hydrograph_function = None, chainage: int | float = 0):
        
        self.initial_depth = initial_depth
        
        self.bed_level = bed_level
        self.initial_stage = bed_level + initial_depth
        self.storage_stage = self.initial_stage
                
        if condition in ['hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve']:
            self.condition = condition
        else:
            raise ValueError("Invalid boundary condition.")
        
        self.rating_curve = rating_curve
        self.hydrograph = None
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
        
    
    def condition_residual(self, time = None, depth = None, width = None, discharge = None, bed_slope = None, manning_co = None, time_step = None):
        if self.condition == 'hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph_Q(time)
            residual = discharge - Q_t
            
        elif self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.storage_stage is None:
                raise ValueError("Fixed depth is not defined.")
            else:                    
                residual = depth - (self.storage_stage - self.bed_level)
        
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
            
            residual = discharge - self.rating_curve.discharge(self.bed_level + depth)
        
        return residual
        
        
    def condition_derivative_wrt_A(self, time = None, area = None, width = None, bed_slope = None, manning_co = None):
        dy_dA = 1. / width
        
        if self.condition == 'hydrograph':
            derivative = 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dy_dA * (1 - 0)
        
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


    def mass_balance(self, inflow, duration) -> float:
        """
        Computes the new storage stage using a mass-balance equation.

        Parameters
        ----------
        inflow : float
            The rate of flow entering the storage.
        duration : float
            The time during which the inflow occurs. Normally, this is the models time step.

        """
        
        if self.reservoir_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
                
        outflow = self.reservoir_exit_rating_curve.discharge(self.storage_stage)
        
        added_volume = (inflow - outflow) * duration
        added_depth = added_volume / float(self.reservoir_area)
                
        new_storage_stage = self.storage_stage + added_depth
        
        if new_storage_stage < self.initial_stage:
            new_storage_stage = self.initial_stage    
        
        return new_storage_stage
       
    def mass_balance_deriv_Q(self, inflow, duration) -> float:
        if self.reservoir_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
        
        derivative = duration / float(self.reservoir_area)
        
        if self.mass_balance(inflow, duration) <= self.initial_stage:
            derivative = 0
        
        return derivative
    
    def mass_balance_deriv_res_h(self, inflow, duration) -> float:        
        if self.reservoir_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
        
        der_outflow = self.reservoir_exit_rating_curve.derivative(self.storage_stage)
        
        derivative = 1 - der_outflow * duration / float(self.reservoir_area)
        
        if self.mass_balance(inflow, duration) <= self.initial_stage:
            derivative = 0
        
        return derivative
    
    def condition_derivative_wrt_res_h(self):
        if self.condition == 'fixed_depth':
            derivative = -1
        else:
            derivative = 0
            
        return derivative