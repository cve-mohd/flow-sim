from utility import RatingCurve


class Boundary:
    def __init__(self, initial_depth, condition, bed_level, rating_curve: RatingCurve = None,
                 flow_hydrograph_function = None, stage_hydrograph_function = None, chainage: int | float = 0):
        
        self.initial_depth = initial_depth
        
        self.bed_level = bed_level
        self.initial_stage = bed_level + initial_depth
        self.storage_stage = self.initial_stage
                
        if condition in ['flow_hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve', 'stage_hydrograph']:
            self.condition = condition
        else:
            raise ValueError("Invalid boundary condition.")
        
        self.rating_curve = rating_curve
        self.flow_hydrograph = None
        self.stage_hydrograph = None
        self.chainage = chainage
        
        self.storage_area = None
        self.storage_exit_rating_curve = None
        self.active_storage = False
        
        if flow_hydrograph_function is not None:
            self.flow_hydrograph_func = flow_hydrograph_function
        else:
            self.flow_hydrograph_func = self.interpolate_flow_hydrograph
            
        if stage_hydrograph_function is not None:
            self.stage_hydrograph_func = stage_hydrograph_function
        else:
            self.stage_hydrograph_func = self.interpolate_stage_hydrograph
    
    def set_storage_behavior(self, storage_area: float, storage_exit_rating_curve: RatingCurve):
        self.storage_area = storage_area
        self.storage_exit_rating_curve = storage_exit_rating_curve
        self.active_storage = True
    
    def interpolate_flow_hydrograph(self, t):
        if self.flow_hydrograph is None:
            raise ValueError("Flow hydrograph is not defined.")
        
        max_time, _ = self.flow_hydrograph[-1]
        if t > max_time:
            raise ValueError("Specified time is out of bound.")
                
        from numpy import interp
        times, flows = zip(*self.flow_hydrograph)
        
        return float(interp(t, times, flows))

    def get_flow_from_hydrograph(self, time):
        return self.flow_hydrograph_func(time)
        
    def build_flow_hydrograph(self, times, discharges):
        if len(times) != len(discharges):
            raise ValueError("Times and discharges must have the same length.")
        
        self.flow_hydrograph = list(zip(times, discharges))
            
    def set_flow_hydrograph(self, func):
        self.flow_hydrograph_func = func
        
    ###########################
    
    def interpolate_stage_hydrograph(self, time):
        if self.stage_hydrograph is None:
            raise ValueError("Stage hydrograph is not defined.")
        
        max_time, _ = self.stage_hydrograph[-1]
        if time > max_time:
            raise ValueError("Specified time is out of bound.")
                
        from numpy import interp
        times, stages = zip(*self.stage_hydrograph)
        
        return float(interp(time, times, stages))

    def get_stage_from_hydrograph(self, time):
        return self.stage_hydrograph_func(time)
    
    def build_stage_hydrograph(self, times, stages):
        if len(times) != len(stages):
            raise ValueError("Times and stages must have the same length.")
        
        self.stage_hydrograph = list(zip(times, stages))
        
    def set_stage_hydrograph(self, func):
        self.stage_hydrograph_func = func
        
    ########################

    def set_rating_curve(self, rating_curve: RatingCurve):
        self.rating_curve = rating_curve
        
    def condition_residual(self, time = None, depth = None, width = None, discharge = None, bed_slope = None, manning_co = None, time_step = None):
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.get_flow_from_hydrograph(time)
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
            
            Q = A ** (5./3) * bed_slope / (manning_co * P ** (2./3) * abs(bed_slope) ** 0.5)
            
            residual = discharge - Q
        
        elif self.condition == 'rating_curve':
            if depth is None or discharge is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            residual = discharge - self.rating_curve.discharge(self.bed_level + depth)
            
        if self.condition == 'stage_hydrograph':
            if time is None or depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage_t = self.get_stage_from_hydrograph(time)
            residual = self.bed_level + depth - stage_t
        
        return residual
        
    def condition_derivative_wrt_A(self, time = None, area = None, width = None, bed_slope = None, manning_co = None):
        dy_dA = 1. / width
        
        if self.condition == 'flow_hydrograph':
            derivative = 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dy_dA
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or manning_co is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            P = width + 2 * area/width
            R = area/P
            
            derivative = - bed_slope / (manning_co / abs(bed_slope) ** 0.5) * (
                -4. / (3*width) * R ** (5./3) + 5./3 * R ** (2./3)
                )
        
        elif self.condition == 'rating_curve':
            if width is None or area is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage = self.bed_level + area/width
            
            derivative = 0 - self.rating_curve.derivative_wrt_stage(stage) * dy_dA
            
        elif self.condition == 'stage_hydrograph':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dy_dA
        
        return derivative
        
    def condition_derivative_wrt_Q(self):
        if self.condition == 'flow_hydrograph':
            derivative = 1
            
        elif self.condition == 'fixed_depth':
            derivative = 0
        
        elif self.condition == 'normal_depth':
            derivative = 1
        
        elif self.condition == 'rating_curve':
            derivative = 1
        
        elif self.condition == 'stage_hydrograph':
            derivative = 0
                        
        return derivative

    def mass_balance(self, duration, inflow, stage) -> float:
        """
        Computes the new storage stage using a mass-balance equation.

        Parameters
        ----------
        inflow : float
            The rate of flow entering the storage.
        duration : float
            The time during which the inflow occurs. Normally, this is the models time step.

        """
        
        if self.storage_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
                
        outflow = self.storage_exit_rating_curve.discharge(stage)
        
        added_volume = (inflow - outflow) * duration
        added_depth = added_volume / float(self.storage_area)
                
        new_storage_stage = stage + added_depth
        
        if new_storage_stage < self.initial_stage:
            new_storage_stage = self.initial_stage
                
        return new_storage_stage
       
    def mass_balance_deriv_wrt_Q(self, duration, inflow, stage) -> float:
        if self.storage_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
                    
        derivative = duration / float(self.storage_area)
        
        if self.mass_balance(duration, inflow, stage) <= self.initial_stage:
            derivative = 0
        
        return derivative
    
    def mass_balance_deriv_wrt_stage(self, duration, inflow, stage) -> float:        
        if self.storage_exit_rating_curve is None:
            raise ValueError("Storage is undefined.")
        
        der_outflow = self.storage_exit_rating_curve.derivative(stage)
        
        derivative = 1 - der_outflow * duration / float(self.storage_area)
        
        if self.mass_balance(duration, inflow, stage) <= self.initial_stage:
            derivative = 0
        
        return derivative
    
    def condition_derivative_wrt_res_h(self):
        if self.condition == 'fixed_depth':
            derivative = -1
        else:
            derivative = 0
            
        return derivative
    
    def status(self):
        tostring = f'Initial depth = {self.initial_depth}\nCondition: '
        if self.condition == 'flow_hydrograph':
            tostring += 'Flow hydrograph\n'
            tostring += '\tValues:\n'
            for a, b in self.flow_hydrograph:
                tostring += '\t\t' + f'{a:.2f}, {b:.2f}\n'
            
        elif self.condition == 'fixed_depth':
            tostring += 'Fixed depth\n'
            tostring += '\tStorage: '
            if self.active_storage:
                tostring += 'active\n'
                tostring += f'\tStorage area = {self.storage_area}'
            else:
                tostring += 'inactive'
        
        elif self.condition == 'normal_depth':
            tostring += 'Normal depth'
        
        elif self.condition == 'rating_curve':
            tostring += 'Rating curve\n'
            tostring += '\tEquation: ' + self.rating_curve.tostring()
        
        elif self.condition == 'stage_hydrograph':
            tostring += 'Stage hydrograph\n'
            tostring += '\tValues:\n'
            for a, b in self.stage_hydrograph:
                tostring += '\t\t' + f'{a:.2f}, {b:.2f}\n'
            
                        
        return tostring
    