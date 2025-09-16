from src.utility import RatingCurve, Hydrograph, Hydraulics


class Boundary:
    def __init__(self,
                 condition: str,
                 bed_level: float,
                 chainage: int | float,
                 initial_depth: float,
                 rating_curve: RatingCurve = None,
                 hydrograph: Hydrograph = None):
        
        self.initial_depth = initial_depth
        
        self.bed_level = bed_level
        self.initial_stage = bed_level + initial_depth
        self.storage_stage = self.initial_stage
        self.chainage = chainage
                
        if condition in ['flow_hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve', 'stage_hydrograph']:
            self.condition = condition
        else:
            raise ValueError("Invalid boundary condition.")
        
        self.rating_curve = rating_curve
        self.hydrograph = hydrograph
        
        self.storage_area = None
        self.storage_exit_rating_curve = None
        self.active_storage = False
    
    def set_storage(self, storage_area: float, storage_exit_rating_curve: RatingCurve):
        self.storage_area = storage_area
        self.storage_exit_rating_curve = storage_exit_rating_curve
        self.active_storage = True
    
    def set_rating_curve(self, rating_curve: RatingCurve):
        self.rating_curve = rating_curve
        
    def condition_residual(self, time = None, depth = None, width = None, flow_rate = None, bed_slope = None, roughness = None):
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph.get_at(time=time)
            residual = flow_rate - Q_t
            
        elif self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.storage_stage is None:
                raise ValueError("Fixed depth is not defined.")
            else:                    
                residual = depth - (self.storage_stage - self.bed_level)
        
        elif self.condition == 'normal_depth':
            if width is None or depth is None or flow_rate is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            normal_flow = Hydraulics.normal_flow(A=width*depth, S_0=bed_slope, n=roughness, B=width)
            #print(f'Normal = {normal_flow}, current = {flow_rate}')
            residual = flow_rate - normal_flow
        
        elif self.condition == 'rating_curve':
            if depth is None or flow_rate is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            residual = flow_rate - self.rating_curve.discharge(self.bed_level + depth)
            
        if self.condition == 'stage_hydrograph':
            if time is None or depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage_t = self.hydrograph.get_at(time=time)
            residual = self.bed_level + depth - stage_t
        
        return residual
        
    def df_dA(self, area = None, width = None, bed_slope = None, roughness = None):
        dy_dA = 1. / width
        
        if self.condition == 'flow_hydrograph':
            derivative = 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dy_dA
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = 0 - Hydraulics.dQn_dA(A=area, S=bed_slope, n=roughness, B=width)
        
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
        
    def df_dQ(self):
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
    
    def df_dn(self, depth, width, bed_slope, roughness):
        if self.condition == 'normal_depth':
            df_dn = 0 - Hydraulics.dQn_dn(A=width*depth, S_0=bed_slope, n=roughness, B=width)
        else:
            df_dn = 0
        
        return df_dn

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
            
        return tostring
    