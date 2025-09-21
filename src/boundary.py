from src.utility import RatingCurve, Hydrograph, Hydraulics, LumpedStorage

class Boundary:
    def __init__(self,
                 condition: str,
                 bed_level: float,
                 chainage: int | float,
                 initial_depth: float = None,
                 rating_curve: RatingCurve = None,
                 hydrograph: Hydrograph = None):
        
        self.condition = condition
        self.bed_level = bed_level
        
        if self.condition not in ['flow_hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve', 'stage_hydrograph']:
            raise ValueError("Invalid boundary condition.")
                
        if initial_depth is None:
            self.initial_depth = self.initial_stage = None
        else:
            self.initial_depth = initial_depth
            self.initial_stage = bed_level + initial_depth
        
        self.chainage = chainage
                        
        self.rating_curve = rating_curve
        self.hydrograph = hydrograph
        
        self.lumped_storage = None
    
    def set_lumped_storage(self, lumped_storage: LumpedStorage):
        self.lumped_storage = lumped_storage
        self.lumped_storage.stage = self.initial_stage
    
    def condition_residual(self, time = None, depth = None, width = None, flow_rate = None, bed_slope = None, roughness = None):
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph.get_at(time=time)
            residual = flow_rate - Q_t
            
        elif self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.initial_depth is None:
                raise ValueError("Fixed depth is not defined.")
            else:
                if self.lumped_storage is not None:
                    residual = depth - (self.lumped_storage.stage - self.bed_level)
                else:
                    residual = depth - self.initial_depth
        
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
        dh_dA = 1/width
        
        if self.condition == 'flow_hydrograph':
            derivative = 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dh_dA
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = 0 - Hydraulics.dQn_dA(A=area, S=bed_slope, n=roughness, B=width)
        
        elif self.condition == 'rating_curve':
            if width is None or area is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage = self.bed_level + area/width
            
            derivative = 0 - self.rating_curve.derivative_wrt_stage(stage) * dh_dA
            
        elif self.condition == 'stage_hydrograph':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            derivative = dh_dA
        
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
    
    def df_dYN(self):
        if self.condition == 'fixed_depth':
            derivative = -1
        else:
            derivative = 0
            
        return derivative
    