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
        self.chainage = chainage
        self.rating_curve = rating_curve
        self.hydrograph = hydrograph
        
        if self.condition not in ['flow_hydrograph', 'fixed_depth', 'normal_depth', 'rating_curve', 'stage_hydrograph']:
            raise ValueError("Invalid boundary condition.")
                
        if initial_depth is None:
            self.initial_depth = self.initial_stage = None
        else:
            self.initial_depth = initial_depth
            self.initial_stage = bed_level + initial_depth
        
        self.lumped_storage = None
    
    def set_lumped_storage(self, lumped_storage: LumpedStorage):
        self.lumped_storage = lumped_storage
        self.lumped_storage.stage = self.initial_stage
    
    def condition_residual(self, time = None, h = None, B = None, Q = None, S0 = None, n = None, nondimensionalization = False):
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Qt = self.hydrograph.get_at(time=time)
            return Q / Qt - 1 if nondimensionalization else Q - Qt
            
        elif self.condition == 'fixed_depth':
            if h is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.initial_depth is None:
                raise ValueError("Fixed depth is not defined.")
            else:
                if self.lumped_storage is not None:
                    hs = self.lumped_storage.stage - self.bed_level
                    return h / hs - 1 if nondimensionalization else h - hs
                else:
                    return h / self.initial_depth - 1 if nondimensionalization else h - self.initial_depth
        
        elif self.condition == 'normal_depth':
            if B is None or h is None or Q is None or S0 is None or n is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Qn = Hydraulics.normal_flow(A=B*h, S_0=S0, n=n, B=B)
            return Q / Qn - 1 if nondimensionalization else Q - Qn
        
        elif self.condition == 'rating_curve':
            if h is None or Q is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Qs = self.rating_curve.discharge(self.bed_level + h)
            return Q / Qs - 1 if nondimensionalization else Q - Qs
            
        if self.condition == 'stage_hydrograph':
            if time is None or h is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            ht = self.hydrograph.get_at(time=time) - self.bed_level
            return h / ht - 1 if nondimensionalization else h - ht
        
    def df_dA(self, time = None, A = None, Q = None, B = None, S0 = None, n = None, nondimensionalization = False):
        dh_dA = 1.0 / B
        
        if self.condition == 'flow_hydrograph':
            return 0
            
        elif self.condition == 'fixed_depth':
            if self.lumped_storage is not None:
                return dh_dA / (self.lumped_storage.stage - self.bed_level) if nondimensionalization else dh_dA
            else:
                if self.initial_depth is None:
                    raise ValueError("Fixed depth is not defined.")
                else:
                    return dh_dA / self.initial_depth if nondimensionalization else dh_dA
        
        elif self.condition == 'normal_depth':
            if B is None or Q is None or S0 is None or n is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Qn = Hydraulics.normal_flow(A=A, S_0=S0, n=n, B=B)
            dQn_dA = Hydraulics.dQn_dA(A=A, S=S0, n=n, B=B)
            return -Q / Qn**2 * dQn_dA if nondimensionalization else -dQn_dA
        
        elif self.condition == 'rating_curve':
            if Q is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Y = self.bed_level + A/B
            Qs = self.rating_curve.discharge(Y)
            dQs_dA = self.rating_curve.derivative_wrt_stage(Y) * dh_dA
            return -Q / Qs**2 * dQs_dA if nondimensionalization else -dQs_dA
            
        if self.condition == 'stage_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            ht = self.hydrograph.get_at(time=time) - self.bed_level
            return dh_dA / ht if nondimensionalization else dh_dA
                
    def df_dQ(self, time = None, area = None, width = None, bed_slope = None, roughness = None, nondimensionalization = False):
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 1 / self.hydrograph.get_at(time=time) if nondimensionalization else 1
            
        elif self.condition == 'fixed_depth':
            return 0.0
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 1 / Hydraulics.normal_flow(A=area, S_0=bed_slope, n=roughness, B=width) if nondimensionalization else 1
        
        elif self.condition == 'rating_curve':
            if width is None or area is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 1 / self.rating_curve.discharge(self.bed_level + area/width) if nondimensionalization else 1
            
        if self.condition == 'stage_hydrograph':
            return 0.0
    
    def df_dn(self, area, flow, width, bed_slope, roughness, nondimensionalization = False):
        if self.condition == 'normal_depth':
            Qn = Hydraulics.normal_flow(A=area, S_0=bed_slope, n=roughness, B=width)
            dQn_dn = Hydraulics.dQn_dn(A=area, S_0=bed_slope, n=roughness, B=width)
            return -flow / Qn**2 * dQn_dn if nondimensionalization else -dQn_dn
        else:
            df_dn = 0
        
        return df_dn
    
    def df_dYN(self, depth, nondimensionalization = False):
        if self.condition == 'fixed_depth':
            return -depth / (self.lumped_storage.stage - self.bed_level)**2 if nondimensionalization else -1
        else:
            return 0
                