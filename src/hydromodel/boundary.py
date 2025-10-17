from .rating_curve import RatingCurve
from .hydrograph import Hydrograph
from .hydraulics import normal_flow, dQn_dA, dQn_dn
from .lumped_storage import LumpedStorage

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
    
    def condition_residual(self, time = None, depth = None, width = None, flow_rate = None, bed_slope = None, roughness = None,
                           duration = None, vol_in = None, Y_old = None):
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph.get_at(time=time)
            return flow_rate - Q_t
            
        elif self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            elif self.initial_depth is None:
                raise ValueError("Fixed depth is not defined.")
            else:
                if self.lumped_storage is not None:                    
                    reservoir_stage = self.lumped_storage.mass_balance(
                        duration=duration,
                        vol_in=vol_in,
                        Y_old=Y_old,
                        time=time
                    )
                    reservoir_depth = reservoir_stage - self.bed_level

                    hl = self.lumped_storage.energy_loss(
                        Q=flow_rate, n=roughness, h=depth
                    )
                    
                    if not self.lumped_storage.stage_hydrograph:
                        self.lumped_storage.stage_hydrograph.append([time, reservoir_stage])
                    elif self.lumped_storage.stage_hydrograph[-1][0] == time:
                        self.lumped_storage.stage_hydrograph[-1][1] = reservoir_stage
                    else:
                        self.lumped_storage.stage_hydrograph.append([time, reservoir_stage])

                    return depth - (reservoir_depth + hl)
                else:
                    return depth - self.initial_depth
        
        elif self.condition == 'normal_depth':
            if width is None or depth is None or flow_rate is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Qn = normal_flow(A=width*depth, S_0=bed_slope, n=roughness, B=width)
            return flow_rate - Qn
        
        elif self.condition == 'rating_curve':
            if depth is None or flow_rate is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return flow_rate - self.rating_curve.discharge(stage=self.bed_level + depth, time=time)
            
        if self.condition == 'stage_hydrograph':
            if time is None or depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage_t = self.hydrograph.get_at(time=time)
            return self.bed_level + depth - stage_t
        
    def df_dA(self, area = None, flow_rate = None, width = None, bed_slope = None, roughness = None, time = None):
        dh_dA = 1/width
        
        if self.condition == 'flow_hydrograph':
            return 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            if self.lumped_storage is not None:
                dhl_dA = self.lumped_storage.dhl_dA(Q=flow_rate, n=roughness, h=area/width)
            else:
                dhl_dA = 0
            
            return dh_dA - dhl_dA
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 0 - dQn_dA(A=area, S=bed_slope, n=roughness, B=width)
        
        elif self.condition == 'rating_curve':
            if width is None or area is None or time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage = self.bed_level + area/width
            return 0 - self.rating_curve.dQ_dz(stage, time=time) * dh_dA
            
        elif self.condition == 'stage_hydrograph':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return dh_dA
        
    def df_dQ(self, area = None, flow_rate = None, roughness = None, depth = None, duration = None, vol_in = None, Y_old = None, time = None):
        if self.condition == 'flow_hydrograph':
            return 1
            
        elif self.condition == 'fixed_depth':
            if self.lumped_storage is not None:
                dY_new_dvol = self.lumped_storage.dY_new_dvol_in(
                    duration=duration,
                    vol_in=vol_in,
                    Y_old=Y_old,
                    time=time
                )
                dvol_dQ = 0.5 * duration
                
                dhl_dQ = self.lumped_storage.dhl_dQ(Q=flow_rate, n=roughness, h=depth)
                
                return 0 - (dY_new_dvol * dvol_dQ + dhl_dQ)
            else:
                return 0
        
        elif self.condition == 'normal_depth':
            return 1
        
        elif self.condition == 'rating_curve':
            return 1
        
        elif self.condition == 'stage_hydrograph':
            return 0
    
    def df_dn(self, depth, roughness, width, bed_slope, flow_rate=None):
        if self.condition == 'normal_depth':
            return 0 - dQn_dn(A=width*depth, S_0=bed_slope, n=roughness, B=width)
        
        elif self.condition == 'fixed_depth' and self.lumped_storage:
            dhl_dn = self.lumped_storage.dhl_dn(Q=flow_rate, n=roughness, h=depth)
            return 0 - (0 + dhl_dn)
                
        else:
            return 0
        
    def condition_type(self):
        if self.condition in['flow_hydrograph', 'normal_depth', 'rating_curve']:
            return 1
        else:
            return 0
    