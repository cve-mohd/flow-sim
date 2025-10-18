from .rating_curve import RatingCurve
from .hydrograph import Hydrograph
from .hydraulics import normal_flow, dQn_dA, dQn_dn
from .lumped_storage import LumpedStorage

class Boundary:
    """Contains the necessary functionality for upstream and downstream boundary handling.
    """
    def __init__(self,
                 condition: str,
                 bed_level: float,
                 chainage: int | float,
                 initial_depth: float = None,
                 rating_curve: RatingCurve = None,
                 hydrograph: Hydrograph = None):
        """Initializes a channel boundary

        Args:
            condition (str): Boundary condition type (e.g., 'flow_hydrograph').
            bed_level (float): Bed elevation above a datum.
            chainage (int | float): Horizontal distance from a datum (used to compute channel length).
            initial_depth (float, optional): Initial flow depth at the boundary. Defaults to None.
            rating_curve (RatingCurve, optional): Stage-discharge relationship for 'rating_curve' boundary conditions. Defaults to None.
            hydrograph (Hydrograph, optional): Flow or stage hydrograph for hydrograph-type boundary conditions. Defaults to None.
        """
        
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
        """Connect the boundary to a 0D storage element.

        Args:
            lumped_storage (LumpedStorage): The storage element.
        """
        self.lumped_storage = lumped_storage
    
    def condition_residual(self, time: int | float = None,
                           depth: float = None,
                           width: float = None,
                           flow_rate: float = None,
                           bed_slope: float = None,
                           roughness: float = None,
                           duration: int | float = None,
                           vol_in: float = None) -> float:
        """Computes the residual of the boundary condition equation.

        Args:
            time (int | float, optional): Current temporal coordinate. Defaults to None.
            depth (float, optional): Current flow depth at the boundary. Defaults to None.
            width (float, optional): Cross-sectional width of the channel at the boundary. Defaults to None.
            flow_rate (float, optional): Flow rate at the boundary. Defaults to None.
            bed_slope (float, optional): Bed slope at the boundary. Defaults to None.
            roughness (float, optional): Manning's roughness coefficient. Defaults to None.
            duration (int | float, optional): Time step. Defaults to None.
            vol_in (float, optional): Water volume passing through the boundary. Defaults to None.

        Returns:
            float: Residual
        """
        if self.condition == 'flow_hydrograph':
            if time is None or flow_rate is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            Q_t = self.hydrograph.get_at(time=time)
            return flow_rate - Q_t
            
        elif self.condition == 'fixed_depth':
            if depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
                        
            if self.lumped_storage:
                if roughness is None or duration is None or vol_in is None or time is None or flow_rate is None:
                    raise ValueError("Insufficient arguments for boundary condition.")
                
                k = time//duration
                if k == 1:
                    Y_old = depth + self.bed_level
                else:
                    Y_old = self.lumped_storage.stage_hydrograph[k-2][1]
                                    
                reservoir_stage = self.lumped_storage.mass_balance(
                    duration=duration,
                    vol_in=vol_in,
                    Y_old=Y_old,
                    time=time
                )
                head_loss = self.lumped_storage.energy_loss(
                    Q=flow_rate, n=roughness, h=depth
                )
                
                interface_stage = reservoir_stage + head_loss
                interface_depth = interface_stage - self.bed_level
                                
                if not self.lumped_storage.stage_hydrograph:
                    self.lumped_storage.stage_hydrograph.append([time, reservoir_stage])
                elif self.lumped_storage.stage_hydrograph[-1][0] == time:
                    self.lumped_storage.stage_hydrograph[-1][1] = reservoir_stage
                else:
                    self.lumped_storage.stage_hydrograph.append([time, reservoir_stage])

                return depth - interface_depth
            else:
                if self.initial_depth is None:
                    raise ValueError("Fixed depth is not defined.")
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
        
    def df_dA(self, area: float = None,
              flow_rate: float = None,
              width: float = None,
              bed_slope: float = None,
              roughness: float = None,
              time: int | float = None) -> float:
        """Computes the derivative of the boundary condition equation with respect to the cross-sectional flow area.

        Args:
            area (float, optional): Cross-sectional flow area at the boundary. Defaults to None.
            flow_rate (float, optional): Flow rate at the boundary. Defaults to None.
            width (float, optional): Cross-sectional width of the channel at the boundary. Defaults to None.
            bed_slope (float, optional): Bed slope at the boundary. Defaults to None.
            roughness (float, optional): Manning's roughness coefficient. Defaults to None.
            time (int | float, optional): Current temporal coordinate. Defaults to None.

        Returns:
            float: df/dA
        """        
        if self.condition == 'flow_hydrograph':
            return 0
            
        elif self.condition == 'fixed_depth':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            if self.lumped_storage is not None:
                if flow_rate is None or area is None or roughness is None:
                    raise ValueError("Insufficient arguments for boundary condition.")
                dhl_dA = self.lumped_storage.dhl_dA(Q=flow_rate, n=roughness, h=area/width)
            else:
                dhl_dA = 0
            
            dh_dA = 1/width
            return dh_dA - dhl_dA
        
        elif self.condition == 'normal_depth':
            if width is None or area is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 0 - dQn_dA(A=area, S=bed_slope, n=roughness, B=width)
        
        elif self.condition == 'rating_curve':
            if width is None or area is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            stage = self.bed_level + area/width
            dh_dA = 1/width
            return 0 - self.rating_curve.dQ_dz(stage, time=time) * dh_dA
            
        elif self.condition == 'stage_hydrograph':
            if width is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            dh_dA = 1/width
            return dh_dA
        
    def df_dQ(self, flow_rate: float = None,
              roughness: float = None,
              depth: float = None,
              duration: int | float = None,
              time: int | float = None,
              vol_in: float = None) -> float:
        """Computes the derivative of the boundary condition equation with respect to flow rate.

        Args:
            flow_rate (float, optional): Flow rate at the boundary. Defaults to None.
            roughness (float, optional): Manning's roughness coefficient. Defaults to None.
            depth (float, optional): Current flow depth at the boundary. Defaults to None.
            duration (int | float, optional): Time step. Defaults to None.
            vol_in (float, optional): Water volume passing through the boundary. Defaults to None.
            time (int | float, optional): Current temporal coordinate. Defaults to None.

        Returns:
            float: df/dQ
        """
        if self.condition == 'flow_hydrograph':
            return 1
            
        elif self.condition == 'fixed_depth':
            if self.lumped_storage is not None:
                if flow_rate is None or depth is None or roughness is None\
                    or duration is None or time is None or vol_in is None:
                    raise ValueError("Insufficient arguments for boundary condition.")
                
                k = time//duration
                if k == 1:
                    Y_old = depth + self.bed_level
                else:
                    Y_old = self.lumped_storage.stage_hydrograph[k-2][1]
                
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
    
    def df_dn(self, depth: float,
              roughness: float,
              width: float,
              bed_slope: float,
              flow_rate: float = None) -> float:
        """Computes the derivative of the boundary condition equation with respect to Manning's roughness.

        Args:
            depth (float, optional): Current flow depth at the boundary. Defaults to None.
            roughness (float, optional): Manning's roughness coefficient. Defaults to None.
            width (float, optional): Cross-sectional width of the channel at the boundary. Defaults to None.
            bed_slope (float, optional): Bed slope at the boundary. Defaults to None.
            flow_rate (float, optional): Flow rate at the boundary. Defaults to None.

        Returns:
            float: df/dn
        """
        if self.condition == 'normal_depth':
            if width is None or depth is None or bed_slope is None or roughness is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            return 0 - dQn_dn(A=width*depth, S_0=bed_slope, n=roughness, B=width)
        
        elif self.condition == 'fixed_depth' and self.lumped_storage:
            if flow_rate is None or roughness is None or depth is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            dhl_dn = self.lumped_storage.dhl_dn(Q=flow_rate, n=roughness, h=depth)
            return 0 - (0 + dhl_dn)
                
        else:
            return 0
        
    def condition_type(self) -> bool:
        """Returns 1 if the boundary equation is Q-dependant, and 0 otherwise.
        """
        return (self.condition in['flow_hydrograph', 'normal_depth', 'rating_curve'])
    