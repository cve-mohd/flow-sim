from .rating_curve import RatingCurve
from .hydrograph import Hydrograph
from .hydraulics import normal_flow, dQn_dA
from .lumped_storage import LumpedStorage
from .cross_section import CrossSection

class Boundary:
    """Contains the necessary functionality for upstream and downstream boundary handling.
    """
    def __init__(self,
                 condition: str,
                 chainage: int | float,
                 bed_level: float = None,
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
        self.cross_section: CrossSection = None
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
    
    def condition_residual(self,
                           depth: float,
                           flow: float,
                           time: int | float = None,
                           duration: int | float = None,
                           vol_in: float = None) -> float:
        """Computes the residual of the boundary condition equation.

        Args:
            area (float, optional): Current flow area at the boundary. Defaults to None.
            time (int | float, optional): Current temporal coordinate. Defaults to None.
            depth (float, optional): Current flow depth at the boundary. Defaults to None.
            hydraulic_radius (float, optional): Hydraulic radius. Defaults to None.
            flow_rate (float, optional): Flow rate at the boundary. Defaults to None.
            bed_slope (float, optional): Bed slope at the boundary. Defaults to None.
            roughness (float, optional): Manning's roughness coefficient. Defaults to None.
            duration (int | float, optional): Time step. Defaults to None.
            vol_in (float, optional): Water volume passing through the boundary. Defaults to None.

        Returns:
            float: Residual
        """
        hw = self.cross_section.z_min + depth
        S0 = self.cross_section.bed_slope
        
        unknown = flow if self.condition_type() else depth
        if unknown is None:
            raise ValueError("Insufficient arguments for boundary condition.")
        
        if self.condition == 'flow_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            target = self.hydrograph.get_at(time=time)
        
        elif self.condition == 'normal_depth':
            target = normal_flow(bed_slope=S0, K=self.cross_section.conveyance(hw=hw))
        
        elif self.condition == 'rating_curve':
            target = self.rating_curve.discharge(stage=self.bed_level+depth, time=time)
        
        elif self.condition == 'fixed_depth':
            if self.lumped_storage is None:
                target = self.initial_depth
            else:
                if duration is None or vol_in is None or time is None:
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
                
                R = self.cross_section.hydraulic_radius(hw=hw)
                n = self.cross_section.get_equivalent_n(hw=hw)
                area = self.cross_section.area(hw=hw)
                head_loss = self.lumped_storage.energy_loss(
                    entry_area=area, flow=flow, roughness=n, hydraulic_radius=R
                )
                
                interface_stage = reservoir_stage + head_loss
                                
                if not self.lumped_storage.stage_hydrograph:
                    self.lumped_storage.stage_hydrograph.append([time, reservoir_stage])
                elif self.lumped_storage.stage_hydrograph[-1][0] == time:
                    self.lumped_storage.stage_hydrograph[-1][1] = reservoir_stage
                else:
                    self.lumped_storage.stage_hydrograph.append([time, reservoir_stage])
                
                target = interface_stage - self.bed_level
            
        if self.condition == 'stage_hydrograph':
            if time is None:
                raise ValueError("Insufficient arguments for boundary condition.")
            
            target = self.hydrograph.get_at(time=time) - self.bed_level
        
        return unknown - target
        
    def df_dh(self,
              depth: float,
              flow_rate: float,
              time: int | float = None) -> float:
        """Computes the derivative of the boundary condition equation with respect to the cross-sectional flow area.

        Args:
            area (float, optional): Cross-sectional flow area at the boundary. Defaults to None.
            flow_rate (float, optional): Flow rate at the boundary. Defaults to None.
            bed_slope (float, optional): Bed slope at the boundary. Defaults to None.
            roughness (float, optional): Manning's roughness coefficient. Defaults to None.
            time (int | float, optional): Current temporal coordinate. Defaults to None.

        Returns:
            float: df/dA
        """        
        if self.condition == 'flow_hydrograph':
            return 0
            
        hw = depth + self.bed_level
        S0 = self.cross_section.bed_slope
        
        dh_dA = self.cross_section.dh_dA(hw=hw)
            
        if self.condition == 'fixed_depth':            
            if self.lumped_storage is not None:
                R = self.cross_section.hydraulic_radius(hw=hw)
                n = self.cross_section.get_equivalent_n(hw=hw)
                dR_dA = self.cross_section.dR_dA(hw=hw)
                area = self.cross_section.area(hw=hw)
                dhl_dA = self.lumped_storage.dhl_dA(entry_area=area, flow=flow_rate, roughness=n, hydraulic_radius=R, dR_dA=dR_dA)
            else:
                dhl_dA = 0
            
            return 1 - dhl_dA / dh_dA
        
        elif self.condition == 'normal_depth':
            return 0 - dQn_dA(S_0=S0, dK_dA=self.cross_section.dK_dA(hw=hw)) / dh_dA
        
        elif self.condition == 'rating_curve':
            stage = self.bed_level + depth
            return 0 - self.rating_curve.dQ_dz(stage, time=time)
            
        elif self.condition == 'stage_hydrograph':
            return 1
        
    def df_dQ(self,
              depth: float,
              flow_rate: float,
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
        if self.condition_type():
            return 1
        
        hw = depth + self.bed_level
                
        if self.condition == 'fixed_depth':
            if self.lumped_storage is not None:
                if duration is None or time is None or vol_in is None:
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
                
                R = self.cross_section.hydraulic_radius(hw=hw)
                n = self.cross_section.get_equivalent_n(hw=hw)
                area = self.cross_section.area(hw=hw)
                dhl_dQ = self.lumped_storage.dhl_dQ(entry_area=area, flow=flow_rate, roughness=n, hydraulic_radius=R)
                
                return 0 - (dY_new_dvol * dvol_dQ + dhl_dQ)
            else:
                return 0
                
        elif self.condition == 'stage_hydrograph':
            return 0
            
    def condition_type(self) -> bool:
        """Returns 1 if the boundary equation is Q-dependant, and 0 otherwise.
        """
        return (self.condition in['flow_hydrograph', 'normal_depth', 'rating_curve'])
    