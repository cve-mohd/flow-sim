from .boundary import Boundary
from .utility import compute_radii_curv
from . import hydraulics
import numpy as np

class Channel:
    """
    Represents a channel with hydraulic and geometric attributes.        
    """
    def __init__(self,
                 upstream_boundary: Boundary,
                 downstream_boundary: Boundary,
                 width: float,
                 initial_flow: float,
                 roughness: float,
                 dry_roughness: float = None,
                 interpolation_method: str = 'GVF_equation',
                 n_steepness: float = 0.15):
        """Initializes a Channel object.

        Args:
            upstream_boundary (Boundary): Upstream boundary object.
            downstream_boundary (Boundary): Downstream boundary object.
            width (float): Cross-sectional width.
            initial_flow (float): Flow rate through the channel at t=0.
            roughness (float): Manning's roughness coefficient.
            dry_roughness (float, optional): Manning's roughness coefficient for initially-dry regions. Defaults to None.
            interpolation_method (str, optional): Method of computing initial conditions. Defaults to 'GVF_equation'.
            n_steepness (float, optional): Steepness of the n transition curve. Defaults to 0.15.
        """
        self.conditions_initialized = False
        self.initial_flow_rate = initial_flow
        self.roughness = roughness
        self.dry_roughness = dry_roughness
        self.n_steepness = n_steepness
        
        self.width = np.array([width, width], dtype=np.float64)
        self.bed_level = np.array([upstream_boundary.bed_level,
                                    downstream_boundary.bed_level], dtype=np.float64)
        self.level_chainages = np.array([upstream_boundary.chainage,
                                         downstream_boundary.chainage], dtype=np.float64)
        self.width_chainages = np.array([upstream_boundary.chainage,
                                         downstream_boundary.chainage], dtype=np.float64)
        
        self.length = downstream_boundary.chainage - upstream_boundary.chainage
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary       
                
        if interpolation_method in ['linear', 'GVF_equation', 'steady-state']:
            self.interpolation_method = interpolation_method
        else:
            raise ValueError("Invalid interpolation method.")
        
        self.initial_conditions = None
        self.coordinated = False
    
    def Se(self, A: float, Q: float, i: int) -> float:
        """Computes the energy slope at a location.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: Energy slope (Se)
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A=A)
        R = self.hydraulic_radius(i=i, h=h)
        
        Sf = hydraulics.Sf(A=A, Q=Q, n=n, R=R)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            Sc = hydraulics.Sc(h=h, A=A, Q=Q, n=n, R=R, rc=rc)
        else:
            Sc = 0
            
        return Sf + Sc
    
    def dSe_dA(self, A: float, Q: float, i: int) -> float:
        """Computes the derivative of the energy slope w.r.t. flow area.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: dSe/dA
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A=A)
        R = self.hydraulic_radius(i=i, h=h)
        dR_dA = self.dR_dA(i=i, h=h)
        T = self.top_width(i=i, h=h)
        
        dSf_dA = hydraulics.dSf_dA(A=A, Q=Q, n=n, R=R, dR_dA=dR_dA) + hydraulics.dSf_dn(A=A, Q=Q, n=n, R=R) * self.dn_dA(A=A, i=i)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dA = hydraulics.dSc_dA(h=h, A=A, Q=Q, n=n, R=R, rc=rc, dR_dA=dR_dA, T=T) + hydraulics.dSc_dn(h=h, A=A, Q=Q, n=n, R=R, rc=rc) * self.dn_dA(A=A, i=i)
        else:
            dSc_dA = 0
        
        return dSf_dA + dSc_dA
    
    def dSe_dQ(self, A: float, Q: float, i: int) -> float:
        """Computes the derivative of the energy slope w.r.t. flow rate.

        Args:
            A (float): Cross-sectional flow area.
            Q (float): Flow rate.
            i (int): Index of spatial node where Se is computed.

        Returns:
            float: dSe/dQ
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A=A)
        R = self.hydraulic_radius(i=i, h=h)
        dR_dA = self.dR_dA(i=i, h=h)
        T = self.top_width(i=i, h=h)
        
        dSf_dQ = hydraulics.dSf_dQ(A=A, Q=Q, n=n, R=R)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dQ = hydraulics.dSc_dA(h=h, A=A, Q=Q, n=n, R=R, rc=rc, dR_dA=dR_dA, T=T) + hydraulics.dSc_dn(h=h, A=A, Q=Q, n=n, R=R, rc=rc) * self.dn_dA(A=A, i=i)
        else:
            dSc_dQ = 0
                
        return dSf_dQ + dSc_dQ

    def normal_flow(self, A: float, i: int) -> float:
        """Computes the normal flow rate for a given flow area at a given location.

        Args:
            A (float): Cross-sectional flow area.
            i (int): Spatial node index.

        Returns:
            float: Normal flow rate.
        """
        n = self.get_n(A=A, i=i)
        h = self.depth_at(i=i, A=A)
        R = self.hydraulic_radius(i=i, h=h)
        S_0 = self.bed_slopes[i]
        
        return hydraulics.normal_flow(A=A, S_0=S_0, n=n, R=R)
        
    def normal_area(self, Q: float, i: int) -> float:
        """Computes the normal flow area for a given flow rate at a given location.

        Args:
            Q (float): Flow rate.
            i (int): Spatial node index.

        Returns:
            float: Normal flow area.
        """
        raise ValueError("channel.normal_area is WIP")
        R = self.hydraulic_radius(i=i, h=h)
        B = self.width[i]
        h = self.downstream_boundary.initial_depth
        h = 1 if h is None else h
        A_guess = h * B
        n = self.get_n(A=A_guess, i=i)
        S_0 = self.bed_slopes[i]
        
        return hydraulics.normal_area(Q=Q, A_guess=A_guess, S_0=S_0, n=n, R=R)
            
    def initialize_conditions(self, n_nodes: int) -> None:
        """
        Computes the initial values of the flow variables (A, Q) at each node.
        
        Parameters
        ----------
        n_nodes : int
            Number of spatial nodes along the channel.
            
        Returns
        -------
        None.

        """
        self.initialize_geometry(n_nodes=n_nodes)
        self.initial_conditions = np.zeros(shape=(n_nodes, 2), dtype=np.float64)
        
        if self.interpolation_method == 'linear':
            h0 = self.upstream_boundary.initial_depth
            hN = self.downstream_boundary.initial_depth
                    
            for i in range(n_nodes):
                distance = self.length * i / (n_nodes-1)
                
                h = h0 + (hN - h0) * distance / self.length
                A, Q = self.area_at(i=i, h=h), self.initial_flow_rate
                
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
                
        elif self.interpolation_method == 'GVF_equation':
            dx = self.length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth
            
            # Add last node
            self.initial_conditions[n_nodes-1, 0] = self.area_at(i=-1, h=h)
            self.initial_conditions[n_nodes-1, 1] = self.initial_flow_rate

            for i in reversed(range(n_nodes-1)):
                distance = i * dx
    
                A, Q, B = self.area_at(i=i, h=h), self.initial_flow_rate, self.width[i]
                Sf = self.Se(A, Q, i)
                
                Fr = hydraulics.froude_num(A, Q, B)
                denominator = 1 - Fr**2
                
                if abs(denominator) < 1e-6:
                    dhdx = 0.0
                else:
                    S0 = -(self.bed_level[i+1]-self.bed_level[i]) / dx
                    dhdx = (S0 - Sf) / denominator

                h -= dhdx * dx

                if h < 0:
                    raise ValueError("GVF failed.")

                A = h * B
                    
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
        
        elif self.interpolation_method == 'steady-state':
            for i in range(n_nodes):
                Q = self.initial_flow_rate
                A = self.normal_area(Q, i)
                
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
            
        else:
            raise ValueError("Invalid flow type.")
        
        self.conditions_initialized = True
    
    def get_n(self, A: float = None, i: int = None) -> float:
        """Retrieves Manning's roughness coefficient.

        Args:
            A (float, optional): Cross-sectional flow area. Defaults to None.
            i (int, optional): Index of spatial node. Defaults to None.

        Returns:
            float: n
        """
        
        if self.dry_roughness is None or not self.conditions_initialized:
            return self.roughness
        
        if A is None or i is None:
            raise ValueError("Insufficient parameters.")
        
        wet_h = self.wet_depth(i)
        h = A / self.width[i]
        
        return hydraulics.effective_roughness(depth=h, wet_roughness=self.roughness, dry_roughness=self.dry_roughness, wet_depth=wet_h, steepness=self.n_steepness)
              
    def dn_dA(self, A: float, i: int) -> float:
        """Computes the derivative of Manning's roughness coefficient w.r.t. flow area.

        Args:
            A (float): Cross-sectional flow area.
            i (int): Index of spatial node.

        Returns:
            float: dn/dA
        """
        
        if self.dry_roughness is None:
            return 0
        
        B = self.width[i]
        wet_h = self.wet_depth(i)
        
        dn_dh = hydraulics.dn_dh(depth=A/B,
                                 steepness=self.n_steepness,
                                 roughness=self.roughness,
                                 dry_roughness=self.dry_roughness,
                                 wet_depth=wet_h)
        # dn/dA = dn/dh * dh/dA, dh/dA = 1/B
        return dn_dh * 1./B
    
    def set_coords(self, coords: list | np.ndarray, chainages: list | np.ndarray) -> None:
        """Sets horizontal coordinates along the channel.

        Args:
            coords (list | np.ndarray): Coordinates (list of x,y pairs).
            chainages (list | np.ndarray): Respective chainages along the channel.
        """
        self.coords_chainages = np.asarray(chainages, dtype=np.float64)
        self.coords = np.asarray(coords, dtype=np.float64)
        
        self.upstream_boundary.chainage = self.coords_chainages[0]
        self.downstream_boundary.chainage = self.coords_chainages[-1]
        self.length = self.downstream_boundary.chainage - self.upstream_boundary.chainage
        self.coordinated = True
        
    def set_intermediate_widths(self, widths: list | np.ndarray, chainages: list | np.ndarray) -> None:
        """Sets cross-sectional width values at positions along the channel.

        Args:
            widths (list | np.ndarray): Cross-sectional width values.
            chainages (list | np.ndarray): Respective chainages along the channel.
        """
        widths    = np.asarray(widths,    dtype=np.float64)
        chainages = np.asarray(chainages, dtype=np.float64)
        
        if widths.shape != chainages.shape:
            raise ValueError("Widths and chainages must have the same length.")

        # prepend upstream values if needed
        if chainages[0] > self.upstream_boundary.chainage:
            widths    = np.insert(widths,    0, self.width[0])
            chainages = np.insert(chainages, 0, self.upstream_boundary.chainage)

        self.width = widths
        self.width_chainages = chainages

    def set_intermediate_bed_levels(self, bed_levels: list | np.ndarray, chainages: list | np.ndarray) -> None:
        """Sets bed elevation values at positions along the channel.

        Args:
            bed_levels (list | np.ndarray): Bed elevation values.
            chainages (list | np.ndarray): Respective chainages along the channel.
        """
        bed_levels = np.asarray(bed_levels, dtype=np.float64)
        chainages  = np.asarray(chainages,  dtype=np.float64)

        if bed_levels.shape != chainages.shape:
            raise ValueError("Bed levels and chainages must have the same length.")
                            
        if chainages[0] > self.upstream_boundary.chainage:
            bed_levels = np.insert(bed_levels, 0, self.upstream_boundary.bed_level)
            chainages  = np.insert(chainages,  0, self.upstream_boundary.chainage)
        else:
            self.upstream_boundary.bed_level = bed_levels[0]
        
        if chainages[-1] < self.downstream_boundary.chainage:
            bed_levels = np.append(bed_levels, self.downstream_boundary.bed_level)
            chainages = np.append(chainages, self.downstream_boundary.chainage)
        else:
            self.downstream_boundary.bed_level = bed_levels[-1]
        
        self.bed_level = bed_levels
        self.level_chainages = chainages

    def initialize_geometry(self, n_nodes: int) -> None:
        """Interpolates geometric attributes along the channel.

        Args:
            n_nodes (int): Number of spatial nodes along the channel.
        """
        self.chainages = np.linspace(
            start=self.upstream_boundary.chainage,
            stop=self.downstream_boundary.chainage,
            num=n_nodes
        )

        if self.coordinated:
            x = np.interp(self.chainages, self.coords_chainages, self.coords[:, 0])
            y = np.interp(self.chainages, self.coords_chainages, self.coords[:, 1])
            self.curv, self.radii_curv = compute_radii_curv(x_coords=x, y_coords=y)

        self.width = np.interp(
            self.chainages,
            np.array(self.width_chainages, dtype=np.float64),
            np.array(self.width, dtype=np.float64)
        )
        self.bed_level = np.interp(
            self.chainages,
            np.array(self.level_chainages, dtype=np.float64),
            np.array(self.bed_level, dtype=np.float64)
        )
        self.bed_slopes = -np.gradient(self.bed_level, self.chainages)
        self.surface_area = np.trapezoid(self.width, self.chainages)
    
    def wet_depth(self, i: int) -> float:
        """Retrieves the depth value of the initially-submerged portion of the channel at a given location.

        Args:
            i (int): Spatial node index.

        Returns:
            float: Depth value.
        """
        return self.initial_conditions[i, 0] / self.width[i]
    
    def area_at(self, i, h):
        return self.width[i] * h
    
    def depth_at(self, i, A):
        return A/self.width[i]

    def hydraulic_radius(self, i, h):
        return self.area_at(i, h) / self.wetted_perimeter(i, h)
    
    def wetted_perimeter(self, i, h):
        return self.width[i] + 2 * h
    
    def dR_dA(self, i, h):
        B = self.width[i]
        A = self.area_at(i=i, h=h)
        P = B + 2.0 * h
        dP_dA = 2.0 / B

        return (P - A * dP_dA) / (P**2)

    def top_width(self, i, h):
        return self.width[i]