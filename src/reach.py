from src.boundary import Boundary
from src.utility import Hydraulics
import numpy as np

class Reach:
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
                 interpolation_method: str = 'GVF_equation'):
        """
        Initialized an instance.
        """
        self.conditions_initialized = False
        self.initial_flow_rate = initial_flow
        self.roughness = roughness
        self.dry_roughness = dry_roughness
        
        self.width = np.array([width, width], dtype=float)
        self.bed_level = np.array([upstream_boundary.bed_level,
                                    downstream_boundary.bed_level], dtype=float)
        self.level_chainages = np.array([upstream_boundary.chainage,
                                         downstream_boundary.chainage], dtype=float)
        self.width_chainages = np.array([upstream_boundary.chainage,
                                         downstream_boundary.chainage], dtype=float)
        
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
        """
        Computes the friction slope using Manning's equation.
        
        Parameters
        ----------
        A : float
            The cross-sectional flow area.
        Q : float
            The discharge.
            
        Returns
        -------
        float
            The computed friction slope.
            
        """       
        n = self.get_n(A=A, i=i)
        B = self.width[i]
        
        Sf = Hydraulics.Sf(A=A, Q=Q, n=n, B=B)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            Sc = Hydraulics.Sc(A=A, Q=Q, n=n, B=B, rc=rc)
        else:
            Sc = 0
            
        return Sf + Sc
    
    def dSe_dA(self, A: float, Q: float, i: int) -> float:
        n = self.get_n(A=A, i=i)
        B = self.width[i]
        
        dSf_dA = Hydraulics.dSf_dA(A=A, Q=Q, n=n, B=B) + Hydraulics.dSf_dn(A=A, Q=Q, n=n, B=B) * self.dn_dA(A, i=i)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dA = Hydraulics.dSc_dA(A=A, Q=Q, n=n, B=B, rc=rc) + Hydraulics.dSc_dn(A=A, Q=Q, n=n, B=B, rc=rc) * self.dn_dA(A, i=i)
        else:
            dSc_dA = 0
        
        return dSf_dA + dSc_dA
    
    def dSe_dQ(self, A: float, Q: float, i: int) -> float:
        n = self.get_n(A=A, i=i)
        B = self.width[i]
        
        dSf_dQ = Hydraulics.dSf_dQ(A=A, Q=Q, n=n, B=B)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dQ = Hydraulics.dSc_dA(A=A, Q=Q, n=n, B=B, rc=rc) + Hydraulics.dSc_dn(A=A, Q=Q, n=n, B=B, rc=rc) * self.dn_dA(A, i=i)
        else:
            dSc_dQ = 0
                
        return dSf_dQ + dSc_dQ

    def normal_flow(self, A: float, i: int):
        n = self.get_n(A=A, i=i)
        B = self.width[i]
        S_0 = self.bed_slopes[i]
        
        return Hydraulics.normal_flow(A=A, S_0=S_0, n=n, B=B)
        
    def normal_area(self, Q: float, i: int):
        B = self.width[i]
        h = self.downstream_boundary.initial_depth
        h = 1 if h is None else h
        A_guess = h * B
        n = self.get_n(A=A_guess, i=i)
        S_0 = self.bed_slopes[i]
        
        return Hydraulics.normal_area(Q=Q, A_guess=A_guess, S_0=S_0, n=n, B=B)
            
    def initialize_conditions(self, n_nodes: int) -> None:
        """
        Computes the initial conditions.
        Computes the initial values of the flow variables at each node
        using the initial boundary values.
        
        Parameters
        ----------
        n_nodes : int
            The number of spatial nodes along the channel, including the two boundaries.
            
        Returns
        -------
        None.

        """
        self.initialize_geometry(n_nodes=n_nodes)
        self.initial_conditions = np.zeros(shape=(n_nodes, 2), dtype=float)
        
        if self.interpolation_method == 'linear':
            y_0 = self.upstream_boundary.initial_depth
            y_n = self.downstream_boundary.initial_depth
                    
            for i in range(n_nodes):
                distance = self.length * i / (n_nodes-1)
                
                y = y_0 + (y_n - y_0) * distance / self.length
                A, Q = y * self.width[i], self.initial_flow_rate
                
                self.initial_conditions[i, 0] = A
                self.initial_conditions[i, 1] = Q
                
        elif self.interpolation_method == 'GVF_equation':
            dx = self.length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth
            
            # Add last node
            self.initial_conditions[n_nodes-1, 0] = h*self.width[-1]
            self.initial_conditions[n_nodes-1, 1] = self.initial_flow_rate

            for i in reversed(range(n_nodes-1)):
                distance = i * dx
    
                A, Q, B = self.width[i] * h, self.initial_flow_rate, self.width[i]
                Sf = self.Se(A, Q, i)
                
                Fr = Hydraulics.froude_num(A, Q, B)
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
    
    def get_n(self, A: float, i: int, steepness = 0.15):
        if self.dry_roughness is None or not self.conditions_initialized:
            return self.roughness
        
        wet_h = self.wet_depth(i)
        h = A / self.width[i]
        
        return Hydraulics.effective_roughness(depth=h, roughness=self.roughness, dry_roughness=self.dry_roughness, wet_depth=wet_h, steepness=steepness)
              
    def dn_dA(self, A: float, i: int, steepness = 0.15):
        if self.dry_roughness is None:
            return 0
        
        B = self.width[i]
        wet_h = self.wet_depth(i)
        
        dn_dh = Hydraulics.dn_dh(depth=A/B,
                                 steepness=steepness,
                                 roughness=self.roughness,
                                 dry_roughness=self.dry_roughness,
                                 wet_depth=wet_h)
        # dn/dA = dn/dh * dh/dA, dh/dA = 1/B
        return dn_dh * 1./B
    
    def set_coords(self, coords, chainages):
        self.coords_chainages = np.asarray(chainages, dtype=float)
        self.coords = np.asarray(coords, dtype=float)
        
        self.upstream_boundary.chainage = self.coords_chainages[0]
        self.downstream_boundary.chainage = self.coords_chainages[-1]
        self.length = self.downstream_boundary.chainage - self.upstream_boundary.chainage
        self.coordinated = True
        
    def set_intermediate_widths(self, widths, chainages):
        widths    = np.asarray(widths,    dtype=float)
        chainages = np.asarray(chainages, dtype=float)

        if widths.shape != chainages.shape:
            raise ValueError("Widths and chainages must have the same length.")

        # prepend upstream values if needed
        if chainages[0] > self.upstream_boundary.chainage:
            widths    = np.insert(widths,    0, self.width[0])
            chainages = np.insert(chainages, 0, self.upstream_boundary.chainage)

        self.width = widths
        self.width_chainages = chainages

    def set_intermediate_bed_levels(self, bed_levels, chainages):
        bed_levels = np.asarray(bed_levels, dtype=float)
        chainages  = np.asarray(chainages,  dtype=float)

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

    def initialize_geometry(self, n_nodes):
        from numpy import interp, gradient, linspace, array, trapezoid
        from src.utility import compute_radii_curv
        
        self.chainages = linspace(
            start=self.upstream_boundary.chainage,
            stop=self.downstream_boundary.chainage,
            num=n_nodes
        )

        if self.coordinated:
            x = interp(self.chainages, self.coords_chainages, self.coords[:, 0])
            y = interp(self.chainages, self.coords_chainages, self.coords[:, 1])
            self.curv, self.radii_curv = compute_radii_curv(x_coords=x, y_coords=y)

        self.width = interp(
            self.chainages,
            array(self.width_chainages, dtype=float),
            array(self.width, dtype=float)
        )
        self.bed_level = interp(
            self.chainages,
            array(self.level_chainages, dtype=float),
            array(self.bed_level, dtype=float)
        )
        self.bed_slopes = -gradient(self.bed_level, self.chainages)
        self.surface_area = trapezoid(self.width, self.chainages)
    
    def wet_depth(self, i):
        return self.initial_conditions[i, 0] / self.width[i]
    