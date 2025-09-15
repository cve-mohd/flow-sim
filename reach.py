from boundary import Boundary
from utility import Hydraulics

class Reach:
    """
    Represents a channel with hydraulic and geometric attributes.
    
    Attributes
    ----------
    bed_slope : float
        The slope of the channel bed.
    manning_co : float
        The channel's Manning coefficient or coefficient of roughness.
    width : float
        The width of the channel cross-section.
    length : float
        The length of the channel segment to be analyzed.
    initial_conditions : list of tuple of float.
        A list of tuples, where each tuple stores the initial values of the
        flow variables (A and Q) at the spatial point corresponding to its index.
        
    """
    def __init__(self,
                 upstream_boundary: Boundary,
                 downstream_boundary: Boundary,
                 width: float,
                 initial_flow_rate: float,
                 roughness: float,
                 dry_roughness: float = None,
                 interpolation_method: str = 'GVF_equation'):
        """
        Initialized an instance.

        Parameters
        ----------
        manning_co : float
            The channel's Manning coefficient or coefficient of roughness.
        width : float
            The width of the channel cross-section.
        length : float
            The length of the channel.

        """
        self.conditions_initialized = False
        self.widths = [width, width]
        self.initial_flow_rate = initial_flow_rate
        self.roughness = roughness
        self.bed_levels = [upstream_boundary.bed_level, downstream_boundary.bed_level]
        
        self.dry_roughness = dry_roughness
        
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary
        
        self.length = self.downstream_boundary.chainage - self.upstream_boundary.chainage
                
        if interpolation_method in ['linear', 'GVF_equation']:
            self.interpolation_method = interpolation_method
        
        self.initial_conditions = []
        
        self.level_chainages = [self.upstream_boundary.chainage, self.downstream_boundary.chainage]
        self.width_chainages = [self.upstream_boundary.chainage, self.downstream_boundary.chainage]
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
        B = self.widths[i]
        
        Sf = Hydraulics.Sf(A=A, Q=Q, n=n, B=B)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            Sc = Hydraulics.Sc(A=A, Q=Q, n=n, B=B, rc=rc)
        else:
            Sc = 0
            
        return Sf + Sc
    
    def dSe_dA(self, A: float, Q: float, i: int) -> float:
        n = self.get_n(A=A, i=i)
        B = self.widths[i]
        
        dSf_dA = Hydraulics.dSf_dA(A=A, Q=Q, n=n, B=B) + Hydraulics.dSf_dn(A=A, Q=Q, n=n, B=B) * self.dn_dA(A, i=i)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dA = Hydraulics.dSc_dA(A=A, Q=Q, n=n, B=B, rc=rc) + Hydraulics.dSc_dn(A=A, Q=Q, n=n, B=B, rc=rc) * self.dn_dA(A, i=i)
        else:
            dSc_dA = 0
        
        return dSf_dA + dSc_dA
    
    def dSe_dQ(self, A: float, Q: float, i: int) -> float:
        n = self.get_n(A=A, i=i)
        B = self.widths[i]
        
        dSf_dQ = Hydraulics.dSf_dQ(A=A, Q=Q, n=n, B=B)
        
        if self.coordinated:
            rc = self.radii_curv[i]
            dSc_dQ = Hydraulics.dSc_dA(A=A, Q=Q, n=n, B=B, rc=rc) + Hydraulics.dSc_dn(A=A, Q=Q, n=n, B=B, rc=rc) * self.dn_dA(A, i=i)
        else:
            dSc_dQ = 0
                
        return dSf_dQ + dSc_dQ

    def normal_flow(self, A: float, i: int):
        n = self.get_n(A=A, i=i)
        B = self.widths[i]
        S_0 = self.bed_slopes[i]
        
        return Hydraulics.normal_flow(A=A, S_0=S_0, n=n, B=B)
        
    def normal_area(self, Q: float, i: int):
        B = self.widths[i]
        A_guess = self.downstream_boundary.initial_depth * B
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
        
        if self.interpolation_method == 'linear':
            y_0 = self.upstream_boundary.initial_depth
            y_n = self.downstream_boundary.initial_depth
                    
            for i in range(n_nodes):
                distance = self.length * float(i) / float(n_nodes-1)
                
                y = y_0 + (y_n - y_0) * distance / self.length
                
                A, Q = y * self.widths[i], self.initial_flow_rate
                self.initial_conditions.append((A, Q))
                
        elif self.interpolation_method == 'GVF_equation':
            dx = self.length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth
            
            # Add last node
            self.initial_conditions = [(h*self.widths[-1], self.initial_flow_rate)]

            for i in reversed(range(n_nodes-1)):
                distance = i * dx
    
                A, Q, B = self.widths[i] * h, self.initial_flow_rate, self.widths[i]
                Sf = self.Se(A, Q, i)
                
                Fr = Hydraulics.froude_num(A, Q, B)
                denominator = 1 - Fr**2
                
                if abs(denominator) < 1e-6:
                    dhdx = 0.0
                else:
                    S0 = -(self.bed_levels[i+1]-self.bed_levels[i])/dx
                    dhdx = (S0 - Sf) / denominator

                h -= dhdx * dx

                if h < 0:
                    raise ValueError("GVF failed.")

                A = h * B
                    
                self.initial_conditions.insert(0, (A, Q))
                
        else:
            raise ValueError("Invalid flow type.")
        
        self.conditions_initialized = True
    
    def get_n(self, A: float, i: int, steepness = 0.15):
        if self.dry_roughness is None or not self.conditions_initialized:
            return self.roughness
        
        wet_h = self.wet_depth(i)
        h = A/self.widths[i]
        
        return Hydraulics.effective_roughness(depth=h, roughness=self.roughness, dry_roughness=self.dry_roughness, wet_depth=wet_h, steepness=steepness)
              
    def dn_dA(self, A: float, i: int, steepness = 0.15):
        if self.dry_roughness is None:
            return 0
        
        B = self.widths[i]
        wet_h = self.wet_depth(i)
        
        dn_dh = Hydraulics.dn_dh(depth=A/B,
                                 steepness=steepness,
                                 roughness=self.roughness,
                                 dry_roughness=self.dry_roughness,
                                 wet_depth=wet_h)
        # dn/dA = dn/dh * dh/dA, dh/dA = 1/B
        return dn_dh * 1./B
    
    def set_intermediate_widths(self, widths: list, chainages: list):
        if len(widths) != len(chainages):
            raise ValueError("")
        
        self.widths = [self.widths[0]] + widths
        self.width_chainages = [self.upstream_boundary.chainage] + chainages

    def set_intermediate_bed_levels(self, bed_levels: list, chainages: list):
        if len(bed_levels) != len(chainages):
            raise ValueError("")
        
        self.bed_levels = [self.upstream_boundary.bed_level] + bed_levels + [self.downstream_boundary.bed_level]
        self.level_chainages = [self.upstream_boundary.chainage] + chainages + [self.downstream_boundary.chainage]

    def initialize_geometry(self, n_nodes):
        from numpy import interp, linspace, gradient, array, trapezoid
        from utility import compute_radii_curv
        
        chainages = linspace(
            start=self.upstream_boundary.chainage,
            stop=self.downstream_boundary.chainage,
            num=n_nodes
        )

        if self.coordinated:
            x_coords = interp(chainages, self.coords_chainages, self.x_coords)
            y_coords = interp(chainages, self.coords_chainages, self.y_coords)
            
            rc = compute_radii_curv(x_coords=x_coords, y_coords=y_coords)
            self.radii_curv = rc.tolist()

        widths = interp(
            chainages,
            array(self.width_chainages, dtype=float),
            array(self.widths, dtype=float)
        )
        bed_levels = interp(
            chainages,
            array(self.level_chainages, dtype=float),
            array(self.bed_levels, dtype=float)
        )

        self.widths = widths.astype(float).tolist()
        self.bed_levels = bed_levels.astype(float).tolist()
        self.bed_slopes = -gradient(bed_levels, chainages)

        surface_area = trapezoid(widths, chainages)
        self.surface_area = float(surface_area)

    def set_coords(self, coords, chainages):
        from numpy import asarray
        
        self.coords_chainages = asarray(chainages, dtype=float)
        x, y = zip(*coords)
        self.x_coords, self.y_coords = list(x), list(y)
        
        self.upstream_boundary.chainage = float(self.coords_chainages[0])
        self.downstream_boundary.chainage = float(self.coords_chainages[-1])
        self.length = self.downstream_boundary.chainage - self.upstream_boundary.chainage
        self.coordinated = True
    
    def wet_depth(self, i):
        return self.initial_conditions[i][0] / self.widths[i]
    