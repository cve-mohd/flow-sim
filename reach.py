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

    def Sf(self, A: float, Q: float, B: float, wet_depth: float = None) -> float:
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
        n = self.get_n(A=A, B=B, wet_depth=wet_depth)
        return Hydraulics.Sf(A=A, Q=Q, n=n, B=B)
    
    def dSf_dA(self, A: float, Q: float, B: float, wet_depth: float = None) -> float:
        n = self.get_n(A=A, B=B, wet_depth=wet_depth)
        
        d1 = Hydraulics.dSf_dA(A=A, Q=Q, n=n, B=B)
        d2 = Hydraulics.dSf_dn(A=A, Q=Q, n=n, B=B) * self.dn_dA(A, B, wet_depth=wet_depth)
        
        return d1 + d2
    
    def dSf_dQ(self, A: float, Q: float, B: float, wet_depth: float = None) -> float:
        n = self.get_n(A=A, B=B, wet_depth=wet_depth)
        return Hydraulics.dSf_dQ(A=A, Q=Q, n=n, B=B)

    def normal_flow(self, A: float, B: float, S_0: float, wet_depth: float = None):
        n = self.get_n(A=A, B=B, wet_depth=wet_depth)
        return Hydraulics.normal_flow(A=A, S_0=S_0, n=n, B=B)
        
    def normal_area(self, Q: float, B: float, S_0: float, wet_depth: float = None):
        A_guess = self.downstream_boundary.initial_depth * B
        n = self.get_n(A=A_guess, B=B, wet_depth=wet_depth)
        
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
            from scipy.constants import g

            dx = self.length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth
            
            # Add last node
            self.initial_conditions = [(h*self.widths[-1], self.initial_flow_rate)]

            for i in reversed(range(n_nodes-1)):
                distance = i * dx
    
                A, Q, B = self.widths[i] * h, self.initial_flow_rate, self.widths[i]
                Sf = self.Sf(A, Q, B, wet_depth=A/B)
                
                Fr2 = Q**2 * B / (g * A**3)
                denominator = 1 - Fr2
                
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
    
    def get_n(self, A: float, B: float, wet_depth: float = None, steepness = 0.15):
        if self.dry_roughness is None:
            return self.roughness
        
        if wet_depth is None:
            raise ValueError("Wet depth was not provided.")
        
        h = A/B
        return Hydraulics.effective_roughness(depth=h, roughness=self.roughness, dry_roughness=self.dry_roughness, wet_depth=wet_depth, steepness=steepness)
      
    def dn_dA(self, A: float, B: float, wet_depth: float = None, steepness = 0.15):
        if self.dry_roughness is None:
            return 0
        
        if wet_depth is None:
            raise ValueError("Wet depth was not provided.")
        
        dn_dh = Hydraulics.dn_dh(depth=A/B,
                                 steepness=steepness,
                                 roughness=self.roughness,
                                 dry_roughness=self.dry_roughness,
                                 wet_depth=wet_depth)
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
        from utility import compute_curvature
        
        chainages = linspace(
            start=self.upstream_boundary.chainage,
            stop=self.downstream_boundary.chainage,
            num=n_nodes
        )

        if self.coordinated:
            x_coords = interp(chainages, self.coords_chainages, self.x_coords)
            y_coords = interp(chainages, self.coords_chainages, self.y_coords)
            
            kappa = compute_curvature(x_coords=x_coords, y_coords=y_coords)
            self.curvature = kappa.tolist()

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
        
        self.upstream_boundary.chainage = self.coords_chainages[0]
        self.downstream_boundary.chainage = self.coords_chainages[-1]
        self.coordinated = True