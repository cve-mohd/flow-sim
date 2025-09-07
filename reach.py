from boundary import Boundary
from math import sqrt
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
                 channel_roughness: float,
                 floodplain_roughness: float = None,
                 bankful_depth: float = None,
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
        self.width = width
        self.conditions_initialized = False
        self.fixed_width = True
        self.fixed_bed_slope = True
        self.initial_flow_rate = initial_flow_rate
        self.channel_roughness = channel_roughness
        
        self.bankful_depth = bankful_depth
        self.floodplain_roughness = floodplain_roughness
        
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary
        
        self.length = self.downstream_boundary.chainage - self.upstream_boundary.chainage
        
        z1 = self.upstream_boundary.bed_level
        z2 = self.downstream_boundary.bed_level
        self.bed_slope = float(z1 - z2) / self.length
        
        if interpolation_method in ['linear', 'GVF_equation']:
            self.interpolation_method = interpolation_method
        
        self.initial_conditions = []

    def Sf(self, A: float, Q: float) -> float:
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
        n = self.get_n(A=A)
        return Hydraulics.Sf(A=A, Q=Q, n=n, B=self.width)
    
    def dSf_dA(self, A: float, Q: float) -> float:
        n = self.get_n(A=A)
        
        d1 = Hydraulics.dSf_dA(A=A, Q=Q, n=n, B=self.width)
        d2 = Hydraulics.dSf_dn(A=A, Q=Q, n=n, B=self.width) * Hydraulics.dn_dh(depth=A/self.width,
                                                                               steepness=0.15,
                                                                               channel_roughness=self.channel_roughness,
                                                                               floodplain_roughness=self.floodplain_roughness,
                                                                               bankful_depth=self.bankful_depth) * 1./self.width
        
        return d1 + d2
    
    def dSf_dQ(self, A: float, Q: float) -> float:
        n = self.get_n(A=A)
        return Hydraulics.dSf_dQ(A=A, Q=Q, n=n, B=self.width)

    def normal_flow(self, A):
        n = self.get_n(A=A)
        return Hydraulics.normal_flow(A=A, S=self.bed_slope, n=n, B=self.width)
        
    def normal_area(self, Q):
        A_guess = self.downstream_boundary.initial_depth * self.width
        n = self.get_n(A=A_guess)
        
        return Hydraulics.normal_area(Q=Q, A_guess=A_guess, S_0=self.bed_slope, n=n, B=self.width)
            
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
        if self.interpolation_method == 'linear':
            y_0 = self.upstream_boundary.initial_depth
            y_n = self.downstream_boundary.initial_depth
                    
            for i in range(n_nodes):
                distance = self.length * float(i) / float(n_nodes-1)
                
                y = y_0 + (y_n - y_0) * distance / self.length
                
                A, Q = y * self.width, self.initial_flow_rate
                self.initial_conditions.append((A, Q))
                
        elif self.interpolation_method == 'GVF_equation':
            from scipy.constants import g

            dx = self.length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth

            for i in reversed(range(n_nodes)):
                distance = i * dx
    
                A, Q = self.width * h, self.initial_flow_rate
                Sf = self.Sf(A, Q)
                Fr2 = Q ** 2 / (g * A ** 3 / self.width)

                denominator = 1 - Fr2
                if abs(denominator) < 1e-6:
                    dhdx = 0.0
                else:
                    dhdx = (self.bed_slope - Sf) / denominator

                if i < n_nodes - 1:
                    h -= dhdx * dx

                if h < 0:
                    raise ValueError("GVF failed.")

                A = h * self.width
                    
                self.initial_conditions.insert(0, (A, Q))
                
        else:
            raise ValueError("Invalid flow type.")
        
        self.conditions_initialized = True
    
    def get_n(self, A, steepness = 0.15):
        h = A/self.width
        return Hydraulics.effective_roughness(depth=h,
                                              steepness=steepness,
                                              channel_roughness=self.channel_roughness,
                                              floodplain_roughness=self.floodplain_roughness,
                                              bankful_depth=self.bankful_depth)
      
    def dn_dA(self, A, steepness = 0.15):
        h = A/self.width
        dn_dh = Hydraulics.dn_dh(depth=h,
                                 steepness=steepness,
                                 channel_roughness=self.channel_roughness,
                                 floodplain_roughness=self.floodplain_roughness,
                                 bankful_depth=self.bankful_depth)
        dh_dA = 1. / self.width
        return dn_dh * dh_dA
    
    def set_intermediate_widths(self, widths: list, chainages: list):
        if not self.conditions_initialized:
            raise ValueError("Conditions must be initialized first.")
        
        if len(widths) != len(chainages):
            raise ValueError("")
        
        widths = [self.width] + widths + [self.width]
        chainages = [0] + chainages + [self.downstream_boundary.chainage]
        
        n_nodes = len(self.initial_conditions)
        
        interp_widths = []
        for i in range(n_nodes):
            x = self.length * float(i) / float(n_nodes-1)
            interp_widths.append(self.calc_B(x, chainages=chainages, widths=widths))
            
        self.width = interp_widths
        self.fixed_width = False
        
    def set_intermediate_bed_levels(self, bed_levels: list, chainages: list):
        if not self.conditions_initialized:
            raise ValueError("Conditions must be initialized first.")
        
        if len(bed_levels) != len(chainages):
            raise ValueError("")
        
        bed_levels = [self.upstream_boundary.bed_level] + bed_levels + [self.downstream_boundary.bed_level]
        chainages = [0] + chainages + [self.downstream_boundary.chainage]
        
        n_nodes = len(self.initial_conditions)
        
        interp_bed_slopes = []
        for i in range(n_nodes):
            x = self.length * float(i) / float(n_nodes-1)
            interp_bed_slopes.append(self.calc_S0(x, chainages=chainages, bed_levels=bed_levels))
            
        self.bed_slope = interp_bed_slopes
        self.fixed_bed_slope = False
        
    def calc_B(self, x, chainages, widths):
        if self.fixed_width:
            return self.width
                        
        from numpy import interp        
        return float(interp(x, chainages, widths))
    
    def calc_S0(self, x, chainages, bed_levels):
        if self.fixed_bed_slope:
            return self.bed_slope
        
        from numpy import array, gradient, interp
        chainages = array(chainages, dtype=float)
        levels = array(bed_levels, dtype=float)

        slopes = gradient(levels, chainages)
        return -float(interp(x, chainages, slopes))
    