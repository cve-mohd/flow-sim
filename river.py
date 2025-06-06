import boundary
from settings import APPROX_R, BED_SLOPE_CORRECTION


class River:
    """
    Represents a river with hydraulic and geometric attributes.
    
    Attributes
    ----------
    bed_slope : float
        The slope of the river bed.
    manning_co : float
        The river's Manning coefficient or coefficient of roughness.
    width : float
        The width of the river cross-section.
    length : float
        The length of the river segment to be analyzed.
    initial_conditions : list of tuple of float.
        A list of tuples, where each tuple stores the initial values of the
        flow variables (A and Q) at the spatial point corresponding to its index.
        
    """

    def __init__(self, length: float, width: float, bed_slope: float, manning_co: float,
                 upstream_boundary: boundary.Boundary, downstream_boundary: boundary.Boundary):
        """
        Initialized an instance.

        Parameters
        ----------
        bed_slope : float
            The slope of the river bed.
        manning_co : float
            The river's Manning coefficient or coefficient of roughness.
        width : float
            The width of the river cross-section.
        length : float
            The length of the river.

        """
        self.length = length
        self.width = float(width)
        self.bed_slope = bed_slope
        self.manning_co = manning_co
        
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary
        
        self.initial_conditions = []
        
        if BED_SLOPE_CORRECTION:
            A = upstream_boundary.initial_depth * self.width
            Q = downstream_boundary.initial_discharge
            self.bed_slope = self.friction_slope(A, Q)
        

    def friction_slope(self, A: float, Q: float, approx_R = APPROX_R) -> float:
        """
        Computes the friction slope using Manning's equation.
        
        Parameters
        ----------
        A : float
            The cross-sectional flow area.
        Q : float
            The discharge.
        W : float
            The channel width.
            
        Returns
        -------
        float
            The computed friction slope.
            
        """       
        if approx_R:
            P = self.width
        else:
            P = self.width + 2 * A / self.width
            
        Sf = (self.manning_co * P ** (2. / 3) * Q / A ** (5. / 3)) ** 2
        
        return Sf
    
    def friction_slope_deriv_A(self, A: float, Q: float, approx_R = APPROX_R) -> float:
        if approx_R:
            d_Sf = -10./3 * Q ** 2 * self.manning_co ** 2 * self.width ** (4./3) * A ** (-13./3)
        else:
            d_Sf = -10./3 * Q ** 2 * self.manning_co ** 2 * (self.width + 2 * A / self.width) ** (4./3) * A ** (-13./3)
            + Q ** 2 * self.manning_co ** 2 * 4. / 3 * (self.width + 2 * A / self.width) ** (1./3) * A ** (-10./3) * 2 / self.width

        return d_Sf
    
    def friction_slope_deriv_Q(self, A: float, Q: float, approx_R = APPROX_R) -> float:
        if approx_R:
            P = self.width
        else:
            P = self.width + 2 * A / self.width
            
        d_Sf = 2 * Q * self.manning_co ** 2 * P ** (4./3) * A ** (-10./3)
        
        return d_Sf


    def inflow_Q(self, time: float) -> float:
        """
        Computes the discharge at a given time using the upstream flow hydrograph.
        
        Parameters
        ----------
        time : float
            The time in seconds.
            
        Returns
        -------
        float
            The computed discharge in cubic meters per second.
            
        """
        return self.upstream_boundary.hydrograph_Q(time)


    def manning_Q(self, A, slope = None):
        S = self.bed_slope
        if slope is not None:
            S = slope
            
        P = self.width + 2. * A / self.width
        Q = A ** (5./3) * S ** 0.5 / (self.manning_co * P ** (2./3))
        return Q
    
    
    def manning_A(self, Q, A_guess, tolerance, slope = None):
        if slope is not None:
            S = slope
        else:
            S = self.bed_slope
            
        trial_A = A_guess
        trial_Q = self.manning_Q(trial_A, S)
            
        while abs(trial_Q - Q) >= tolerance:
            error = (trial_Q - Q) / Q
            trial_A -= 0.1 * error * trial_A
            trial_Q = self.manning_Q(trial_A, S)
            
        return trial_A
        
        
    def initialize_conditions(self, delta_x: float) -> None:
        """
        Computes the initial conditions.
        Computes the initial values of the flow variables at each node
        using the initial boundary values.
        
        Parameters
        ----------
        n_nodes : int
            The number of spatial nodes along the river, including the two boundaries.
            
        Returns
        -------
        None.

        """
        
        n_nodes = self.length // delta_x + 1
        
        for i in range(n_nodes):
            y = (boundary.Upstream.initial_depth
                 + (boundary.Downstream.initial_depth - boundary.Upstream.initial_depth) * i / float(n_nodes - 1))

            Q = (boundary.Upstream.initial_discharge
                 + (boundary.Downstream.initial_discharge - boundary.Upstream.initial_discharge) * i / float(n_nodes - 1))

            A = y * self.width
            
            self.initial_conditions.append((A, Q))


    def upstream_bc(self, time = None, depth = None, discharge = None):
        return self.upstream_boundary.condition_residual(time, depth, self.width, discharge, self.bed_slope, self.manning_co)
        
    def downstream_bc(self, time = None, depth = None, discharge = None):
        return self.downstream_boundary.condition_residual(time, depth, self.width, discharge, self.bed_slope, self.manning_co)
        
    def upstream_bc_deriv_A(self, time = None, area = None):
        return self.upstream_boundary.condition_derivative_wrt_A(time, area, self.width, self.bed_slope, self.manning_co)
        
    def upstream_bc_deriv_Q(self):
        return self.upstream_boundary.condition_derivative_wrt_Q()
        
    def downstream_bc_deriv_A(self, time = None, area = None):
        return self.downstream_boundary.condition_derivative_wrt_A(time, area, self.width, self.bed_slope, self.manning_co)
        
    def downstream_bc_deriv_Q(self):
        return self.downstream_boundary.condition_derivative_wrt_Q()
        