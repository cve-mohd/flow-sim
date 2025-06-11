from boundary import Boundary

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

    def __init__(self, length: float, width: float, initial_flow_rate: float, manning_co: float,
                 upstream_boundary: Boundary, downstream_boundary: Boundary, bed_slope: float | str = 'real', flow_type: str = 'varied', buffer_length: float = 0):
        """
        Initialized an instance.

        Parameters
        ----------
        bed_slope : any
            A value, 'normal', or 'real'.
        manning_co : float
            The river's Manning coefficient or coefficient of roughness.
        width : float
            The width of the river cross-section.
        length : float
            The length of the river.

        """
        self.real_length = length
        self.buffer_length = buffer_length
        self.total_length = self.real_length + self.buffer_length
        
        self.width = float(width)
        self.initial_flow_rate = initial_flow_rate
        self.manning_co = manning_co
        
        self.upstream_boundary = upstream_boundary
        self.downstream_boundary = downstream_boundary
        
        if flow_type in ['varied', 'uniform']:
            self.flow_type = flow_type
        
        self.initial_conditions = []
        
        if isinstance(bed_slope, float):
            self.bed_slope = bed_slope
        
        elif bed_slope == 'normal':
            A = upstream_boundary.initial_depth * self.width
            Q = self.initial_flow_rate
            self.bed_slope = self.friction_slope(A, Q, False)
            
        elif bed_slope == 'real':
            y1 = self.upstream_boundary.bed_level
            y2 = self.downstream_boundary.bed_level
            self.bed_slope = float(y1 - y2) / float(self.real_length)
            
        else:
            raise ValueError("Invalid bed slope argument.")
        

    def friction_slope(self, A: float, Q: float, approx_R = False) -> float:
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
        if approx_R:
            P = self.width
        else:
            P = self.width + 2 * A / self.width
            
        Sf = (self.manning_co * P ** (2. / 3) * Q / A ** (5. / 3)) ** 2
                
        return Sf
    
    def friction_slope_deriv_A(self, A: float, Q: float, approx_R = False) -> float:
        if approx_R:
            d_Sf = -10./3 * Q ** 2 * self.manning_co ** 2 * self.width ** (4./3) * A ** (-13./3)
        else:
            d_Sf = -10./3 * Q ** 2 * self.manning_co ** 2 * (self.width + 2 * A / self.width) ** (4./3) * A ** (-13./3)
            + Q ** 2 * self.manning_co ** 2 * 4. / 3 * (self.width + 2 * A / self.width) ** (1./3) * A ** (-10./3) * 2 / self.width

        return d_Sf
    
    def friction_slope_deriv_Q(self, A: float, Q: float, approx_R = False) -> float:
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
    
    def manning_Q_deriv(self, A, slope = None):
        pass
    
    def manning_A(self, flow_rate, tolerance = 1e-3, slope = None):
        if slope is not None:
            S = slope
        else:
            S = self.bed_slope
            
        trial_A = self.width * self.upstream_boundary.initial_depth
        q = self.manning_Q(trial_A, S)
            
        while abs(q - flow_rate) >= tolerance:
            error = (q - flow_rate) / flow_rate
            trial_A -= 0.1 * error * trial_A
            q = self.manning_Q(trial_A, S)
            
        return trial_A
        
        
    def initialize_conditions(self, n_nodes: int) -> None:
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
        if self.flow_type == 'uniform':
            y_0 = self.upstream_boundary.initial_depth
            y_n = self.downstream_boundary.initial_depth
                    
            for i in range(n_nodes):
                distance = self.total_length * float(i) / float(n_nodes-1)
                
                if distance <= self.real_length:
                    y = (y_0 + (y_n - y_0) * distance / self.real_length)
                    
                else:
                    y = y_n

                A, Q = y * self.width, self.initial_flow_rate
                self.initial_conditions.append((A, Q))
                
        elif self.flow_type == 'varied':
            from scipy.constants import g

            dx = self.total_length / (n_nodes - 1)
            h = self.downstream_boundary.initial_depth

            for i in reversed(range(n_nodes)):
                distance = i * dx

                if distance > self.real_length:
                    A, Q = self.downstream_boundary.initial_depth * self.width, self.initial_flow_rate
                                
                else:
                    A = self.width * h
                    Sf = self.friction_slope(A, self.initial_flow_rate)
                    Fr2 = self.initial_flow_rate**2 / (g * A**3 / self.width)

                    denominator = 1 - Fr2
                    if abs(denominator) < 1e-6:
                        dhdx = 0.0
                    else:
                        dhdx = (self.bed_slope - Sf) / denominator

                    if i < n_nodes - 1:
                        h -= dhdx * dx

                    if h < 0:
                        raise ValueError("GVH failed.")

                    A, Q = h * self.width, self.initial_flow_rate
                    
                self.initial_conditions.insert(0, (A, Q))
                
        else:
            raise ValueError("Invalid flow type.")
            

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
        