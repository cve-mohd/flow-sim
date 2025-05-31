import boundary
from settings import REACHES, APPROX_R, BED_SLOPE_CORRECTION


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

    def __init__(self, length: list[float], width: list[float], bed_slope: list[float], manning_co: list[float]):
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
        self.width = [float(i) for i in width]
        self.bed_slope = bed_slope
        self.manning_co = manning_co
        
        self.initial_conditions = []
        
        if BED_SLOPE_CORRECTION:
            A = boundary.Upstream.initial_depth * self.width[0]
            Q = boundary.Upstream.initial_discharge
            self.bed_slope[0] = self.friction_slope(0, A, Q)
        

    def friction_slope(self, distance: float, A: float, Q: float, approx_R = APPROX_R) -> float:
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
        reach_index = self.reach_given_distance(distance)
        
        if approx_R:
            P = self.width[reach_index]
        else:
            P = self.width[reach_index] + 2 * A / self.width[reach_index]
            
        Sf = (self.manning_co[reach_index] * P ** (2. / 3) * Q / A ** (5. / 3)) ** 2
        
        return Sf
    
    def friction_slope_deriv_A(self, distance: float, A: float, Q: float, approx_R = APPROX_R) -> float:
        reach_index = self.reach_given_distance(distance)
        
        if approx_R:
            d_Sf = -10./3 * Q ** 2 * self.manning_co[reach_index] ** 2 * self.width[reach_index] ** (4./3) * A ** (-13./3)
        else:
            d_Sf = -10./3 * Q ** 2 * self.manning_co[reach_index] ** 2 * (self.width[reach_index] + 2 * A / self.width[reach_index]) ** (4./3) * A ** (-13./3)
            + Q ** 2 * self.manning_co[reach_index] ** 2 * 4. / 3 * (self.width[reach_index] + 2 * A / self.width[reach_index]) ** (1./3) * A ** (-10./3) * 2 / self.width[reach_index]

        return d_Sf
    
    def friction_slope_deriv_Q(self, distance: float, A: float, Q: float, approx_R = APPROX_R) -> float:
        reach_index = self.reach_given_distance(distance)
        
        if approx_R:
            P = self.width[reach_index]
        else:
            P = self.width[reach_index] + 2 * A / self.width[reach_index]
            
        d_Sf = 2 * Q * self.manning_co[reach_index] ** 2 * P ** (4./3) * A ** (-10./3)
        
        return d_Sf

    @staticmethod
    def inflow_Q(time: float) -> float:
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
        return boundary.Upstream.inflow_hydrograph(time)

    @staticmethod
    def rating_curve_us(water_depth: float) -> float:
        """
        Computes the discharge for a given water depth using
        the rating curve equation of the upstream boundary.

        Parameters
        ----------
        water_depth : float
            The water depth at the upstream boundary in meters.

        Returns
        -------
        Q : float
            The computed discharge in cubic meters per second.

        """
        return boundary.Upstream.rating_curve(water_depth)

    @staticmethod
    def rating_curve_ds(water_depth: float) -> float:
        """
        Computes the discharge for a given water depth using
        the rating curve equation of the downstream boundary.

        Parameters
        ----------
        water_depth : float
            The water depth at the downstream boundary in meters.

        Returns
        -------
        Q : float
            The computed discharge in cubic meters per second.

        """
        return boundary.Downstream.rating_curve(water_depth)
    
    
    def manning_Q(self, distance: float, A, slope = None):
        reach_index = self.reach_given_distance(distance)
        
        S = self.bed_slope[reach_index]
        if slope is not None:
            S = slope
            
        P = self.width[reach_index] + 2. * A / self.width[reach_index]
        Q = A ** (5./3) * S ** 0.5 / (self.manning_co[reach_index] * P ** (2./3))
        return Q
    
    
    def manning_A(self, distance, Q, A_guess, tolerance, slope = None):
        reach_index = self.reach_given_distance(distance)
        
        S = self.bed_slope[reach_index]
        if slope is not None:
            S = slope
            
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
        
        n_nodes = sum(self.length) // delta_x + 1
        
        for i in range(n_nodes):
            y = (boundary.Upstream.initial_depth
                 + (boundary.Downstream.initial_depth - boundary.Upstream.initial_depth) * i / float(n_nodes - 1))

            Q = (boundary.Upstream.initial_discharge
                 + (boundary.Downstream.initial_discharge - boundary.Upstream.initial_discharge) * i / float(n_nodes - 1))

            reach_index = self.reach_given_distance(i * delta_x)

            A = y * self.width[reach_index]
            
            self.initial_conditions.append((A, Q))


    def reach_given_distance(self, distance):
        if distance == -1:
            return REACHES - 1
        
        d = distance
        for i in range(len(self.length)):
            if d <= self.length[i]:
                return i
            else:
                d -= self.length[i]
                
        return -1