import boundary
from settings import APPROX_R


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

    def __init__(self, bed_slope: float, manning_co: float, width: float, length: float):
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
        self.bed_slope = bed_slope
        self.manning_co = manning_co
        self.width = float(width)
        self.length = length
        self.initial_conditions = []

    def friction_slope(self, A: float, Q: float, approx_R = APPROX_R) -> float:
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
        for i in range(n_nodes):
            y = (boundary.Upstream.initial_depth
                 + (boundary.Downstream.initial_depth - boundary.Upstream.initial_depth) * i / float(n_nodes - 1))

            Q = (boundary.Upstream.initial_discharge
                 + (boundary.Downstream.initial_discharge - boundary.Upstream.initial_discharge) * i / float(n_nodes - 1))

            A = y * self.width

            self.initial_conditions.append((A, Q))
