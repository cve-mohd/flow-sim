import initial_conditions
import upstream_bc
import downstream_bc


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
        self.width = width
        self.length = length
        self.initial_conditions = []

    def friction_slope(self, A: float, Q: float) -> float:
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
        Sf = self.manning_co ** 2 * self.width ** (4. / 3) * Q ** 2 / A ** (10. / 3)

        return Sf

    @staticmethod
    def inflow_Q(time_seconds: float) -> float:
        """
        Computes the discharge at a given time using the upstream flow hydrograph.
        
        Parameters
        ----------
        time_seconds : float
            The time in seconds.
            
        Returns
        -------
        float
            The computed discharge in cubic meters per second.
            
        """
        return upstream_bc.inflow_hydrograph(time_seconds)

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
        return upstream_bc.rating_curve(water_depth)

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
        return downstream_bc.rating_curve(water_depth)

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
            y = (initial_conditions.us_y
                 + (initial_conditions.ds_y - initial_conditions.us_y) * i / (n_nodes - 1))

            Q = (initial_conditions.us_Q
                 + (initial_conditions.ds_Q - initial_conditions.us_Q) * i / (n_nodes - 1))

            A = y * self.width

            self.initial_conditions.append((A, Q))
