import numpy as np
from river import River
from scipy.constants import g
from utility import Utility


class LaxModel:
    """
    Implements the Preissmann implicit finite difference scheme to numerically
    solve the Saint-Venant equations.

    Attributes
    ----------
    river : River
        An instance of the `River` class, representing the river being modeled.
    delta_t : float
        Time step for the simulation in seconds.
    delta_x : float
        Spatial step for the simulation in meters.
    celerity : float
        Ratio of spatial to time step, representing the wave celerity.
    n_nodes : int
        Number of spatial nodes along the river.
    A_previous : list of float
        Cross-sectional areas at the previous time step.
    Q_previous : list of float
        Discharges at the previous time step.
    A_current : list of float
        Cross-sectional areas at the current iteration of the current time step.
    Q_current : list of float
        Discharges at the current iteration of the current time step.
    results_A : list of list of float
        Stores the computed A values over time.
    results_Q : list of list of float
        Stores the computed Q values over time.
    results_V : list of list of float
        Stores the computed V values over time.
    results_y : list of list of float
        Stores the computed y values over time.
    S_h : float
        Slope due to backwater effects.
        
    """
    
    def __init__(self,
                 river: River,
                 delta_t: int | float,
                 delta_x: int | float):
        """
        Initializes the class.

        Parameters
        ----------
        river : River
            The River object on which the simulation is performed.
        delta_t : float
            Time step for the simulation in seconds.
        delta_x : float
            Spatial step for the simulation in meters.
            
        """

        # Initialize the scheme discretization parameters.
        self.delta_t, self.delta_x = delta_t, delta_x
        self.celerity = self.delta_x / float(self.delta_t)

        # Inizialize the river attributes.
        self.river = river
        self.n_nodes = int(self.river.length / self.delta_x + 1)
        self.W = self.river.width
        self.n = self.river.manning_co
        self.S_0 = self.river.bed_slope
        self.S_h = None

        # Declare empty lists for the flow variables at the previous time step, j.
        self.A_previous = []
        self.Q_previous = []

        # Declare empty lists for the flow variables at the advanced time step, j + 1.
        self.A_current = []
        self.Q_current = []

        # Declare an empty list to store the simulation results.
        self.results_A = []
        self.results_Q = []
        self.results_V = []
        self.results_y = []

        # Read the initial conditions of the river.
        self.initialize_t0()
        
        # Compute the slope due to backwater effects.
        self.S_h = self.backwater_effects_calc()


    def initialize_t0(self):
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the river.

        Returns
        -------
        None.

        """
        
        # Compute the initial conditions at all nodes in the 'River' object.
        self.river.initialize_conditions(self.n_nodes)

        # Read the values of A and Q from the 'River' object and assign
        # them to the lists of unknowns, as well as the lists of A and Q at
        # the previous (first) time step.
        for A, Q in self.river.initial_conditions:
            self.A_previous.append(A)
            self.Q_previous.append(Q)

            self.A_current.append(0)
            self.Q_current.append(0)

        # Store the computed values of A and Q in the results list.
        self.results_A.append(self.A_previous)
        self.results_Q.append(self.Q_previous)


    def solve(self, duration: int, approximation: str = 'same'):
        """
        Solves the system of equations using the Lax explicit scheme and stores
        the obtained values of the flow variables.

        Parameters
        ----------
        duration : int
            The simulation duration in seconds.
        approximation : str
            The type of approximation to use for the downstream boundary condition.
            Can be 'same' or 'mirror'.

        Returns
        -------
        None.

        """

        # Loop through the time steps, incrementing the time by delta t every time.
        for time in range(self.delta_t, duration + self.delta_t, self.delta_t):
            print('\n---------- Time = ' + str(time) + 's ----------')
            
            self.computeUpstreamBoundary(time)
                
            for i in range(1, self.n_nodes - 1):
                self.computeNode(i)
                          
            self.computeDownstreamBoundary(approximation)
                        
            self.checkCourantAll()
                
            self.saveAndUpdate()
            

    def computeUpstreamBoundary(self, time):
        self.Q_current[0] = self.river.inflow_Q(time)
        self.A_current[0] = self.upstream_A_given_Q(self.Q_current[0])
        
        
    def upstream_A_given_Q(self, Q, tolerance = 1e-4):
        """Computes the upstream flow area for a given discharge from the rating curve
        using trial and error.

        Args:
            Q (float): The upstream discharge

        Returns:
            float: The computed flow area
        """        
        
        A = self.A_previous[0] # Initial guess
            
        trial_y = A / self.W
        trial_Q = self.river.rating_curve_us(trial_y)
            
        while abs(trial_Q - Q) > tolerance:
            error = (trial_Q - Q) / Q
                
            trial_y -= 0.1 * error * trial_y
            A = trial_y * self.W
                
            trial_Q = self.river.rating_curve_us(trial_y)
            
        return A
        

    def computeNode(self, i):
        self.A_current[i] = self.area_advanced_t(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

        self.Q_current[i] = self.discharge_advanced_t(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

    
    def computeDownstreamBoundary(self, approximation):
        self.A_current[-1] = self.A_previous[-1]

        if approximation == 'same':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               self.A_previous[-1],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-1])
                
        elif approximation == 'mirror':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               self.A_previous[-2],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-2])
                
        else:
            raise ValueError("Invalid approximation method. Choose either 'mirror' or 'same.'")
        

    def area_advanced_t(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        A = 0.5 * (A_i_plus_1 + A_i_minus_1) - (0.5 / self.celerity) * (Q_i_plus_1 - Q_i_minus_1)

        return A


    def discharge_advanced_t(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        Q = (
             - g / (4 * self.W * self.celerity) * (A_i_plus_1 ** 2 - A_i_minus_1 ** 2)
             + 0.5 * g * (self.S_0 - self.S_h) * self.delta_t * (A_i_plus_1 + A_i_minus_1)
             + 0.5 * (Q_i_plus_1 + Q_i_minus_1)
             - 1 / (2 * self.celerity) * (Q_i_plus_1 ** 2 / A_i_plus_1 - Q_i_minus_1 ** 2 / A_i_minus_1)
             - 0.5 * g * self.W ** (4./3) * self.n ** 2 * self.delta_t * (
                         Q_i_plus_1 ** 2 / A_i_plus_1 ** (7./3) + Q_i_minus_1 ** 2 / A_i_minus_1 ** (7./3))
             )
        
        return Q
   
   
    def backwater_effects_calc(self) -> float:
        """
        Computes the slope due to backwater effects.

        Returns
        -------
        None.

        """
        
        S_f = self.river.friction_slope(self.A_previous[0], self.Q_previous[0])
        return self.S_0 - S_f
             
                
    def checkCourantAll(self):
        for A, Q in zip(self.A_current, self.Q_current):
            V = Q / A
            y = A / self.W
            if self.checkCourantCondition(V, y) == False:
                raise ValueError('Courant condition is not satisfied. Velocity: ' + str(V) + ', Depth: ' + str(y))

    
    def checkCourantCondition(self, velocity, depth):
        return (velocity + (g * depth) ** 0.5) / self.celerity <= 1
    
    
    def saveAndUpdate(self):
        self.results_A.append(self.A_current)
        self.results_Q.append(self.Q_current)
        
        V = np.array(self.Q_current) / np.array(self.A_current)
        y = np.array(self.A_current) / self.W
        
        self.results_V.append(V.tolist())
        self.results_y.append(y.tolist())        
        
        self.A_previous = [a for a in self.A_current]
        self.Q_previous = [q for q in self.Q_current]
        
        
    def save_results(self, size: tuple) -> None:
        """
        Saves the results of the simulation in four .csv files, containing
        the computed cross-sectional flow area, discharge, flow depth, and velocity.
        The files are formatted with each row representing a time step and each
        column representing a spatial point.

        Parameters
        ----------
        size : tuple of int
            The number of time steps and spatial steps to save.

        Returns
        -------
        None.

        """
        
        Utility.create_directory_if_not_exists('Results')
        Utility.create_directory_if_not_exists('Results//Lax')
                
        t_step = x_step = 1
        
        if size[0] > 1:
            t_step = (len(self.results_A) - 1) // (size[0] - 1)

        if size[1] > 1:
            x_step =  (self.n_nodes - 1) // (size[1] - 1)

        A = [a[::x_step]
                for a in self.results_A[::t_step]]

        Q = [q[::x_step]
                for q in self.results_Q[::t_step]]

        V = [v[::x_step]
                for v in self.results_V[::t_step]]
        
        y = [y[::x_step]
                for y in self.results_y[::t_step]]

        data = {
            'Area': A,
            'Discharge': Q,
            'Depth': y,
            'Velocity': V
        }

        for key, value in data.items():
            value_str = str(value).replace('], [', '\n')
            for c in "[]' ":
                value_str = value_str.replace(c, '')
            with open(f'Results//Lax//{key}.csv', 'w') as output_file:
                output_file.write(value_str)
    