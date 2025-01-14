import numpy as np
from river import River
from scipy.constants import g


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

        # Read the initial conditions of the river.
        self.initialize_t0()
        
        # Compute the slope due to backwater effects.
        self.backwater_effects_calc()


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


    def solve(self, duration: int, approximation: str = 'zeroslope'):
        """
        Solves the system of equations using the Lax explicit scheme and stores
        the obtained values of the flow variables.

        Parameters
        ----------
        duration : int
            The simulation duration in seconds.

        Returns
        -------
        None.

        """

        # Loop through the time steps, incrementing the time by delta t every time.
        for time in range(self.delta_t, duration + self.delta_t, self.delta_t):
            print('\n---------- Time = ' + str(time) + 's ----------\n')
            
            self.Q_current[0] = self.river.inflow_Q(time)
                       
            self.A_current[0] = self.A_previous[0]
            
            trial_y = self.A_previous[0] / self.W
            trial_Q = self.river.rating_curve_us(trial_y)
            
            while abs(trial_Q - self.Q_current[0]) > 1e-4:
                trial_y -= 0.1 * (trial_Q - self.Q_current[0]) / self.A_current[0]
                self.A_current[0] = trial_y * self.W
                trial_Q = self.river.rating_curve_us(trial_y)
                
            V = self.Q_current[0] / self.A_current[0]
            y = self.A_current[0] / self.W
            if self.checkCourantCondition(V, y) == False:
                raise ValueError('Courant condition is not satisfied. Velocity: ' + str(V) + ', Depth: ' + str(y))
                
            for i in range(1, self.n_nodes - 1):
                self.A_current[i] = self.area_advanced_t(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

                self.Q_current[i] = self.discharge_advanced_t(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])
                
                V = self.Q_current[i] / self.A_current[i]
                y = self.A_current[i] / self.W
                if self.checkCourantCondition(V, y) == False:
                    raise ValueError('Courant condition is not satisfied. Velocity: ' + str(V) + ', Depth: ' + str(y))
                

            self.A_current[-1] = self.A_previous[-1]

            if approximation == 'zeroslope':
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
                raise ValueError('Courant condition is not satisfied. Velocity: ' + str(V) + ', Depth: ' + str(y))
            
            V = self.Q_current[0] / self.A_current[0]
            y = self.A_current[0] / self.W
            if self.checkCourantCondition(V, y) == False:
                raise ValueError('Courant condition is not satisfied. Velocity: ' + str(V) + ', Depth: ' + str(y))

            self.A_previous = [a for a in self.A_current]
            self.Q_previous = [q for q in self.Q_current]

            self.results_A.append(self.A_previous)
            self.results_Q.append(self.Q_previous)
            
        self.results_V = np.array(self.results_Q) / np.array(self.results_A)
        self.results_y = np.array(self.results_A) / self.W
        
        self.results_V = self.results_V.tolist()
        self.results_y = self.results_y.tolist()

    
    def area_advanced_t(self, A_i_minus_1: float, A_i_plus_1: float, Q_i_minus_1: float, Q_i_plus_1: float) -> float:
        """
        Computes the cross-sectional flow area at the advanced time step
        using the Lax explicit scheme.

        Parameters
        ----------
        A_i_minus_1 : float
            The cross-sectional flow area of the (i-1) node.
        A_i_plus_1 : float
            The cross-sectional flow area of the (i+1) node.
        Q_i_minus_1 : float
            The discharge of the (i-1) node.
        Q_i_plus_1 : float
            The discharge of the (i+1) node.

        Returns
        -------
        float
            The computed cross-sectional flow area.

        """
        
        A = 0.5 * (A_i_plus_1 + A_i_minus_1) - (0.5 / self.celerity) * (Q_i_plus_1 - Q_i_minus_1)

        #print(A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1, A)
        
        return A


    def discharge_advanced_t(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        """
        Computes the discharge at the advanced time step
        using the Lax explicit scheme.

        Parameters
        ----------
        A_i_minus_1 : float
            The cross-sectional flow area of the (i-1) node.
        A_i_plus_1 : float
            The cross-sectional flow area of the (i+1) node.
        Q_i_minus_1 : float
            The discharge of the (i-1) node.
        Q_i_plus_1 : float
            The discharge of the (i+1) node.

        Returns
        -------
        float
            The computed discharge.

        """

        Q = (
             - g / (4 * self.W * self.celerity) * (A_i_plus_1 ** 2 - A_i_minus_1 ** 2)
             + 0.5 * g * (self.S_0 - self.S_h) * self.delta_t * (A_i_plus_1 + A_i_minus_1)
             + 0.5 * (Q_i_plus_1 + Q_i_minus_1)
             - 1 / (2 * self.celerity) * (Q_i_plus_1 ** 2 / A_i_plus_1 - Q_i_minus_1 ** 2 / A_i_minus_1)
             - 0.5 * g * self.W ** (4./3) * self.n ** 2 * self.delta_t * (
                         Q_i_plus_1 ** 2 / A_i_plus_1 ** (7./3) + Q_i_minus_1 ** 2 / A_i_minus_1 ** (7./3))
             )
        
        return Q
   
    
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
            with open(f'Results//{key}.csv', 'w') as output_file:
                output_file.write(value_str)


    def backwater_effects_calc(self) -> None:
            """
            Computes the slope due to backwater effects.

            Returns
            -------
            None.

            """
            S_f = self.river.friction_slope(self.A_previous[-1], self.Q_previous[-1])
            self.S_h = self.S_0 - S_f
            
            
    def checkCourantCondition(self, velocity, depth):
        """
        Checks if the Courant condition is satisfied for the current simulation.

        Returns
        -------
        bool
            True if the Courant condition is satisfied, False otherwise.

        """
        
        return (velocity + (g * depth) ** 0.5) / self.celerity <= 1
    
    
"""
def quadratic_extrapolation(indices, values, target_index):
    # Ax = B
    A = [np.array(indices) ** i for i in range(len(indices) - 1, 0, -1)]

    A.append(np.ones(len(indices)))
    A = np.vstack(A).T
    B = np.array(values)

    coeffs = np.linalg.solve(A, B)

    target_value = 0
    for i in range(len(indices)):
        target_value += coeffs[i] * target_index ** (len(indices) - i - 1)

    return target_value
"""
