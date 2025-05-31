import numpy as np
from river import River
from scipy.constants import g
from utility import Utility
from copy import copy


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
        
        # Link the river object.
        self.river = river
        
        # Initialize the scheme parameters.
        self.delta_t, self.delta_x = delta_t, delta_x
        self.n_celerity = self.delta_x / float(self.delta_t)
        self.n_nodes = sum(self.river.length) // self.delta_x + 1

        # Declare empty lists for the flow variables at the previous time step, j.
        self.A_previous = []
        self.Q_previous = []

        # Declare empty lists for the flow variables at the advanced time step, j + 1.
        self.A_current = []
        self.Q_current = []

        # Declare empty lists to store the simulation results.
        self.results_A = []
        self.results_Q = []
        self.results_v = []
        self.results_h = []

        # Read the initial conditions of the river.
        self.initialize_t0()
        

    def initialize_t0(self):
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the river.

        Returns
        -------
        None.

        """
        
        # Compute the initial conditions at all nodes in the 'River' object.
        self.river.initialize_conditions(self.delta_x)

        # Read the values of A and Q from the 'River' object and assign
        # them to the lists of unknowns, as well as the lists of A and Q at
        # the previous (first) time step.
        for A, Q in self.river.initial_conditions:
            self.A_previous.append(A)
            self.Q_previous.append(Q)

            self.A_current.append(A)
            self.Q_current.append(Q)

        # Store the computed values of A and Q in the results list.
        self.append_result()


    def solve(self, duration: int):
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
            print('\n---------- Time = ' + str(time) + 's ----------')
            
            self.compute_upstream_boundary(time)
                
            for i in range(1, self.n_nodes - 1):
                self.compute_node(i)
                          
            self.compute_downstream_boundary()
                        
            self.check_cfl_all()
                
            self.append_result()
            self.update()
            

    def compute_upstream_boundary(self, time):
        self.Q_current[0] = self.river.inflow_Q(time)
        
        from settings import LAX_US_2ND_COND
        
        if LAX_US_2ND_COND == 'rating_curve':
            self.A_current[0] = self.upstream_A_given_Q(self.Q_current[0])
        
        elif LAX_US_2ND_COND == 'constant':
            self.A_current[0] = self.area_advanced_t(self.A_previous[0],
                                                               self.A_previous[1],
                                                               self.Q_previous[0],
                                                               self.Q_previous[1])
                
        elif LAX_US_2ND_COND == 'mirror':
            self.A_current[0] = self.area_advanced_t(self.A_previous[1],
                                                               self.A_previous[1],
                                                               self.Q_previous[1],
                                                               self.Q_previous[1])
                
        else:
            raise ValueError("Invalid approximation method. Choose either 'mirror' or 'constant.'")
        
        
    def upstream_A_given_Q(self, Q, tolerance = 1e-4):
        """Computes the upstream flow area for a given discharge from the rating curve
        using trial and error.

        Args:
            Q (float): The upstream discharge

        Returns:
            float: The computed flow area
        """        
        
        A = self.A_previous[0] # Initial guess
            
        trial_y = A / self.river.width[0]
        trial_Q = self.river.rating_curve_us(trial_y)
            
        while abs(trial_Q - Q) > tolerance:
            error = (trial_Q - Q) / Q
                
            trial_y -= 0.1 * error * trial_y
            A = trial_y * self.river.width[0]
                
            trial_Q = self.river.rating_curve_us(trial_y)
            
        return A
        

    def compute_node(self, i):
        self.A_current[i] = self.area_advanced_t(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

        self.Q_current[i] = self.discharge_advanced_t(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

    
    def compute_downstream_boundary(self):
        from settings import DS_CONDITION, LAX_DS_2ND_COND
        
        if LAX_DS_2ND_COND == 'constant':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               self.A_previous[-1],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-1])
                
        elif LAX_DS_2ND_COND == 'mirror':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               self.A_previous[-2],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-2])
                
        else:
            raise ValueError("Invalid approximation method. Choose either 'mirror' or 'constant.'")
        
        if DS_CONDITION == 'fixed_depth':
            self.A_current[-1] = self.A_previous[-1]
            
        elif DS_CONDITION == 'normal_depth':
            self.A_current[-1] = self.river.manning_A(-1, self.Q_current[-1], self.A_previous[-1], 1e-6)
        
        else:
            raise ValueError("Invalid Downstream boundary condition.")
        

    def area_advanced_t(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        A = 0.5 * (A_i_plus_1 + A_i_minus_1) - (0.5 / self.n_celerity) * (Q_i_plus_1 - Q_i_minus_1)

        return A


    def discharge_advanced_t(self, distance, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        Sf_i_plus_1 = self.river.friction_slope(distance, A_i_plus_1, Q_i_plus_1)
        Sf_i_minus_1 = self.river.friction_slope(distance, A_i_minus_1, Q_i_minus_1)
        
        r = self.river.reach_given_distance(distance)
        
        Q = (
             - g / (4 * self.river.width[r] * self.n_celerity) * (A_i_plus_1 ** 2 - A_i_minus_1 ** 2)
             + 0.5 * g * self.delta_t * self.river.bed_slope[r] * (A_i_plus_1 + A_i_minus_1)
             + 0.5 * (Q_i_plus_1 + Q_i_minus_1)
             - 1 / (2 * self.n_celerity) * (Q_i_plus_1 ** 2 / A_i_plus_1 - Q_i_minus_1 ** 2 / A_i_minus_1)
             - 0.5 * g * self.delta_t * (A_i_plus_1 * Sf_i_plus_1 + A_i_minus_1 * Sf_i_minus_1)
             )
        
        return Q
   
                
    def check_cfl_all(self):
        l = 0
        for A, Q in zip(self.A_current, self.Q_current):
            V = Q / A
            
            r = self.river.reach_given_distance(l)
            y = A / self.river.width[r]
            
            if self.check_cfl_condition(V, y) == False:
                analytical_celerity = max(V + (g * y) ** 0.5, V - (g * y) ** 0.5)
                raise ValueError('Courant condition is not satisfied. Analytical celerity = '
                                 + str(analytical_celerity) + ', Numerical celerity: ' + str(self.n_celerity))
                
            l += self.delta_x

    
    def check_cfl_condition(self, velocity, depth):
        analytical_celerity = max(velocity + (g * depth) ** 0.5, velocity - (g * depth) ** 0.5)
        return self.n_celerity >= analytical_celerity
    
    
    def append_result(self):            
        self.results_A.append(copy(self.A_current))
        self.results_Q.append(copy(self.Q_current))
        
        V = np.array(self.Q_current) / np.array(self.A_current)
        #y = np.array(self.A_current) / self.river.width
        
        self.results_v.append(V.tolist())
        #self.results_h.append(y.tolist())
    
    
    def update(self): 
        self.A_previous = copy(self.A_current)
        self.Q_previous = copy(self.Q_current)
        
        
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

        v = [v[::x_step]
                for v in self.results_v[::t_step]]
        
        #h = [h[::x_step]
                #for h in self.results_h[::t_step]]
        
        
        data = {
            'Area': A,
            'Discharge': Q,
            #'Depth': h,
            'Velocity': v
        }

        for key, value in data.items():
            value_str = str(value).replace('], [', '\n')
            for c in "[]' ":
                value_str = value_str.replace(c, '')
            with open(f'Results//Lax//{key}.csv', 'w') as output_file:
                output_file.write(value_str)
    