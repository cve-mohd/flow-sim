from solver import Solver
from river import River
from scipy.constants import g
from utility import Utility


class LaxSolver(Solver):
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
                 temporal_step: int | float,
                 spatial_step: int | float,
                 secondary_boundary_conditions: tuple = ('constant', 'constant'),
                 fit_spatial_step: bool = True):
        """
        Initializes the class.

        Parameters
        ----------
        river : River
            The River object on which the simulation is performed.
        temporal_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
            
        """
        super().__init__(river, temporal_step, spatial_step, fit_spatial_step)
        
        self.secondary_boundary_conditions = secondary_boundary_conditions
        
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
        self.river.initialize_conditions(self.number_of_nodes)

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


    def run(self, duration: int, verbose: int = 1):
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
        for time in range(self.time_step, duration + self.time_step, self.time_step):
            if verbose >= 1:
                print('\n---------- Time = ' + str(time) + 's ----------')
            
            self.compute_upstream_boundary(time)
                
            for i in range(1, self.number_of_nodes - 1):
                self.compute_node(i)
                          
            self.compute_downstream_boundary()
                        
            self.check_cfl_all()
                
            self.append_result()
            self.update()
            
        self.solved = True
        self.total_sim_duration = duration
            

    def compute_upstream_boundary(self, time):
        self.Q_current[0] = self.river.inflow_Q(time)
        
        if self.secondary_boundary_conditions[0] == 'rating_curve':
            stage = self.river.upstream_boundary.rating_curve.stage(self.Q_current[0])
            depth = stage - self.river.upstream_boundary.bed_level
            self.A_current[0] = depth * self.river.width
        
        elif self.secondary_boundary_conditions[0] == 'constant':
            self.A_current[0] = self.area_advanced_t(self.A_previous[0],
                                                        self.A_previous[1],
                                                        self.Q_previous[0],
                                                        self.Q_previous[1])
                
        elif self.secondary_boundary_conditions[0] == 'mirror':
            self.A_current[0] = self.area_advanced_t(self.A_previous[1],
                                                        self.A_previous[1],
                                                        self.Q_previous[1],
                                                        self.Q_previous[1])
            
        elif self.secondary_boundary_conditions[0] == 'extrapolate':
            self.A_current[0] = self.area_advanced_t(2 * self.A_previous[0] - self.A_previous[1],
                                                        self.A_previous[1],
                                                        2 * self.Q_previous[0] - self.Q_previous[1],
                                                        self.Q_previous[1])
                
        else:
            raise ValueError("Invalid secondary upstream boundary condition.")
               

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
        if self.secondary_boundary_conditions[1] == 'constant':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               self.A_previous[-1],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-1])
        elif self.secondary_boundary_conditions[1] == 'mirror':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               self.A_previous[-2],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-2])
        elif self.secondary_boundary_conditions[1] == 'extrapolate':
            self.Q_current[-1] = self.discharge_advanced_t(self.A_previous[-2],
                                                               2 * self.A_previous[-1] - self.A_previous[-2],
                                                               self.Q_previous[-2],
                                                               2 * self.Q_previous[-1] - self.Q_previous[-2],)
        else:
            raise ValueError("Invalid secondary downstream boundary condition.")
        
        if self.river.downstream_boundary.condition == 'fixed_depth':
            if self.active_storage:
                inflow = 0.5 * (self.Q_previous[-1] + self.Q_current[-1])
                new_stage = self.river.downstream_boundary.mass_balance(inflow, self.time_step)
                self.river.downstream_boundary.storage_stage = new_stage
                
            self.A_current[-1] = self.river.width * (self.river.downstream_boundary.storage_stage - self.river.downstream_boundary.bed_level)
            
        elif self.river.downstream_boundary.condition == 'normal_depth':
            self.A_current[-1] = self.river.manning_A(self.Q_current[-1])
            
        elif self.river.downstream_boundary.condition == 'rating_curve':
            stage = self.river.downstream_boundary.initial_stage
            stage = self.river.downstream_boundary.rating_curve.stage(self.Q_current[-1], stage)
            depth = stage - self.river.downstream_boundary.bed_level
            
            self.A_current[-1] = depth * self.river.width
            
        else:
            raise ValueError("Invalid downstream boundary condition.")
        

    def area_advanced_t(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        A = 0.5 * (A_i_plus_1 + A_i_minus_1) - (0.5 / self.num_celerity) * (Q_i_plus_1 - Q_i_minus_1)

        return A


    def discharge_advanced_t(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        Sf_i_plus_1 = self.river.friction_slope(A_i_plus_1, Q_i_plus_1)
        Sf_i_minus_1 = self.river.friction_slope(A_i_minus_1, Q_i_minus_1)
        
        Q = (
             - g / (4 * self.river.width * self.num_celerity) * (A_i_plus_1 ** 2 - A_i_minus_1 ** 2)
             + 0.5 * g * self.time_step * self.river.bed_slope * (A_i_plus_1 + A_i_minus_1)
             + 0.5 * (Q_i_plus_1 + Q_i_minus_1)
             - 1. / (2 * self.num_celerity) * (Q_i_plus_1 ** 2 / A_i_plus_1 - Q_i_minus_1 ** 2 / A_i_minus_1)
             - 0.5 * g * self.time_step * (A_i_plus_1 * Sf_i_plus_1 + A_i_minus_1 * Sf_i_minus_1)
             )
        
        return Q
   
                
    def check_cfl_all(self):
        for i, (A, Q) in enumerate(zip(self.A_current, self.Q_current)):
            V = Q / A
            h = A / self.river.width
            
            if self.check_cfl_condition(V, h) == False:
                analytical_celerity = max(V + (g * h) ** 0.5, V - (g * h) ** 0.5)
                raise ValueError(f'CFL condition is not satisfied.\nSpatial node = {i} (of {self.number_of_nodes-1})\nArea = {A}\nFlow rate = {Q}\n'
                                 + f'CFL number = {analytical_celerity/self.num_celerity}')

    
    def check_cfl_condition(self, velocity, depth):
        analytical_celerity = max(velocity + (g * depth) ** 0.5, velocity - (g * depth) ** 0.5)
        return self.num_celerity >= analytical_celerity
    
    
    def update(self):
        from copy import copy
        self.A_previous = copy(self.A_current)
        self.Q_previous = copy(self.Q_current)
        
        
    def save_results(self, size: tuple = (-1, -1), path: str = 'Results//Lax') -> None:
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
        from numpy import array
        
        Utility.create_directory_if_not_exists(path)
                
        t_step = x_step = 1
        
        if size[0] > 1:
            t_step = (len(self.computed_areas) - 1) // (size[0] - 1)

        if size[1] > 1:
            x_step =  (self.number_of_nodes - 1) // (size[1] - 1)

        areas = [a[::x_step]
                for a in self.computed_areas[::t_step]]
        
        flow_rates = [q[::x_step]
                for q in self.computed_flow_rates[::t_step]]

        # Compute velocities
        velocities = array(self.computed_flow_rates) / array(self.computed_areas)
        velocities = velocities.tolist()
        velocities = [v[::x_step]
                for v in velocities[::t_step]]
        
        # Compute depths and levels
        depths = array(self.computed_areas) / self.river.width
        depths = depths.tolist()
        
        levels = []
        for sublist in depths:
            levels.append([sublist[i] + self.river.upstream_boundary.bed_level - self.river.bed_slope * self.spatial_step * i for i in range(len(sublist))])
            
        levels = [levels[::x_step]
                for levels in levels[::t_step]]
        
        depths = [h[::x_step]
                for h in depths[::t_step]]
        
        
        data = {
            'Area': areas,
            'Discharge': flow_rates,
            'Depth': depths,
            'Velocity': velocities,
            'Level': levels
        }
        
        header = [self.river.upstream_boundary.chainage + x * self.spatial_step for x in range(0, self.number_of_nodes, x_step)]
        header = str(header)
        for c in "[]' ":
            header = header.replace(c, '')
            
        header += '\n'

        for key, value in data.items():
            value_str = str(value).replace('], [', '\n')
            for c in "[]' ":
                value_str = value_str.replace(c, '')
            with open(path + f'//{key}.csv', 'w') as output_file:
                output_file.write(header)
                output_file.write(value_str)
    