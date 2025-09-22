from src.solver import Solver
from src.channel import Channel
from scipy.constants import g

class LaxSolver(Solver):
    """
    Implements the Lax-Friedrichs explicit finite difference scheme to numerically
    solve the Saint-Venant equations.
        
    """
    def __init__(self,
                 reach: Channel,
                 temporal_step: int | float,
                 spatial_step: int | float,
                 secondary_boundary_conditions: tuple = ('constant', 'constant'),
                 fit_spatial_step: bool = True):
        """
        Initializes the class.

        Parameters
        ----------
        channel : Channel
            The Channel object on which the simulation is performed.
        temporal_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
        secondary_boundary_conditions : tuple of str
            The approximation methods used at the two boundaries. Options: 'constant', 'mirror', 'linear'.
        fit_spatial_step : bool
            Wheter or not the size of the spatial step should be adjusted suit the total length of the reach.
            
        """
        super().__init__(reach, temporal_step, spatial_step, fit_spatial_step)
        
        self.secondary_BC = secondary_boundary_conditions        
        self.initialize_t0()
                
    def initialize_t0(self):
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the channel.

        Returns
        -------
        None.

        """
        for A, Q in self.channel.initial_conditions:
            self.A_previous.append(A)
            self.Q_previous.append(Q)

            self.A_current.append(A)
            self.Q_current.append(Q)

        self.append_result()

    def run(self, duration: int, auto: bool = False, verbose: int = 1):
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
        time = 0
        running = True
        
        while running:
            time += self.time_step
            if time > duration and not auto:
                running = False
                time -= self.time_step
                break
            
            if verbose >= 1:
                print('---------- Time = ' + str(time) + 's ----------')
                
            for i in range(self.number_of_nodes):
                self.compute_node(i, time)
                        
            self.check_cfl_all()
            
            self.append_result()
            self.update()
        
        super().finalize(time=time, verbose=verbose)
        
    def ghost_node(self):
        if self.secondary_BC[0] == 'constant':
            return (self.area_at(0, 0), self.flow_at(0, 0))
                                
        elif self.secondary_BC[0] == 'mirror':
            return (self.area_at(1, 0), self.flow_at(1, 0))
                                
        elif self.secondary_BC[0] == 'linear':
            return (2 * self.area_at(0, 0) - self.area_at(1, 0),
                    2 * self.flow_at(0, 0) - self.flow_at(1, 0))
        
    def compute_upstream_node(self, time):
        BC = self.channel.upstream_boundary.condition
        if BC == 'flow_hydrograph':
            self.Q_current[0] = self.channel.upstream_boundary.hydrograph.get_at(time)
        
            if self.secondary_BC[0] == 'rating_curve':
                stage = self.channel.upstream_boundary.rating_curve.stage(self.flow_at(0, 1))
                depth = stage - self.channel.upstream_boundary.bed_level
                self.A_current[0] = depth * self.width_at(0)        
            else:
                self.A_current[0] = self.new_area(self.ghost_node()[0],
                                                  self.A_previous[1],
                                                  self.ghost_node()[1],
                                                  self.Q_previous[1])
            return
                
        elif BC == 'stage_hydrograph':
            stage = self.channel.upstream_boundary.hydrograph.get_at(time)
            depth = stage - self.channel.upstream_boundary.bed_level
            self.A_current[0] = depth * self.width_at(0)
        
            if self.secondary_BC[0] == 'rating_curve':
                self.Q_current[0] = self.channel.upstream_boundary.rating_curve.discharge(stage)
            else:
                self.Q_current[0] = self.new_flow(self.ghost_node()[0],
                                                  self.A_previous[1],
                                                  self.ghost_node()[1],
                                                  self.Q_previous[1])
                
        elif BC == 'fixed_depth':
            depth = self.channel.upstream_boundary.storage_stage - self.channel.upstream_boundary.bed_level
            self.A_current[0] = depth * self.width_at(0)
            
            if self.secondary_BC[0] == 'rating_curve':
                self.Q_current[0] = self.channel.upstream_boundary.rating_curve.discharge(stage)
            else:
                self.Q_current[0] = self.new_flow(self.ghost_node()[0],
                                                  self.A_previous[1],
                                                  self.ghost_node()[1],
                                                  self.Q_previous[1])
            
        elif BC == 'normal_depth':        
            if self.secondary_BC[0] == 'rating_curve':
                self.Q_current[0] = self.channel.upstream_boundary.rating_curve.discharge(stage)
            else:
                self.Q_current[0] = self.new_flow(self.ghost_node()[0],
                                                  self.A_previous[1],
                                                  self.ghost_node()[1],
                                                  self.Q_previous[1])
                
            self.A_current[0] = self.channel.normal_area(self.flow_at(0, 1), 0)
        
        elif BC == 'rating_curve':
            if self.secondary_BC[0] == 'rating_curve':
                raise ValueError("Duplicate BC.")
            else:
                self.Q_current[0] = self.new_flow(self.ghost_node()[0],
                                                  self.A_previous[1],
                                                  self.ghost_node()[1],
                                                  self.Q_previous[1])
            
            self.Q_current[0] = self.channel.upstream_boundary.rating_curve.discharge(stage)

    def compute_node(self, i, time):
        if i == 0:
            self.compute_upstream_node(time)
            
        elif i == self.number_of_nodes - 1:
            self.compute_downstream_node()
            
        else:            
            self.A_current[i] = self.new_area(self.area_at(i-1, 0),
                                              self.area_at(i+1, 0),
                                              self.flow_at(i-1, 0),
                                              self.flow_at(i+1, 0))
                                                    
            self.Q_current[i] = self.new_flow(self.area_at(i-1, 0),
                                              self.area_at(i+1, 0),
                                              self.flow_at(i-1, 0),
                                              self.flow_at(i+1, 0))
    
    def compute_downstream_node(self):
        if self.secondary_BC[1] == 'constant':
            self.Q_current[-1] = self.new_flow(self.A_previous[-2],
                                                               self.A_previous[-1],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-1])
        elif self.secondary_BC[1] == 'mirror':
            self.Q_current[-1] = self.new_flow(self.A_previous[-2],
                                                               self.A_previous[-2],
                                                               self.Q_previous[-2],
                                                               self.Q_previous[-2])
        elif self.secondary_BC[1] == 'linear':
            self.Q_current[-1] = self.new_flow(self.A_previous[-2],
                                                               2 * self.A_previous[-1] - self.A_previous[-2],
                                                               self.Q_previous[-2],
                                                               2 * self.Q_previous[-1] - self.Q_previous[-2],)
        else:
            raise ValueError("Invalid secondary downstream boundary condition.")
        
        if self.channel.downstream_boundary.condition == 'fixed_depth':
            if self.active_storage:
                inflow = 0.5 * (self.Q_previous[-1] + self.Q_current[-1])
                new_stage = self.channel.downstream_boundary.mass_balance(inflow, self.time_step)
                self.channel.downstream_boundary.storage_stage = new_stage
                
            self.A_current[-1] = self.channel.width * (self.channel.downstream_boundary.storage_stage - self.channel.downstream_boundary.bed_level)
            
        elif self.channel.downstream_boundary.condition == 'normal_depth':
            self.A_current[-1] = self.channel.manning_A(self.Q_current[-1])
            
        elif self.channel.downstream_boundary.condition == 'rating_curve':
            stage = self.channel.downstream_boundary.initial_stage
            stage = self.channel.downstream_boundary.rating_curve.stage(self.Q_current[-1], stage)
            depth = stage - self.channel.downstream_boundary.bed_level
            
            self.A_current[-1] = depth * self.channel.width
            
        else:
            raise ValueError("Invalid downstream boundary condition.")

    def new_area(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        A = 0.5 * (A_i_plus_1 + A_i_minus_1) - (0.5 / self.num_celerity) * (Q_i_plus_1 - Q_i_minus_1)

        return A

    def new_flow(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        Sf_i_plus_1 = self.channel.friction_slope(A_i_plus_1, Q_i_plus_1)
        Sf_i_minus_1 = self.channel.friction_slope(A_i_minus_1, Q_i_minus_1)
        
        Q = (
             - g / (4 * self.channel.width * self.num_celerity) * (A_i_plus_1 ** 2 - A_i_minus_1 ** 2)
             + 0.5 * g * self.time_step * self.channel.bed_slope * (A_i_plus_1 + A_i_minus_1)
             + 0.5 * (Q_i_plus_1 + Q_i_minus_1)
             - 1. / (2 * self.num_celerity) * (Q_i_plus_1 ** 2 / A_i_plus_1 - Q_i_minus_1 ** 2 / A_i_minus_1)
             - 0.5 * g * self.time_step * (A_i_plus_1 * Sf_i_plus_1 + A_i_minus_1 * Sf_i_minus_1)
             )
        
        return Q
                
    def check_cfl_all(self):
        for i, (A, Q) in enumerate(zip(self.A_current, self.Q_current)):
            V = Q / A
            h = A / self.channel.width
            
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
        