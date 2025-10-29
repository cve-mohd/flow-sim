from .solver import Solver
from .channel import Channel
from scipy.constants import g

class LaxSolver(Solver):
    """
    Implements the Lax-Friedrichs explicit finite difference scheme to numerically
    solve the Saint-Venant equations.
        
    """
    def __init__(self,
                 channel: Channel,
                 time_step: int | float,
                 spatial_step: int | float,
                 simulation_time: int,
                 secondary_BC: tuple = ('constant', 'constant'),
                 fit_spatial_step: bool = True,
                 regularization: bool = False):
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
        super().__init__(channel=channel,
                         time_step=time_step,
                         spatial_step=spatial_step,
                         simulation_time=simulation_time,
                         fit_spatial_step=fit_spatial_step,
                         regularization=regularization)
        
        self.secondary_BC = secondary_BC
        
        self.initialize_t0()
                
    def initialize_t0(self):
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the channel.

        Returns
        -------
        None.

        """
        self.area[0, :] = self.channel.initial_conditions[:, 0]
        self.flow[0, :] = self.channel.initial_conditions[:, 1]

    def run(self, verbose: int = 1):
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
        running = True
        
        while running:
            self.time_level += 1
            if self.time_level >= self.max_timelevels:
                running = False
                self.time_level = self.max_timelevels-1
                break
            
            if verbose >= 1:
                print(f'\n> Time level #{self.time_level}')
                
            for i in range(self.number_of_nodes):
                self.compute_node(i)
                        
            self.check_cfl_all()
        
        super().finalize(verbose=verbose)
        
    def us_ghost_node(self):
        if self.secondary_BC[0] == 'constant':
            return (self.area_at(k=-1, i=0), self.flow_at(k=-1, i=0), self.water_level_at(k=-1, i=0), self.Se_at(k=-1, i=0))
                                
        elif self.secondary_BC[0] == 'mirror':
            return (self.area_at(k=-1, i=1), self.flow_at(k=-1, i=1), self.water_level_at(k=-1, i=1), self.Se_at(k=-1, i=1))
                                
        elif self.secondary_BC[0] == 'linear':
            return (2 * self.area_at(k=-1, i=0) - self.area_at(k=-1, i=1),
                    2 * self.flow_at(k=-1, i=0) - self.flow_at(k=-1, i=1),
                    2 * self.water_level_at(k=-1, i=0) - self.water_level_at(k=-1, i=1),
                    2 * self.Se_at(k=-1, i=0) - self.Se_at(k=-1, i=1))
            
    def ds_ghost_node(self):
        if self.secondary_BC[0] == 'constant':
            return (self.area_at(k=-1, i=-1), self.flow_at(k=-1, i=-1), self.water_level_at(k=-1, i=-1), self.Se_at(k=-1, i=-1))
                                
        elif self.secondary_BC[0] == 'mirror':
            return (self.area_at(k=-1, i=-2), self.flow_at(k=-1, i=-2), self.water_level_at(k=-1, i=-2), self.Se_at(k=-1, i=-2))
                                
        elif self.secondary_BC[0] == 'linear':
            return (2 * self.area_at(k=-1, i=-1) - self.area_at(k=-1, i=-2),
                    2 * self.flow_at(k=-1, i=-1) - self.flow_at(k=-1, i=-2),
                    2 * self.water_level_at(k=-1, i=-1) - self.water_level_at(k=-1, i=-2),
                    2 * self.Se_at(k=-1, i=-1) - self.Se_at(k=-1, i=-2))
        
    def compute_upstream_node(self):
        ghost_A, ghost_Q, ghost_Y, ghost_Se = self.us_ghost_node()
        if self.channel.upstream_boundary.condition_type():
            A = self.new_area(A_im1=ghost_A,
                              A_ip1=self.area_at(k=-1, i=1),
                              Q_im1=ghost_Q,
                              Q_ip1=self.flow_at(k=-1, i=1))
            
            Q = -self.channel.upstream_boundary.condition_residual(
                time=self.time_level*self.time_step,
                depth=A/self.width_at(i=0),
                width=self.width_at(i=0),
                flow_rate=0,
                bed_slope=self.bed_slope_at(i=0),
                roughness=self.channel.get_n(A=A, i=0)
            )
            
        else:
            Q = self.new_flow(A_im1=ghost_A,
                              A_ip1=self.area_at(k=-1, i=1),
                              Q_im1=ghost_Q,
                              Q_ip1=self.flow_at(k=-1, i=1),
                              Y_im1=ghost_Y,
                              Y_ip1=self.water_level_at(k=-1, i=1),
                              Se_im1=ghost_Se,
                              Se_ip1=self.Se_at(k=-1, i=1))
            
            A = -self.width_at(i=0) * self.channel.upstream_boundary.condition_residual(
                time=self.time_level*self.time_step,
                depth=0,
                width=self.width_at(i=0)
            )
            
        self.area[self.time_level, 0] = A
        self.flow[self.time_level, 0] = Q

    def compute_node(self, i):
        if i == 0:
            self.compute_upstream_node()
        elif i == self.number_of_nodes - 1:
            self.compute_downstream_node()
        else:                
            self.area[self.time_level, i] = self.new_area(A_im1=self.area_at(k=-1, i=i-1),
                                                          A_ip1=self.area_at(k=-1, i=i+1),
                                                          Q_im1=self.flow_at(k=-1, i=i-1),
                                                          Q_ip1=self.flow_at(k=-1, i=i+1))
                                                          
            self.flow[self.time_level, i] = self.new_flow(A_im1=self.area_at(k=-1, i=i-1),
                                                          A_ip1=self.area_at(k=-1, i=i+1),
                                                          Q_im1=self.flow_at(k=-1, i=i-1),
                                                          Q_ip1=self.flow_at(k=-1, i=i+1),
                                                          Y_im1=self.water_level_at(k=-1, i=i-1),
                                                          Y_ip1=self.water_level_at(k=-1, i=i+1),
                                                          Se_im1=self.Se_at(k=-1, i=i-1),
                                                          Se_ip1=self.Se_at(k=-1, i=i+1))
    
    def compute_downstream_node(self):
        ghost_A, ghost_Q, ghost_Y, ghost_Se = self.ds_ghost_node()
        
        if self.channel.downstream_boundary.condition_type():
            A = self.new_area(A_im1=self.area_at(k=-1, i=-2),
                              A_ip1=ghost_A,
                              Q_im1=self.flow_at(k=-1, i=-2),
                              Q_ip1=ghost_Q)
            
            Q = -self.channel.downstream_boundary.condition_residual(
                time=self.time_level*self.time_step,
                depth=A/self.width_at(i=-1),
                width=self.width_at(i=-1),
                flow_rate=0,
                bed_slope=self.bed_slope_at(i=-1),
                roughness=self.channel.get_n(A=A, i=-1)
            )
            
        else:
            Q = self.new_flow(A_im1=self.area_at(k=-1, i=-2),
                              A_ip1=ghost_A,
                              Q_im1=self.flow_at(k=-1, i=-2),
                              Q_ip1=ghost_Q,
                              Y_im1=self.water_level_at(k=-1, i=-2),
                              Y_ip1=ghost_Y,
                              Se_im1=self.Se_at(k=-1, i=-2),
                              Se_ip1=ghost_Se)
                        
            A = -self.width_at(i=-1) * self.channel.downstream_boundary.condition_residual(
                time=self.time_level*self.time_step,
                depth=0,
                width=self.width_at(i=-1),
                duration=self.time_step,
                vol_in=0.5*(self.flow_at(k=-1, i=-1) + Q)*self.time_step,
                Y_old=self.water_level_at(k=-1, i=-1)
            )
            
        self.area[self.time_level, -1] = A
        self.flow[self.time_level, -1] = Q
        
    def new_area(self, A_im1, A_ip1, Q_im1, Q_ip1):
        avg_A  = self.cell_avg(ip1=A_ip1, im1=A_im1)
        
        dQ_dx = self.spatial_diff(
            ip1=Q_ip1,
            im1=Q_im1,
        )
                    
        return -dQ_dx * self.time_step + avg_A

    def new_flow(self, A_im1, A_ip1, Q_im1, Q_ip1, Y_ip1, Y_im1, Se_ip1, Se_im1):        
        avg_A  = self.cell_avg(ip1=A_ip1, im1=A_im1)
        avg_Q  = self.cell_avg(ip1=Q_ip1, im1=Q_im1)
        avg_Se = self.cell_avg(
            ip1=Se_ip1,
            im1=Se_im1,
        )
                
        dQ2A_dx = self.spatial_diff(
            ip1=Q_ip1**2 / A_ip1,
            im1=Q_im1**2 / A_im1
        )
        
        dY_dx = self.spatial_diff(
            ip1=Y_ip1,
            im1=Y_im1
        )
            
        return -(dQ2A_dx + g * avg_A * (dY_dx + avg_Se)) * self.time_step + avg_Q
                
    def check_cfl_all(self):
        for i in range(self.number_of_nodes):
            V = self.flow_at(i=i) / self.area_at(i=i)
            h = self.depth_at(i=i)
            
            if self.check_cfl_condition(V, h) == False:
                analytical_celerity = max(V + (g * h) ** 0.5, V - (g * h) ** 0.5)
                raise ValueError(
                    f'CFL condition failed at i={i}, k={self.time_level}. CFL number = {analytical_celerity/self.num_celerity}'
                )
    
    def check_cfl_condition(self, velocity, depth):
        analytical_celerity = max(velocity + (g * depth) ** 0.5, velocity - (g * depth) ** 0.5)
        return self.num_celerity >= analytical_celerity
    
    def time_diff(self, kp1, ip1, im1):
        k = self.cell_avg(ip1=ip1, im1=im1)
        return (kp1 - k) / self.time_step
    
    def spatial_diff(self, ip1, im1):
        return 0.5 * (ip1 - im1) / self.spatial_step
            
    def cell_avg(self, ip1, im1):
        return 0.5 * (ip1 + im1)
    