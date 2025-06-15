from river import River
import numpy as np
from scipy.constants import g
from utility import Utility


class Solver:
    """
    Implements the Preissmann implicit finite difference scheme to numerically
    solve the Saint-Venant equations.

    Attributes
    ----------
    river : River
        An instance of the `River` class, representing the river being modeled.
    time_step : float
        Time step for the simulation in seconds.
    spatial_step : float
        Spatial step for the simulation in meters.
    num_celerity : float
        Ratio of spatial to time step, representing the wave celerity.
    number_of_nodes : int
        Number of spatial nodes along the river.
    A_previous : list of float
        Cross-sectional areas at the previous time step.
    Q_previous : list of float
        Discharges at the previous time step.
    A_current : list of float
        Cross-sectional areas at the current iteration of the current time step.
    Q_current : list of float
        Discharges at the current iteration of the current time step.
    computed_areas : list of list of float
        Stores the computed A values over time.
    computed_flow_rates : list of list of float
        Stores the computed Q values over time.
            
    """

    def __init__(self,
                 river: River,
                 time_step: int | float,
                 spatial_step: int | float,
                 fit_spatial_step = True):
        """
        Initializes the class.

        Parameters
        ----------
        river : River
            The River object on which the simulation is performed.
        time_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
            
        """
        self.type = None
        self.river = river
        self.solved = False
        self.active_storage = self.river.downstream_boundary.active_storage
                
        self.time_step, self.spatial_step = time_step, spatial_step
        self.num_celerity = self.spatial_step / float(self.time_step)

        if fit_spatial_step:
            self.number_of_nodes = int(round(self.river.total_length / self.spatial_step) + 1)
            self.spatial_step = self.river.total_length / (self.number_of_nodes - 1)
        else:
            self.number_of_nodes = int(self.river.total_length // self.spatial_step + 1)

        self.A_previous = []
        self.Q_previous = []

        self.A_current = []
        self.Q_current = []

        self.computed_areas = []
        self.computed_flow_rates = []
        
        self.peak_amplitude_profile = None
        self.peak_inflow = 0
        self.peak_outflow = 0
        self.peak_inflow_time = 0
        self.peak_arrival_time = 0
        self.depth_0 = 0
        self.total_sim_duration = 0

    def append_result(self):
        if isinstance(self.A_current, list):
            self.computed_areas.append([float(i) for i in self.A_current])
            self.computed_flow_rates.append([float(i) for i in self.Q_current])
        
        else:
            self.computed_areas.append(self.A_current.tolist())
            self.computed_flow_rates.append(self.Q_current.tolist())
            
        time = len(self.computed_areas) * self.time_step
        
        if self.Q_current[0] > self.peak_inflow:
            self.peak_inflow = self.Q_current[0]
            self.peak_inflow_time = time
        
        if self.Q_current[-1] > self.peak_outflow:
            self.peak_outflow = self.Q_current[-1]
            self.peak_arrival_time = time
        
        if self.peak_amplitude_profile is None:
            self.depth_0 = np.array(self.computed_areas[0]) / self.river.width
            self.peak_amplitude_profile = self.depth_0 - self.depth_0
        else:
            peak_amplitudes = np.array(self.A_current) / self.river.width - self.depth_0
            self.peak_amplitude_profile = [float(max(i, j)) for i,j in zip(peak_amplitudes, self.peak_amplitude_profile)]
        
    
    def save_results(self, size: tuple = (-1, -1), path: str = 'Results') -> None:
        """
        Saves the results of the simulation in four .csv files, containing
        the computed cross-sectional flow area, discharge, flow depth, and velocity.
        The files are formatted with each row representing a time step and each
        column representing a spatial point. The left-most column contains the
        time coordinate starting at t=0.

        Parameters
        ----------
        size : tuple of float
            The number of temporal and spatial nodes to save.

        Returns
        -------
        None.

        """
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
        velocities = np.array(self.computed_flow_rates) / np.array(self.computed_areas)
        velocities = velocities.tolist()
        velocities = [v[::x_step]
                for v in velocities[::t_step]]
        
        # Compute depths and levels
        depths = np.array(self.computed_areas) / self.river.width
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
        
        # Create header with time column first, then spatial coordinates
        spatial_header = [self.river.upstream_boundary.chainage + x * self.spatial_step for x in range(0, self.number_of_nodes, x_step)]
        header = ['Time'] + spatial_header
        header_str = str(header)
        for c in "[]' ":
            header_str = header_str.replace(c, '')
            
        header_str += '\n'

        # Generate time coordinates for each saved time step
        time_coords = [i * t_step * self.time_step for i in range(len(areas))]

        for key, value in data.items():
            # Add time column to each row of data
            value_with_time = []
            for i, row in enumerate(value):
                value_with_time.append([time_coords[i]] + row)
            
            value_str = str(value_with_time).replace('], [', '\n')
            for c in "[]' ":
                value_str = value_str.replace(c, '')
            with open(path + f'//{key}.csv', 'w') as output_file:
                output_file.write(header_str)
                output_file.write(value_str)
        
        # Save peak amplitude profile
        
        value_str = str([0] + self.peak_amplitude_profile)
        for c in "[]' ":
            value_str = value_str.replace(c, '')
                
        with open(path + '//Peak_amplitude_profile.csv', 'w') as output_file:
                output_file.write(header_str)
                output_file.write(value_str)
                
        # Save other data
        
        with open(path + '//Data.txt', 'w') as output_file:
            output_file.write(f'Spatial step = {self.spatial_step} m\n')
            output_file.write(f'Time step = {self.time_step} s\n')
            if self.type == 'preissmann':
                output_file.write(f'Theta = {self.theta}\n')
                
            output_file.write(f'Simulation duration = {Utility.seconds_to_hms(self.total_sim_duration)}\n')
            
            output_file.write(f'Peak inflow = {self.peak_inflow:.2f} m^3/s\n')
            output_file.write(f'Peak inflow time = {Utility.seconds_to_hms(self.peak_inflow_time)}\n')
            
            output_file.write(f'Peak outflow = {self.peak_outflow:.2f} m^3/s\n')
            output_file.write(f'Peak arrival time = {Utility.seconds_to_hms(self.peak_arrival_time)}\n')
            
            output_file.write(f'Peak travel time = {Utility.seconds_to_hms(self.peak_arrival_time - self.peak_inflow_time)}\n')
            
            x = np.array(self.computed_flow_rates)
            Q_in = x[:, 0]
            Q_out = x[:, -1]
            mass_imbalance = np.sum(Q_in - Q_out) * self.time_step

            output_file.write(f'Mass imbalance (total inflow - total outflow) = {mass_imbalance:.2f} m^3\n')
                
    
    def get_results(self, parameter: str, spatial_node: int = None, temporal_node: int = None):
        if not self.solved:
            raise ValueError("Not solved yet.")
        
        reqursted = None

        if parameter == 'a':
            reqursted = self.computed_areas
            
        elif parameter == 'q':
            reqursted = self.computed_flow_rates
            
        elif parameter == 'v':
            velocities = np.array(self.computed_flow_rates) / np.array(self.computed_areas)
            reqursted = velocities
        
        elif parameter == 'h':
            depths = np.array(self.computed_areas) / self.river.width
            reqursted = depths
        
        else:
            raise ValueError("Invalid parameter. Choose between 'a', 'q', 'v', or 'h'.")
        
        if isinstance(reqursted, list):
            reqursted = np.array(reqursted)
        
        if spatial_node is not None:
            if temporal_node is not None:
                return reqursted[temporal_node, spatial_node]
            else:
                return reqursted[:, spatial_node]
        
        if temporal_node is not None:
            return reqursted[temporal_node, :]
            
        return reqursted
    