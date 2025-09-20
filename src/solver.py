from src.reach import Reach
import numpy as np
from scipy.constants import g
from math import sqrt


class Solver:
    def __init__(self,
                 reach: Reach,
                 time_step: int | float,
                 spatial_step: int | float,
                 simulation_time: int,
                 enforce_physicality: bool = True,
                 fit_spatial_step = True):
        """
        Initializes the class.

        Parameters
        ----------
        reach : reach
            The Reach object on which the simulation is performed.
        time_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
            
        """
        self.reach = reach
        self.active_storage = (self.reach.downstream_boundary.lumped_storage is not None)
                
        self.time_step, self.spatial_step = time_step, spatial_step
        self.number_of_nodes = self.reach.length // self.spatial_step + 1
        self.max_timesteps = simulation_time // self.time_step + 1
        
        if fit_spatial_step:
            self.fit_spatial_step()
            
        self.reach.initialize_conditions(n_nodes=self.number_of_nodes)
        self.num_celerity = self.spatial_step / self.time_step

        self.areas = np.empty(shape=(self.max_timesteps, self.number_of_nodes), dtype=float)
        self.flow = np.empty(shape=(self.max_timesteps, self.number_of_nodes), dtype=float)
        
        self.A_previous = None
        self.Q_previous = None

        self.A_current = None
        self.Q_current = None

        self.results = {
            'area': np.zeros(shape=(1, self.number_of_nodes), dtype=float),
            'flow_rate': np.zeros(shape=(1, self.number_of_nodes), dtype=float),
            'velocity': np.zeros(shape=(1, self.number_of_nodes), dtype=float),
            'depth': np.zeros(shape=(1, self.number_of_nodes), dtype=float),
            'level': np.zeros(shape=(1, self.number_of_nodes), dtype=float),
            'wave_celerity': np.zeros(shape=(1, self.number_of_nodes), dtype=float),
            'outflow': np.array(self.reach.initial_conditions[:, 1], dtype=float)
        }
        
        self.peak_amplitudes = None
        self.initial_depths = None
        self.type = None
        self.solved = False
        self.total_sim_duration = 0
        self.enforce_physicality = enforce_physicality

    def fit_spatial_step(self):
        self.number_of_nodes = round(self.reach.length / self.spatial_step) + 1
        self.spatial_step = self.reach.length / (self.number_of_nodes - 1)

    def append_result(self):
        self.results['area'] = np.vstack((self.results['area'], self.A_current))
        self.results['flow_rate'] = np.vstack((self.results['flow_rate'], self.Q_current))
                    
        if self.initial_depths is None:
            self.initial_depths = np.array(self.results['area'][0]) / np.array(self.reach.widths)
            self.peak_amplitudes = np.zeros(shape=(self.number_of_nodes))
        else:
            amplitudes = np.array(self.A_current) / np.array(self.reach.widths) - self.initial_depths
            for i in range(self.number_of_nodes):
                self.peak_amplitudes[i] = max(amplitudes[i], self.peak_amplitudes[i])
            
        if self.type == 'preissmann' and self.active_storage:
            self.results['outflow'] = np.append(self.results['outflow'], self.Q_out)
        
    def prepare_results(self) -> None:
        self.results['velocity'] = self.results['flow_rate'] / self.results['area']
        self.results['depth'] = self.results['area'] / self.reach.widths.reshape(1, -1)
        self.results['level'] = self.results['depth'] + self.reach.bed_levels.reshape(1, -1)
        self.results['wave_celerity'] = self.results['velocity'] + np.sqrt(self.results['depth'] * g)
    
    def save_results(self, size: tuple = (-1, -1), path: str = 'results') -> None:
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
        from src.utility import create_directory_if_not_exists
        create_directory_if_not_exists(path)
        
        # Calculate steps
        t_step = x_step = 1
        
        if size[0] > 1:
            t_step = (len(self.results['area']) - 1) // (size[0] - 1)

        if size[1] > 1:
            x_step =  (self.number_of_nodes - 1) // (size[1] - 1)
                    
        # Compute the sets to be saved based on the steps
        areas = [a[::x_step]
                for a in self.results['area'][::t_step]]

        flow_rates = [q[::x_step]
                for q in self.results['flow_rate'][::t_step]]
        
        velocities = [v[::x_step]
                for v in self.results['velocity'][::t_step]]
        
        levels = [levels[::x_step]
                for levels in self.results['level'][::t_step]]
        
        depths = [h[::x_step]
                for h in self.results['depth'][::t_step]]
        
        celerities = [c[::x_step]
                for c in self.results['wave_celerity'][::t_step]]

        data = {
            'flow_area': areas,
            'flow_rate': flow_rates,
            'flow_depth': depths,
            'flow_velocity': velocities,
            'water_surface_level': levels,
            'analytical_wave_celerity': celerities,
            'bed_profile': [self.reach.bed_levels],
            'widths': [self.reach.widths]
        }
        
        # Create header with time column first, then spatial coordinates
        spatial_header = [self.reach.upstream_boundary.chainage + x * self.spatial_step for x in range(0, self.number_of_nodes, x_step)]
        header = ['Time'] + spatial_header
        header_str = str(header)
        for c in "[]' ":
            header_str = header_str.replace(c, '')
            
        header_str += '\n'

        # Generate time coordinates for each saved time step
        time_coords = [i * t_step * self.time_step for i in range(len(areas))]

        # Save data
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
        value_str = str([''] + self.peak_amplitudes)
        for c in "[]' ":
            value_str = value_str.replace(c, '')
                
        with open(path + '//Peak_amplitude_profile.csv', 'w') as output_file:
                output_file.write(header_str)
                output_file.write(value_str)
                
        # Save outflow hydrograph
        header = ['Time'] + time_coords
        header_str = str(header)
        for c in "[]' ":
            header_str = header_str.replace(c, '')
        header_str += '\n'
        value_str = str([''] + self.results['outflow'])
        for c in "[]' ":
            value_str = value_str.replace(c, '')
                
        with open(path + '//outflow.csv', 'w') as output_file:
                output_file.write(header_str)
                output_file.write(value_str)
                
        # Save data summary
        with open(path + '//Data.txt', 'w') as output_file:
            # Spatial step
            output_file.write(f'Spatial step = {self.spatial_step} m\n')
            
            # Time step
            output_file.write(f'Time step = {self.time_step} s\n')
            
            # Theta
            if self.type == 'preissmann':
                output_file.write(f'Theta = {self.theta}\n')
            
            # Simulation duration
            from src.utility import seconds_to_hms
            output_file.write(f'Simulation duration = {seconds_to_hms(self.total_sim_duration)}\n')
            
            # Mass imbalance
            x = np.array(self.results['flow_rate'])
            Q_in = x[:, 0]
            Q_out = x[:, -1]
            mass_imbalance = np.sum(Q_in - Q_out) * self.time_step
            mass_imbalance_percentage = float(mass_imbalance / self.time_step / np.sum(Q_in)) * 100
            output_file.write(f'Mass imbalance (total inflow - total outflow) = {mass_imbalance:.2f} m^3 = {mass_imbalance_percentage:.4f}% of inflow.\n')
            
            # Peak magnitude and attenuation
            peak_inflow = np.max(Q_in)
            peak_outflow = np.max(Q_out)
            attenuation_percentage = (peak_inflow - peak_outflow) / peak_inflow * 100
            output_file.write(f'Peak inflow = {peak_inflow:.2f} m^3/s\n')
            output_file.write(f'Peak outflow = {peak_outflow:.2f} m^3/s\n')
            output_file.write(f'Attenuation = {attenuation_percentage:.2f}%\n')
            
            # Median volume timing
            cumulative_inflow = [np.sum(Q_in[:i]) for i in range(Q_in.size)]            
            for i in range(len(cumulative_inflow)):
                if cumulative_inflow[i] >= 0.5 * cumulative_inflow[-1]:
                    median_vol_entry_time = i * self.time_step
                    break
                    
            cumulative_outflow = [np.sum(Q_out[:i]) for i in range(Q_out.size)]
            for i in range(len(cumulative_outflow)):
                if cumulative_outflow[i] >= 0.5 * cumulative_outflow[-1]:
                    median_vol_arrival_time = i * self.time_step
                    break
                
            output_file.write(f'Median volume entry time = {seconds_to_hms(median_vol_entry_time)}\n')
            output_file.write(f'Median volume arrival time = {seconds_to_hms(median_vol_arrival_time)}\n')
            output_file.write(f'Median volume travel time = {seconds_to_hms(median_vol_arrival_time - median_vol_entry_time)}\n')
            
            
                
    def get_results(self, parameter: str, spatial_node: int = None, temporal_node: int = None):
        """
        Retrieves the whole or a part of the 2D list containing the computed values of the specified parameter.

        Parameters
        ----------
        parameter : str
            The name of the wanted parameter. It should be one of the following:
            'area', 'flow_rate', 'velocity', 'depth', 'level', 'wave_celerity'.

        Returns
        -------
        float | np.ndarray
            The requested parameter(s).

        """
        
        if not self.solved:
            raise ValueError("Not solved yet.")
        
        if parameter not in ['area', 'flow_rate', 'velocity', 'depth', 'level', 'wave_celerity', 'outflow']:
            raise ValueError(f"'parameter' should take one of these values: 'area', 'flow_rate', 'velocity', 'depth', 'level', 'wave_celerity'\nYou entered: {parameter}")
        
        reqursted = self.results[parameter]
        
        if spatial_node is not None:
            if temporal_node is not None:
                return float(reqursted[temporal_node, spatial_node])
            else:
                return reqursted[:, spatial_node]
        
        elif temporal_node is not None:
            return reqursted[temporal_node, :]
            
        else:
            return reqursted
    
    def finalize(self, time, verbose):
        self.solved = True
        self.total_sim_duration = time
        
        self.prepare_results()
        
        if verbose >= 1:
            print("Simulation completed successfully.")
            
    def area_at(self, i, current_time_level: bool, regularization: bool = None):
        if current_time_level:
            A = self.A_current[i]
        else:
            A = self.A_previous[i]
            
        if regularization is None:
            regularization = self.enforce_physicality
        
        if regularization:
            h_min = 1e-4
            A_min = self.width_at(i) * h_min
            A = self.A_reg(A, A_min)
        
        return A
        
    def flow_at(self, i, current_time_level: bool, chi_scaling: bool = None):
        if current_time_level:
            Q = self.Q_current[i]
        else:
            Q = self.Q_previous[i]
            
        if chi_scaling is None:
            chi_scaling = self.enforce_physicality
            
        if chi_scaling:
            A_reg = self.area_at(i, current_time_level, 1)
            h_min = 1e-4
            A_min = self.width_at(i) * h_min
            
            chi = A_reg / (A_reg + A_min)
            Q = Q * chi
            
        return Q
    
    def width_at(self, i):
        return self.reach.widths[i]
    
    def bed_slope_at(self, i):
        return self.reach.bed_slopes[i]
    
    def depth_at(self, i, current_time_level: bool, regularization: bool = None):
        return self.area_at(i, current_time_level, regularization) / self.width_at(i)
    
    def water_level_at(self, i, current_time_level: bool, regularization: bool = None):
        return self.reach.bed_levels[i] + self.depth_at(i, current_time_level, regularization)
    
    def wet_depth_at(self, i):
        return self.reach.initial_conditions[i][0] / self.width_at(i)
    
    def Se_at(self, i, current_time_level: bool, regularization: bool = None, chi_scaling: bool = None):
        A = self.area_at(i, current_time_level, regularization)
        Q = self.flow_at(i, current_time_level, chi_scaling)
        
        return self.reach.Se(A, Q, i)
    
    def A_reg(self, A, eps=1e-4):
        """
        Regularized wetted area.
        
        Parameters
        ----------
        A : float
            Raw area value.
        eps : float
            Smoothing parameter.

        Returns
        -------
        float
            Regularized area.
            
        """
        h_min = 1e-4
        A_min = self.width_at(0) * h_min
        
        return A_min + 0.5 * (
            (A - A_min) + np.sqrt(
                (A - A_min) ** 2 + eps ** 2
                )
            )
        
    def Q_eff(self, Q, A_reg):
        h_min = 1e-4
        A_min = self.width_at(0) * h_min
        
        chi = A_reg / (A_reg + A_min)
        return Q * chi
        