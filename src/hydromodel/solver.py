import numpy as np
import pandas as pd
from scipy.constants import g
from .channel import Channel
from .utility import create_directory_if_not_exists
from .utility import seconds_to_hms

class Solver:
    def __init__(self,
                 channel: Channel,
                 time_step: int | float,
                 spatial_step: int | float,
                 simulation_time: int,
                 regularization: bool = True,
                 fit_spatial_step: bool = True):
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
        self.channel = channel                
        self.time_step, self.spatial_step = time_step, spatial_step
        self.time_level = 0
        self.number_of_nodes = self.channel.length // self.spatial_step + 1
        self.max_timelevels = simulation_time // self.time_step + 1
        
        if fit_spatial_step:
            self.fit_spatial_step()
            
        self.channel.initialize_conditions(n_nodes=self.number_of_nodes)
        self.num_celerity = self.spatial_step / self.time_step

        self.flow = np.empty(shape=(self.max_timelevels, self.number_of_nodes), dtype=np.float64)
        self.depth = np.empty_like(self.flow)
        self.area = np.empty_like(self.flow)
                        
        self._type = None
        self._solved = False
        self._new_time_level = False
        self.total_sim_duration = 0
        self.regularization = regularization
        self.eps = 1e-4

    def fit_spatial_step(self):
        self.number_of_nodes = round(self.channel.length / self.spatial_step) + 1
        self.spatial_step = self.channel.length / (self.number_of_nodes - 1)
        
    def prepare_results(self) -> None:                
        self.velocity = self.flow / self.area
        self.bed_profile = np.array(object=[xs.bed for xs in self.channel.xs_at_node], dtype=np.float64)
        self.level = self.depth + self.bed_profile
        self.wave_celerity = self.velocity + np.sqrt(g * self.depth)
        
        ref = self.depth[0, :]
        deviation = np.abs(self.depth - ref)
        self.peak_amplitude = deviation.max(axis=0)
        
        if self.channel.downstream_boundary.lumped_storage is not None:
            Y = self.water_level_at(k=0, i=-1)
            initial_storage_stage = Y - self.channel.downstream_boundary.lumped_storage.energy_loss(
                entry_area=self.area_at(k=0, i=-1),
                flow=self.flow_at(k=0, i=-1),
                roughness=self.channel.xs_at_node[-1].get_equivalent_n(hw=Y),
                hydraulic_radius=self.channel.xs_at_node[-1].hydraulic_radius(hw=Y)
            )
            self.channel.downstream_boundary.lumped_storage.stage_hydrograph.insert(0, [0, initial_storage_stage])
            self.storage_stage = np.array(self.channel.downstream_boundary.lumped_storage.stage_hydrograph, dtype=np.float64)
            self.storage_stage = self.storage_stage[:, 1].flatten()
            
            self.storage_outflow = np.empty(shape=(self.max_timelevels), dtype=np.float64)
            if self.channel.downstream_boundary.lumped_storage.rating_curve is None:
                self.storage_outflow[0] = 0
            else:
                self.storage_outflow[0] = min(
                    self.flow_at(k=0, i=-1),
                    self.channel.downstream_boundary.lumped_storage.rating_curve.discharge(stage=self.storage_stage[0], time=0)
                    )
            
            for k in range(1, self.max_timelevels):
                avg_inflow = 0.5 * (self.flow_at(k=k-1, i=-1) + self.flow_at(k=k, i=-1))
                Y1 = self.storage_stage[k-1]; Y2 = self.storage_stage[k]
                vol_change = self.channel.downstream_boundary.lumped_storage.net_vol_change(Y1=Y1, Y2=Y2)
                avg_outflow = avg_inflow - vol_change / self.time_step
                
                self.storage_outflow[k] = avg_outflow * self.flow_at(k=k, i=-1) / avg_inflow
                    
    def save_results(self, folder_path):
        """
        Save all results to a single Excel workbook with multiple sheets.
        """
        create_directory_if_not_exists(folder_path)
        filename = folder_path + "\\results.xlsx"
        
        arrays_2d = {
            "Area": self.area,
            "Flow": self.flow,
            "Velocity": self.velocity,
            "Depth": self.depth,
            "Level": self.level,
            "Wave celerity": self.wave_celerity,
        }

        nt, nx = next(iter(arrays_2d.values())).shape
        time = np.arange(nt) * self.time_step
        distance = np.arange(nx) * self.spatial_step

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            # 2D arrays
            for name, arr in arrays_2d.items():
                df = pd.DataFrame(arr, index=time, columns=distance)
                df.index.name = "Time"
                df.columns.name = "Distance"
                df.to_excel(writer, sheet_name=name)

            # Outflow (1D time series)
            if self.channel.downstream_boundary.lumped_storage is not None:
                df_out = pd.DataFrame({"outflow": self.storage_outflow}, index=time)
                df_out.index.name = "Time"
                df_out.to_excel(writer, sheet_name="Outflow")
                
                if self._type=='preissmann':
                    df_stage = pd.DataFrame({"stage": self.storage_stage}, index=time)
                    df_stage.index.name = "Time"
                    df_stage.to_excel(writer, sheet_name="Reservoir stage")

            # Peak amplitude (row with distance headers)
            df_peak = pd.DataFrame([self.peak_amplitude], columns=distance)
            df_peak.index = ["Peak amplitude"]
            df_peak.columns.name = "Distance"
            df_peak.to_excel(writer, sheet_name="Peak amplitude")

            # Width (row with distance headers)
            df_width = pd.DataFrame([np.array(object=[xs.width for xs in self.channel.xs_at_node], dtype=np.float64)], columns=distance)
            df_width.index = ["Width"]
            df_width.columns.name = "Distance"
            df_width.to_excel(writer, sheet_name="Width")

            # Bed level (row with distance headers)
            df_bed = pd.DataFrame([self.bed_profile], columns=distance)
            df_bed.index = ["Bed level"]
            df_bed.columns.name = "Distance"
            df_bed.to_excel(writer, sheet_name="Bed level")
                
        # Save data summary
        with open(folder_path + '\\Data.txt', 'w') as output_file:
            # Spatial step
            output_file.write(f'Spatial step = {self.spatial_step} m\n')
            
            # Time step
            output_file.write(f'Time step = {self.time_step} s\n')
            
            # Theta
            if self._type == 'preissmann':
                output_file.write(f'Theta = {self.theta}\n')
            
            # Simulation duration
            output_file.write(f'Simulation duration = {seconds_to_hms(self.total_sim_duration)}\n')
            
            # Mass imbalance
            x = np.array(self.flow)
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
                
    def _finalize(self, verbose):
        self._solved = True
        self.total_sim_duration = self.time_level * self.time_step
        
        self.prepare_results()
        
        if verbose >= 1:
            print("Simulation completed successfully.")
            
    def area_at(self, k: int = None, i: int = None, regularization: bool = None):
        if i is None:
            raise ValueError("Spatial node must be specified.")
        
        A = self.channel.area_at(i=i, h=self.depth_at(k=k, i=i))
            
        if regularization is None:
            regularization = self.regularization
        
        if regularization:
            h_min = 1e-4
            A_min = self.channel.area_at(i=i, h=h_min)
            A = self.A_reg(A, A_min)
        
        return A
        
    def flow_at(self, k: int = None, i: int = None, chi_scaling: bool = None):
        if i is None:
            raise ValueError("Spatial node must be specified.")
        
        k = self.time_level if k is None else self.time_level-1 if k == -1 else k
        Q = self.flow[k, i]
            
        if chi_scaling is None:
            chi_scaling = self.regularization
            
        if chi_scaling:
            A_reg = self.area_at(k=k, i=i, regularization=True)
            h_min = 1e-4
            A_min = self.channel.area_at(i=i, h=h_min)
            
            chi = A_reg / (A_reg + A_min)
            Q = Q * chi
            
        return Q
        
    def depth_at(self, k: int = None, i: int = None, regularization: bool = None):
        if i is None:
            raise ValueError("Spatial node must be specified.")
        
        k = self.time_level if k is None else self.time_level-1 if k == -1 else k
        return self.depth[k, i]
    
    def water_level_at(self, k: int = None, i: int = None, regularization: bool = None):
        return self.channel.xs_at_node[i].bed + self.depth_at(k=k, i=i, regularization=regularization)
        
    def Se_at(self, k: int = None, i: int = None, regularization: bool = None, chi_scaling: bool = None):
        return self.channel.Se(h=self.depth_at(k=k, i=i, regularization=regularization),
                               Q=self.flow_at(k=k, i=i, chi_scaling=chi_scaling),
                               i=i)
        
    def dA_dh(self, k: int = None, i: int = None, regularization: bool = None):
        return 1.0 / self.channel.dh_dA(i=i, h=self.depth_at(k=k, i=i, regularization=regularization))
    
    def A_reg(self, A):
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
        A_min = self.channel.area_at(i=0, h=h_min)
        
        return A_min + 0.5 * (
            (A - A_min) + np.sqrt(
                (A - A_min) ** 2 + self.eps ** 2
                )
            )
        
    def Q_eff(self, Q, A_reg):
        h_min = 1e-4
        A_min = self.channel.area_at(i=0, h=h_min)
        
        chi = A_reg / (A_reg + A_min)
        return Q * chi
        