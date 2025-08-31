from solver import Solver
from reach import Reach
import numpy as np
from scipy.constants import g
from utility import Utility


class PreissmannSolver(Solver):
    """
    Implements the Preissmann implicit finite-difference scheme to numerically
    solve the Saint-Venant equations.

    Attributes
    ----------
    theta : float
        Preissmann's weighting factor.
    unknowns : list of float
        Vector of unknowns (x).
            
    """

    def __init__(self,
                 reach: Reach,
                 theta: int | float,
                 time_step: int | float,
                 spatial_step: int | float,
                 enforce_physicality: bool = True,
                 fit_spatial_step = True):
        """
        Initializes the class.

        Parameters
        ----------
        reach : Reach
            The Reach object representing the simulated channel reach.
        theta : float
            Preissmann's weighting factor.
        time_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
            
        """
        super().__init__(reach, time_step, spatial_step, enforce_physicality, fit_spatial_step)
        
        self.theta = theta
        self.unknowns = []
        self.type = 'preissmann'
        
        self.initialize_t0()


    def initialize_t0(self) -> None:
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the reach.

        Returns
        -------
        None.

        """
        self.reach.initialize_conditions(self.number_of_nodes)

        for A, Q in self.reach.initial_conditions:                
            self.unknowns += [A, Q]
            self.A_previous.append(A)
            self.Q_previous.append(Q)

        if self.active_storage:
            self.unknowns.append(self.reach.downstream_boundary.storage_stage)
            
        self.unknowns = np.array(self.unknowns)

        self.results['area'].append(self.A_previous)
        self.results['flow_rate'].append(self.Q_previous)

    def compute_residual_vector(self, time) -> np.ndarray:
        """
        Computes the residual vector R.

        Parameters
        ----------
        time : float
            The current time of the simulation in seconds.

        Returns
        -------
        np.ndarray
            R
            
        """
        equation_list = [self.upstream_residual(time)]
        
        for i in range(self.number_of_nodes - 1):
            equation_list.append(self.continuity_residual(i))
            equation_list.append(self.momentum_residual(i))

        equation_list.append(self.downstream_residual(time=time))
        
        if self.active_storage:
            equation_list.append(self.storage_residual())
            
        return np.array(equation_list)

    def compute_jacobian(self, t) -> np.ndarray:
        """
        Constructs the Jacobian matrix J of the system of equations.

        Returns
        -------
        np.ndarray
            J
            
        """
        if self.active_storage:
            matrix_shape = (2 * self.number_of_nodes + 1, 2 * self.number_of_nodes + 1)
        else:
            matrix_shape = (2 * self.number_of_nodes, 2 * self.number_of_nodes)
            
        jacobian_matrix = np.zeros(shape=matrix_shape)

        jacobian_matrix[0, 0] = self.dU_dA(t)
        jacobian_matrix[0, 1] = self.dU_dQ()

        for row in range(1, 2 * self.number_of_nodes - 1, 2):
            i = (row - 1) // 2
            
            jacobian_matrix[row, row - 1] = self.dC_dAi(i)
            jacobian_matrix[row, row + 0] = self.dC_dQi(i)
            jacobian_matrix[row, row + 1] = self.dC_dAiplus1(i)
            jacobian_matrix[row, row + 2] = self.dC_dQiplus1(i)

            jacobian_matrix[row + 1, row - 1] = self.dM_dAi(i)
            jacobian_matrix[row + 1, row + 0] = self.dM_dQi(i)
            jacobian_matrix[row + 1, row + 1] = self.dM_dAiplus1(i)
            jacobian_matrix[row + 1, row + 2] = self.dM_dQiplus1(i)
        
        if self.active_storage:        
            jacobian_matrix[-2, -3] = self.dD_dA(t)
            jacobian_matrix[-2, -2] = self.dD_dQ()
            jacobian_matrix[-2, -1] = self.dD_dhs()
            
            jacobian_matrix[-1, -2] = self.dS_dQ()
            jacobian_matrix[-1, -1] = self.dS_dhs()
        else:
            jacobian_matrix[-1, -2] = self.dD_dA(t)
            jacobian_matrix[-1, -1] = self.dD_dQ()

        return jacobian_matrix

    def run(self, duration: int, auto = False, tolerance=1e-4, verbose=3) -> None:
        """
        Run the simulation.

        Parameters
        ----------
        duration : int
            The simulation duration in seconds.
        tolerance : float, optional
            The allowed tolerance for the iterative process. The simulation iterates until the cumulative error
            falls below this value. The default is 1e-4.

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
                print('\n---------- Time = ' + str(time) + 's ----------\n')

            iteration = 0
            converged = False

            while not converged:
                iteration += 1
                if iteration - 1 >= 100:
                    raise ValueError(f'Convergence within {iteration - 1} iterations couldn\'t be achieved.')
                if verbose >= 2:
                    print("--- Iteration #" + str(iteration) + ':')

                self.update_guesses()
                
                R = self.compute_residual_vector(time)
                J = self.compute_jacobian(time)
                
                if np.isnan(R).any() or np.isnan(J).any():
                    self.log_status()
                    raise ValueError("Solution failed.\n")

                delta = np.linalg.solve(J, -R)
                self.unknowns += delta

                error = Utility.euclidean_norm(delta)                
                if verbose >= 3:
                    print("Error = " + str(error))
                    
                if error < tolerance:
                    converged = True
                    if iteration == 1 and auto and time >= duration:
                        running = False
                                    
            self.append_result()
            self.update_parameters()
        
        super().finalize(time, verbose)
    
    def update_parameters(self) -> None:
        """
        Updates the values of the flow variables of the previous time step
        to the ones last computed.

        Returns
        -------
        None.

        """
        if self.active_storage:
            self.A_previous = [i for i in self.unknowns[:-1:2]]
            self.Q_previous = [i for i in self.unknowns[1:-1:2]]
        else:
            self.A_previous = [i for i in self.unknowns[::2]]
            self.Q_previous = [i for i in self.unknowns[1::2]]

    def update_guesses(self) -> None:
        """
        Set the trial values of the unknowns to the ones last computed.

        Returns
        -------
        None.

        """
        if self.active_storage:
            self.A_current = self.unknowns[:-1:2]
            self.Q_current = self.unknowns[1:-1:2]
            self.reach.downstream_boundary.storage_stage = self.unknowns[-1]
        else:
            self.A_current = self.unknowns[::2]
            self.Q_current = self.unknowns[1::2]
        
    def upstream_residual(self, time) -> float:
        """
        Computes the residual of the upstream boundary condition equation.

        Parameters
        ----------
        time : float
            Current simulation time in seconds.

        Returns
        -------
        float
            The computed residual.

        """
        A = self.area_at(0, 1)
        Q = self.flow_at(0, 1)
        
        depth = A / self.reach.width
                
        residual = self.reach.upstream_boundary.condition_residual(time, depth, self.reach.width, Q, self.reach.bed_slope,
                                                                   self.reach.manning_co, self.time_step)

        return residual

    def continuity_residual(self, i) -> float:
        """
        Computes the residual of the continuity equation for a specific node.

        Parameters
        ----------
        i : int
            Spatial node index.

        Returns
        -------
        float
            The computed residual.

        """        
        residual = (
            self.num_celerity * (self.area_at(i, 1) + self.area_at(i + 1, 1))
            + 2 * self.theta * (self.flow_at(i + 1, 1) - self.flow_at(i, 1))
            - (
                self.num_celerity * (self.area_at(i + 1, 0) + self.area_at(i, 0))
                - 2 * (1 - self.theta) * (self.flow_at(i + 1, 0) - self.flow_at(i, 0))
            )
        )

        return residual

    def momentum_residual(self, i) -> float:
        """
        Computes the residual of the momentum equation for a specific node.

        Parameters
        ----------
        i : int
            Spatial node index.

        Returns
        -------
        float
            The computed residual.

        """     
        residual = (
            (g * self.theta / self.reach.width) * (self.area_at(i + 1, 1) ** 2 - self.area_at(i, 1) ** 2)
            - self.spatial_step * g * self.theta * self.reach.bed_slope * (self.area_at(i + 1, 1) + self.area_at(i, 1))
            + self.theta * g * self.spatial_step * (self.Sf_at(i + 1, 1) * self.area_at(i + 1, 1) + self.Sf_at(i, 1) * self.area_at(i, 1))
            + self.num_celerity * (self.flow_at(i + 1, 1) + self.flow_at(i, 1))
            + 2 * self.theta * (
                self.flow_at(i + 1, 1) ** 2 / self.area_at(i + 1, 1) - self.flow_at(i, 1) ** 2 / self.area_at(i, 1)
            )
            - (
                self.num_celerity * (self.flow_at(i + 1, 0) + self.flow_at(i, 0))
                - 2 * (1 - self.theta) * (
                    self.flow_at(i + 1, 0) ** 2 / self.area_at(i + 1, 0)
                    + (0.5 * g / self.reach.width) * self.area_at(i + 1, 0) ** 2
                    - self.flow_at(i, 0) ** 2 / self.area_at(i, 0)
                    - (0.5 * g / self.reach.width) * self.area_at(i, 0) ** 2
                )
                + self.spatial_step * (1 - self.theta) * g * (
                    self.area_at(i + 1, 0) * (self.reach.bed_slope - self.Sf_at(i + 1, 0))
                    + self.area_at(i, 0) * (self.reach.bed_slope - self.Sf_at(i, 0))
                )
            )
        )

        return residual

    def downstream_residual(self, time) -> float:
        """
        Computes the residual of the downstream boundary condition equation.

        Returns
        -------
        float
            The computed residual.

        """
        depth = self.area_at(-1, 1) / self.reach.width
        
        residual = self.reach.downstream_boundary.condition_residual(time, depth, self.reach.width, self.flow_at(-1, 1),
                                                                     self.reach.bed_slope, self.reach.manning_co, self.time_step)
        
        return residual

    def storage_residual(self):
        average_inflow = 0.5 * (self.flow_at(-1, 1) + self.flow_at(-1, 0))
        average_stage = 0.5 * (self.reach.downstream_boundary.storage_stage + self.area_at(-1, 0)/float(self.reach.width) + self.reach.downstream_boundary.bed_level)
        
        new_storage_stage = self.reach.downstream_boundary.mass_balance(self.time_step, average_inflow, average_stage)
        
        residual = self.reach.downstream_boundary.storage_stage - new_storage_stage
        
        return residual
    
    def dU_dA(self, time, reg = False) -> int:
        """
        Computes the derivative of the upstream BC residual w.r.t. flow area.

        Returns
        -------
        float
            dU/dA

        """
        dU = self.reach.upstream_boundary.df_dA(time, self.area_at(0, 1), self.reach.width,
                                                self.reach.bed_slope, self.reach.manning_co)
        if not self.enforce_physicality:
            derivative = dU
            
        else:
            dU_dAreg = dU            
            dAreg_dA = self.dAreg_dA(0)
            
            dU_dQe = self.reach.upstream_boundary.df_dQ()
            dQe_dA = self.dQe_dA(0)
            
            derivative = dU_dAreg * dAreg_dA + dU_dQe * dQe_dA
        
        if reg:
            return dU_dAreg
        else:
            return derivative
    
    def dU_dQ(self, eff = False) -> int:
        """
        Computes the derivative of the upstream BC residual w.r.t. flow rate.

        Returns
        -------
        float
            dU/dQ

        """
        dU = self.reach.upstream_boundary.df_dQ()
        
        if not self.enforce_physicality:
            derivative = dU
            
        else:
            dU_dQe = dU            
            dQe_dQ = self.dQe_dQ(0)
            
            derivative = dU_dQe * dQe_dQ
    
        if eff:
            return dU_dQe
        else:
            return derivative

    def dC_dAiplus1(self, i, reg = False) -> float:
        """
        Computes the derivative of the continuity residual w.r.t. flow area of the advanced spatial node.

        Returns
        -------
        float
            dC/dA_(i+1)

        """
        dC = self.num_celerity
        
        if not self.enforce_physicality:
            derivative = dC
            
        else:
            dC_dAreg = dC            
            dAreg_dA = self.dAreg_dA(i + 1)
            
            dC_dQe = self.dC_dQiplus1(i, eff = True)
            dQe_dA = self.dQe_dA(i + 1)
            
            derivative = dC_dAreg * dAreg_dA + dC_dQe * dQe_dA
        
        if reg:
            return dC_dAreg
        else:
            return derivative

    def dC_dAi(self, i, reg = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dC = self.num_celerity
        
        if not self.enforce_physicality:
            derivative = dC
            
        else:
            dC_dAreg = dC            
            dAreg_dA = self.dAreg_dA(i)
            
            dC_dQe = self.dC_dQi(i, eff = True)
            dQe_dA = self.dQe_dA(i)
            
            derivative = dC_dAreg * dAreg_dA + dC_dQe * dQe_dA
        
        if reg:
            return dC_dAreg
        else:
            return derivative

    def dC_dQiplus1(self, i, eff = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dC = 2 * self.theta
        
        if not self.enforce_physicality:
            derivative = dC
            
        else:
            dC_dQe = dC            
            dQe_dQ = self.dQe_dQ(i + 1)
            
            derivative = dC_dQe * dQe_dQ
    
        if eff:
            return dC_dQe
        else:
            return derivative

    def dC_dQi(self, i, eff = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dC = -2 * self.theta
        
        if not self.enforce_physicality:
            derivative = dC
            
        else:
            dC_dQe = dC            
            dQe_dQ = self.dQe_dQ(i)
            
            derivative = dC_dQe * dQe_dQ
    
        if eff:
            return dC_dQe
        else:
            return derivative

    def dM_dAiplus1(self, i, reg = False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(i + 1, 1)
        Q = self.flow_at(i + 1, 1)
        
        Sf = self.reach.friction_slope(A, Q)
        dSf_dA = self.reach.friction_slope_deriv_A(A, Q)
        
        dM = (
            2 * g * self.theta / self.reach.width * A
            - self.spatial_step * g * self.theta * self.reach.bed_slope
            - 2 * self.theta * (Q / A) ** 2
            + self.theta * g * self.spatial_step * (Sf + A * dSf_dA)
            )
        
        if not self.enforce_physicality:
            derivative = dM
            
        else:
            dM_dAreg = dM           
            dAreg_dA = self.dAreg_dA(i + 1)
            
            dM_dQe = self.dM_dQiplus1(i, eff = True)
            dQe_dA = self.dQe_dA(i + 1)
            
            derivative = dM_dAreg * dAreg_dA + dM_dQe * dQe_dA
        
        if reg:
            return dM_dAreg
        else:
            return derivative

    def dM_dAi(self, i, reg = False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        
        A = self.area_at(i, 1)
        Q = self.flow_at(i, 1)
        
        Sf = self.reach.friction_slope(A, Q)
        dSf_dA = self.reach.friction_slope_deriv_A(A, Q)
        
        dM = (
            - 2 * g * self.theta / self.reach.width * A
            - self.spatial_step * g * self.theta * self.reach.bed_slope
            + 2 * self.theta * (Q / A) ** 2
            + self.theta * g * self.spatial_step * (Sf + A * dSf_dA)
            )
        
        if not self.enforce_physicality:
            dM_dA = dM
            
        else:
            dM_dAreg = dM            
            dAreg_dA = self.dAreg_dA(i)
            
            dM_dQe = self.dM_dQi(i, eff = True)
            dQe_dA = self.dQe_dA(i)
            
            dM_dA = dM_dAreg * dAreg_dA + dM_dQe * dQe_dA
        
        if reg:
            return dM_dAreg
        else:
            return dM_dA

    def dM_dQiplus1(self, i, eff = False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dSf_dQ = self.reach.friction_slope_deriv_Q(self.area_at(i + 1, 1), self.flow_at(i + 1, 1))
        
        dM = (
            self.num_celerity
            + 4 * self.theta * self.flow_at(i + 1, 1) / self.area_at(i + 1, 1)
            + self.theta * g * self.spatial_step * self.area_at(i + 1, 1) * dSf_dQ
            )
        
        if not self.enforce_physicality:
            derivative = dM
            
        else:
            dM_dQe = dM            
            dQe_dQ = self.dQe_dQ(i + 1)
            
            derivative = dM_dQe * dQe_dQ
    
        if eff:
            return dM_dQe
        else:
            return derivative

    def dM_dQi(self, i, eff = False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dSf_dQ = self.reach.friction_slope_deriv_Q(self.area_at(i, 1), self.flow_at(i, 1))
        
        dM = (
            self.num_celerity
            - 4 * self.theta * self.flow_at(i, 1) / self.area_at(i, 1)
            + self.theta * g * self.spatial_step * self.area_at(i, 1) * dSf_dQ
            )
        
        if not self.enforce_physicality:
            derivative = dM
            
        else:
            dM_dQe = dM            
            dQe_dQ = self.dQe_dQ(i)
            
            derivative = dM_dQe * dQe_dQ
    
        if eff:
            return dM_dQe
        else:
            return derivative

    def dD_dA(self, time, reg = False) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        dD = self.reach.downstream_boundary.df_dA(time, self.area_at(-1, 1), self.reach.width,
                                                  self.reach.bed_slope, self.reach.manning_co)
        if not self.enforce_physicality:
            derivative = dD
            
        else:
            dD_dAreg = dD            
            dAreg_dA = self.dAreg_dA(-1)
            
            dD_dQe = self.reach.downstream_boundary.df_dQ()
            dQe_dA = self.dQe_dA(-1)
            
            derivative = dD_dAreg * dAreg_dA + dD_dQe * dQe_dA
        
        if reg:
            return dD_dAreg
        else:
            return derivative

    def dD_dQ(self, eff = False):
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        dD = self.reach.downstream_boundary.df_dQ()
        
        if not self.enforce_physicality:
            derivative = dD
            
        else:
            dD_dQe = dD            
            dQe_dQ = self.dQe_dQ(-1)
            
            derivative = dD_dQe * dQe_dQ
    
        if eff:
            return dD_dQe
        else:
            return derivative
    
    def dD_dhs(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the storage depth.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.reach.downstream_boundary.condition_derivative_wrt_res_h()

        return derivative
            
    def dS_dQ(self, eff = False):
        average_inflow = 0.5 * (self.flow_at(-1, 1) + self.flow_at(-1, 0))
        average_stage = 0.5 * (self.reach.downstream_boundary.storage_stage + self.area_at(-1, 0)/float(self.reach.width) + self.reach.downstream_boundary.bed_level)
        
        dhs = 0 - 0.5 * self.reach.downstream_boundary.mass_balance_deriv_wrt_Q(self.time_step, average_inflow, average_stage)
        
        if not self.enforce_physicality:
            derivative = dhs
            
        else:
            dhs_dQe = dhs
            dQe_dQ = self.dQe_dQ(-1)
            
            derivative = dhs_dQe * dQe_dQ
    
        if eff:
            return dhs_dQe
        else:
            return derivative
    
    def dS_dhs(self):
        average_inflow = 0.5 * (self.flow_at(-1, 1) + self.flow_at(-1, 0))
        average_stage = 0.5 * (self.reach.downstream_boundary.storage_stage + self.area_at(-1, 0)/float(self.reach.width) + self.reach.downstream_boundary.bed_level)
        
        derivative = 1 - 0.5 * self.reach.downstream_boundary.mass_balance_deriv_wrt_stage(self.time_step, average_inflow, average_stage)
        
        return derivative
    
    def log_status(self):
        Utility.create_directory_if_not_exists('error_log')
        
        with open('error_log//status.txt', 'w') as output_file:
            output_file.write(f'Spatial step = {self.spatial_step} m\n')
            output_file.write(f'Time step = {self.time_step} s\n')
            output_file.write(f'Theta = {self.theta}\n')
            
            output_file.write(f'Channel length = {self.reach.total_length}\n')
            output_file.write(f'Channel width = {self.reach.width}\n')
            output_file.write(f'Manning\'s coefficient = {self.reach.manning_co}\n')
            output_file.write(f'Bed slope = {self.reach.bed_slope}\n')
            
            output_file.write('\n##############################################################\n\n')
            
            output_file.write(f'Upstream boundary:\n{self.reach.upstream_boundary.status()}\n')
            
            output_file.write('\n##############################################################\n\n')
            
            output_file.write(f'Downstream boundary:\n{self.reach.downstream_boundary.status()}\n')
            
            output_file.write('\n##############################################################\n\n')
            
            output_file.write('Initial conditions:\n')
            for a, q in self.reach.initial_conditions:
                output_file.write(f'A = {a:.2f}, Q = {q:.2f} m^3/s\n')
            
        self.save_results(path='error_log')
            
    def dAreg_dA(self, i, eps=1e-4):
        """
        Derivative of regularized area w.r.t. raw area.
        
        Parameters
        ----------
        i : int
            Index of spatial node.
        eps : float
            Smoothing parameter.

        Returns
        -------
        float
            d(A_reg)/dA
            
        """
        h_min = 1e-4
        A_min = self.reach.width * h_min
        A = self.area_at(i, 1, False)
        
        return 0.5 * (
            1.0 + (A - A_min) / np.sqrt(
                (A - A_min) ** 2 + eps ** 2
                )
            )
        
    def dQe_dA(self, i):
        """
        Derivative of effective (scaled) flow rate w.r.t. raw area.
        
        Parameters
        ----------
        i : int
            Index of spatial node.
        eps : float
            Smoothing parameter.

        Returns
        -------
        float
            d(Q_e)/dA
            
        """
        h_min = 1e-4
        
        A_min = self.reach.width * h_min
        A_reg = self.area_at(i, 1, True)
        Q = self.flow_at(i, 1, False)
        
        return A_min * Q * self.dAreg_dA(i) / (A_reg + A_min) ** 2
        
    def dQe_dQ(self, i):
        """
        Derivative of regularized area w.r.t. raw area.
        
        Parameters
        ----------
        i : int
            Index of spatial node.

        Returns
        -------
        float
            d(A_reg)/dA
            
        """
        h_min = 1e-4
        
        A_min = self.reach.width * h_min
        A_reg = self.area_at(i, True, True)
        
        chi = A_reg / (A_reg + A_min)
        
        return chi
    