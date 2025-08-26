from solver import Solver
from reach import Channel
import numpy as np
from scipy.constants import g
from utility import Utility


class PreissmannSolver(Solver):
    """
    Implements the Preissmann implicit finite difference scheme to numerically
    solve the Saint-Venant equations.

    Attributes
    ----------
    theta : float
        The weighting factor of the Preissmann scheme (between 0.5 and 1).
    Sf_previous : list of float
        Friction slopes at the previous time step.
    unknowns : list of float
        Vector of unknowns for the current iteration (alternating A and Q).
    active_storage : bool
        Whether the downstream boundary condition is a storage-controlled Dirichlet stage.
            
    """

    def __init__(self,
                 channel: Channel,
                 theta: int | float,
                 time_step: int | float,
                 spatial_step: int | float,
                 fit_spatial_step = True):
        """
        Initializes the class.

        Parameters
        ----------
        channel : Channel
            The Channel object on which the simulation is performed.
        theta : float
            The weighting factor of the Preissmann scheme.
        time_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
            
        """
        super().__init__(channel, time_step, spatial_step, fit_spatial_step)
        
        self.theta = theta
        self.Sf_previous = []
        self.unknowns = []
        self.type = 'preissmann'
        
        self.initialize_t0()


    def initialize_t0(self) -> None:
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the channel.

        Returns
        -------
        None.

        """

        # Compute the initial conditions at all nodes in the 'Channel' object.
        self.channel.initialize_conditions(self.number_of_nodes)

        # Read the values of A and Q from the 'Channel' object and assign
        # them to the lists of unknowns, as well as the lists of A and Q at
        # the previous (first) time step.

        for A, Q in self.channel.initial_conditions:
            self.unknowns += [A, Q]

            self.A_previous.append(A)
            self.Q_previous.append(Q)
            
            # Calculate the friction slope at all nodes at the previous (first) time step
            self.Sf_previous.append(self.channel.friction_slope(A, Q))

        if self.active_storage:
            self.unknowns.append(self.channel.downstream_boundary.storage_stage)
            
        # Convert the list of unknowns to a NumPy array.
        self.unknowns = np.array(self.unknowns)

        # Store the computed values of A and Q in the results list.
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
            The vector of the residuals.
            
        """

        # Declare a list to store the residuals and add the
        # upstream boundary condition residuals as its 1st element.
        equation_list = [self.upstream_bc_residual(time)]
        
        # Add the continuity and momentum residuals for all reaches.
        for i in range(self.number_of_nodes - 1):
            equation_list.append(self.continuity_eq_residual(i))
            equation_list.append(self.momentum_eq_residual(i))

        # Lastly, add the downstream boundary condition residuals.
        equation_list.append(self.downstream_bc_residual(time=time))
        
        if self.active_storage:
            equation_list.append(self.storage_eq_residual())
            
        # Return the list as a NumPy array.
        return np.array(equation_list)

    def compute_jacobian(self, t) -> np.ndarray:
        """
        Constructs the Jacobian matrix of the system of equations.

        Returns
        -------
        np.ndarray
            The Jacobian matrix.
            
        """

        # Declare a 2N by 2N matrix, where N is the number of nodes along the channel,
        # and initialize all elements to zeros.
        if self.active_storage:
            matrix_shape = (2 * self.number_of_nodes + 1, 2 * self.number_of_nodes + 1)
        else:
            matrix_shape = (2 * self.number_of_nodes, 2 * self.number_of_nodes)
            
        jacobian_matrix = np.zeros(shape=matrix_shape)

        # Compute the derivatives of the upstream boundary condition with respect to A and Q
        # at the first node, and assign the computed values to the first 2 elements of the first row.
        jacobian_matrix[0, 0] = self.d_upstream_bc_residual_dA(t)
        jacobian_matrix[0, 1] = self.d_upstream_bc_residual_dQ()

        # The loop computes the derivatives of the continuity equation with respect to A and Q at each of the
        # ith and (i+1)th node, and stores their values in the next row The same is done for the momentum
        # equation. The derivatives are placed in their appropriate positions along the matrix diagonal.
        # Alternate between the continuity and momentum equation until the second to last row.
        for row in range(1, 2 * self.number_of_nodes - 1, 2):
            jacobian_matrix[row, row - 1] = self.d_continuity_eq_residual_dAi()
            jacobian_matrix[row, row + 0] = self.d_continuity_eq_residual_dQi()
            jacobian_matrix[row, row + 1] = self.d_continuity_eq_residual_dAi_plus1()
            jacobian_matrix[row, row + 2] = self.d_continuity_eq_residual_dQi_plus1()

            jacobian_matrix[row + 1, row - 1] = self.d_momentum_residual_dAi( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 0] = self.d_momentum_residual_dQi( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 1] = self.d_momentum_residual_dAi_plus1( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 2] = self.d_momentum_residual_dQi_plus1( (row - 1) // 2 )
        
        # Lastly, compute the derivatives of the downstream boundary condition with respect to A and Q
        # at the last node, and place their values in the last 2 places of the last row.
        if self.active_storage:        
            jacobian_matrix[-2, -3] = self.d_downstream_residual_dA(t)
            jacobian_matrix[-2, -2] = self.d_downstream_residual_dQ()
            jacobian_matrix[-2, -1] = self.d_downstream_residual_d_storage_stage()
            
            jacobian_matrix[-1, -2] = self.d_storage_residual_dQ()
            jacobian_matrix[-1, -1] = self.d_storage_residual_d_storage_stage()
        else:
            jacobian_matrix[-1, -2] = self.d_downstream_residual_dA(t)
            jacobian_matrix[-1, -1] = self.d_downstream_residual_dQ()

        return jacobian_matrix

    def run(self, duration: int, auto = True, tolerance=1e-4, verbose=3) -> None:
        """
        Solves the system of equations using the Newton-Raphson method, and stores
        the obtained values of the flow variables.

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

        # Loop through the time steps, incrementing the time by delta t every time.
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
                if verbose >= 2:
                    print("--- Iteration #" + str(iteration) + ':')

                # Update the trial values for the unknown variables.
                self.update_guesses()
                
                # Compute the residual vector.
                R = self.compute_residual_vector(time)
                                
                # Compute the Jacobian matrix.
                J = self.compute_jacobian(time)
                
                if np.isnan(R).any() or np.isnan(J).any():
                    self.log_status()
                    raise ValueError(f"Solution failed.\n")

                # Solve the equation J * delta = -F to compute the delta vector.
                delta = np.linalg.solve(J, -R)

                # Improve the trial values using the computed delta.
                self.unknowns += delta

                # Compute the cumulative error as the Manhattan norm of delta.
                error = Utility.euclidean_norm(delta)
                
                if verbose >= 3:
                    print("Error = " + str(error))
                    
                # End the loop and move to the next time step if the cumulative error is smaller
                # than the specified tolerance. Otherwise, repeat the solution using the updated values.
                if error < tolerance:
                    converged = True
                    if iteration == 1 and auto and time >= duration:
                        running = False
                                    
            # Save the final values of the solved time step.
            self.append_result()

            # Update the values of the previous time step.
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
            self.Sf_previous = [self.channel.friction_slope(A, Q) for A, Q in zip(self.A_previous, self.Q_previous)]
        else:
            self.A_previous = [i for i in self.unknowns[::2]]
            self.Q_previous = [i for i in self.unknowns[1::2]]
            self.Sf_previous = [self.channel.friction_slope(A, Q) for A, Q in zip(self.A_previous, self.Q_previous)]
        

    def update_guesses(self) -> None:
        """
        Updates the trial values of the unknown flow variables to the ones
        last computed.

        Returns
        -------
        None.

        """
        if self.active_storage:
            self.A_current = self.unknowns[:-1:2]
            self.Q_current = self.unknowns[1:-1:2]
            self.channel.downstream_boundary.storage_stage = self.unknowns[-1]
        else:
            self.A_current = self.unknowns[::2]
            self.Q_current = self.unknowns[1::2]
        
    def upstream_bc_residual(self, time) -> float:
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
        depth = float(self.A_current[0]) / self.channel.width
        flow = self.Q_current[0]
        
        residual = self.channel.upstream_boundary.condition_residual(time, depth, self.channel.width, flow, self.channel.bed_slope,
                                                            self.channel.manning_co, self.time_step)

        return residual

    def continuity_eq_residual(self, i) -> float:
        """
        Computes the residual of the continuity equation for a specific node.

        Parameters
        ----------
        i : int
            Index of the node.

        Returns
        -------
        float
            The computed residual.

        """
        residual = (
                self.num_celerity * (self.A_current[i] + self.A_current[i + 1])
                + 2 * self.theta * (self.Q_current[i + 1] - self.Q_current[i])
                - (
                        self.num_celerity * (self.A_previous[i + 1] + self.A_previous[i])
                        - 2 * (1 - self.theta) * (self.Q_previous[i + 1] - self.Q_previous[i])
                )
        )

        return residual

    def momentum_eq_residual(self, i) -> float:
        """
        Computes the residual of the momentum equation for a specific node.

        Parameters
        ----------
        i : int
            Index of the node.

        Returns
        -------
        float
            The computed residual.

        """
        Sf_i = self.channel.friction_slope(self.A_current[i], self.Q_current[i])
        Sf_iplus1 = self.channel.friction_slope(self.A_current[i + 1], self.Q_current[i + 1])
                
        residual = (
                (g * self.theta / self.channel.width) * (self.A_current[i + 1] ** 2 - self.A_current[i] ** 2)
                - self.spatial_step * g * self.theta * self.channel.bed_slope * (self.A_current[i + 1] + self.A_current[i])
                + self.theta * g * self.spatial_step * (Sf_iplus1 * self.A_current[i+1] + Sf_i * self.A_current[i])
                + self.num_celerity * (self.Q_current[i + 1] + self.Q_current[i])
                + 2 * self.theta * (
                        self.Q_current[i + 1] ** 2 / self.A_current[i + 1] - self.Q_current[i] ** 2 / self.A_current[i]
                )
                - (
                        self.num_celerity * (self.Q_previous[i + 1] + self.Q_previous[i])
                        - 2 * (1 - self.theta) * (
                                self.Q_previous[i + 1] ** 2 / self.A_previous[i + 1]
                                + (0.5 * g / self.channel.width) * self.A_previous[i + 1] ** 2
                                - self.Q_previous[i] ** 2 / self.A_previous[i]
                                - (0.5 * g / self.channel.width) * self.A_previous[i] ** 2
                        )
                        + self.spatial_step * (1 - self.theta) * g * (
                                self.A_previous[i + 1] * (self.channel.bed_slope - self.Sf_previous[i + 1])
                                + self.A_previous[i] * (self.channel.bed_slope - self.Sf_previous[i])
                        )
                )
        )

        return residual

    def downstream_bc_residual(self, time) -> float:
        """
        Computes the residual of the downstream boundary condition equation.

        Returns
        -------
        float
            The computed residual.

        """
        depth = float(self.A_current[-1]) / self.channel.width
        flow = self.Q_current[-1]
        
        residual = self.channel.downstream_boundary.condition_residual(time, depth, self.channel.width, flow,
                                                              self.channel.bed_slope, self.channel.manning_co,
                                                              self.time_step)
        
        return residual

    
    def d_upstream_bc_residual_dA(self, time) -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the upstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.channel.upstream_boundary.condition_derivative_wrt_A(time, self.A_current[0], self.channel.width,
                                                                    self.channel.bed_slope, self.channel.manning_co)

        return derivative

    
    def d_upstream_bc_residual_dQ(self) -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the upstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.channel.upstream_boundary.condition_derivative_wrt_Q()

        return derivative

    def d_continuity_eq_residual_dAi_plus1(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.num_celerity

        return derivative

    def d_continuity_eq_residual_dAi(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.num_celerity

        return derivative

    def d_continuity_eq_residual_dQi_plus1(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = 2 * self.theta

        return derivative

    def d_continuity_eq_residual_dQi(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = -2 * self.theta

        return derivative

    def d_momentum_residual_dAi_plus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        Sf = self.channel.friction_slope(self.A_current[i + 1], self.Q_current[i + 1])
        dSf_dA = self.channel.friction_slope_deriv_A(self.A_current[i + 1], self.Q_current[i + 1])
        
        derivative = (
                2 * g * self.theta / self.channel.width * self.A_current[i + 1]
                - self.spatial_step * g * self.theta * self.channel.bed_slope
                - 2 * self.theta * (self.Q_current[i + 1] / self.A_current[i + 1]) ** 2
                + self.theta * g * self.spatial_step * (Sf + self.A_current[i + 1] * dSf_dA)
        )

        return derivative

    def d_momentum_residual_dAi(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        Sf = self.channel.friction_slope(self.A_current[i], self.Q_current[i])
        dSf_dA = self.channel.friction_slope_deriv_A(self.A_current[i], self.Q_current[i])
        
        derivative = (
                - 2 * g * self.theta / self.channel.width * self.A_current[i]
                - self.spatial_step * g * self.theta * self.channel.bed_slope
                + 2 * self.theta * (self.Q_current[i] / self.A_current[i]) ** 2
                + self.theta * g * self.spatial_step * (Sf + self.A_current[i] * dSf_dA)
        )

        return derivative

    def d_momentum_residual_dQi_plus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dSf_dQ = self.channel.friction_slope_deriv_Q(self.A_current[i + 1], self.Q_current[i + 1])
                
        derivative = (
                self.num_celerity
                + 4 * self.theta * self.Q_current[i + 1] / self.A_current[i + 1]
                + self.theta * g * self.spatial_step * self.A_current[i + 1] * dSf_dQ
        )

        return derivative

    def d_momentum_residual_dQi(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dSf_dQ = self.channel.friction_slope_deriv_Q(self.A_current[i], self.Q_current[i])
        
        derivative = (
                self.num_celerity
                - 4 * self.theta * self.Q_current[i] / self.A_current[i]
                + self.theta * g * self.spatial_step * self.A_current[i] * dSf_dQ
        )

        return derivative

    def d_downstream_residual_dA(self, time) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.channel.downstream_boundary.condition_derivative_wrt_A(time, self.A_current[-1], self.channel.width,
                                                                               self.channel.bed_slope, self.channel.manning_co)

        return derivative

    def d_downstream_residual_dQ(self):
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.channel.downstream_boundary.condition_derivative_wrt_Q()

        return derivative
    
    def d_downstream_residual_d_storage_stage(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the storage depth.

        Returns
        -------
        float
            The computed derivative.

        """
        derivative = self.channel.downstream_boundary.condition_derivative_wrt_res_h()

        return derivative
    
    def storage_eq_residual(self):
        average_inflow = 0.5 * (self.Q_current[-1] + self.Q_previous[-1])
        average_stage = 0.5 * (self.channel.downstream_boundary.storage_stage + self.A_previous[-1]/float(self.channel.width) + self.channel.downstream_boundary.bed_level)
        
        new_storage_stage = self.channel.downstream_boundary.mass_balance(self.time_step, average_inflow, average_stage)
        
        residual = self.channel.downstream_boundary.storage_stage - new_storage_stage
        
        return residual
            
    def d_storage_residual_dQ(self):
        average_inflow = 0.5 * (self.Q_current[-1] + self.Q_previous[-1])
        average_stage = 0.5 * (self.channel.downstream_boundary.storage_stage + self.A_previous[-1]/float(self.channel.width) + self.channel.downstream_boundary.bed_level)
        
        derivative = 0 - 0.5 * self.channel.downstream_boundary.mass_balance_deriv_wrt_Q(self.time_step, average_inflow, average_stage)
        
        return derivative
    
    def d_storage_residual_d_storage_stage(self):
        average_inflow = 0.5 * (self.Q_current[-1] + self.Q_previous[-1])
        average_stage = 0.5 * (self.channel.downstream_boundary.storage_stage + self.A_previous[-1]/float(self.channel.width) + self.channel.downstream_boundary.bed_level)
        
        derivative = 1 - 0.5 * self.channel.downstream_boundary.mass_balance_deriv_wrt_stage(self.time_step, average_inflow, average_stage)
        
        return derivative
    
    def log_status(self):
        Utility.create_directory_if_not_exists('error_log')
        
        with open('error_log//status.txt', 'w') as output_file:
            output_file.write(f'Spatial step = {self.spatial_step} m\n')
            output_file.write(f'Time step = {self.time_step} s\n')
            output_file.write(f'Theta = {self.theta}\n')
            
            output_file.write(f'Channel length = {self.channel.total_length}\n')
            output_file.write(f'Channel width = {self.channel.width}\n')
            output_file.write(f'Manning\'s coefficient = {self.channel.manning_co}\n')
            output_file.write(f'Bed slope = {self.channel.bed_slope}\n')
            
            output_file.write('\n##############################################################\n\n')
            
            output_file.write(f'Upstream boundary:\n{self.channel.upstream_boundary.status()}\n')
            
            output_file.write('\n##############################################################\n\n')
            
            output_file.write(f'Downstream boundary:\n{self.channel.downstream_boundary.status()}\n')
            
            output_file.write('\n##############################################################\n\n')
            
            output_file.write('Initial conditions:')
            for a, q in self.channel.initial_conditions:
                output_file.write(f'A = {a:.2f}, Q = {q:.2f} m^3/s\n')
            
        self.save_results(path='error_log')
            