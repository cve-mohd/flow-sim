from solver import Solver
from river import River
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
                 river: River,
                 theta: int | float,
                 time_step: int | float,
                 spatial_step: int | float,
                 fit_spatial_step = True):
        """
        Initializes the class.

        Parameters
        ----------
        river : River
            The River object on which the simulation is performed.
        theta : float
            The weighting factor of the Preissmann scheme.
        time_step : float
            Time step for the simulation in seconds.
        spatial_step : float
            Spatial step for the simulation in meters.
            
        """
        super().__init__(river, time_step, spatial_step, fit_spatial_step)
        
        self.theta = theta
        self.Sf_previous = []
        self.unknowns = []
        
        self.active_storage = ((self.river.downstream_boundary.condition == 'fixed_depth') and (self.river.downstream_boundary.reservoir_exit_rating_curve is not None))

        self.initialize_t0()


    def initialize_t0(self) -> None:
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
            self.unknowns += [A, Q]

            self.A_previous.append(A)
            self.Q_previous.append(Q)
            
            # Calculate the friction slope at all nodes at the previous (first) time step
            self.Sf_previous.append(self.river.friction_slope(A, Q))

        if self.active_storage:
            self.unknowns.append(self.river.downstream_boundary.storage_stage)
            
        # Convert the list of unknowns to a NumPy array.
        self.unknowns = np.array(self.unknowns)

        # Store the computed values of A and Q in the results list.
        self.computed_areas.append(self.A_previous)
        self.computed_flow_rates.append(self.Q_previous)

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
        equation_list = [self.upstream_eq(time)]
        
        # Add the continuity and momentum residuals for all reaches.
        for i in range(self.number_of_nodes - 1):
            equation_list.append(self.continuity_eq(i))
            equation_list.append(self.momentum_eq(i))

        # Lastly, add the downstream boundary condition residuals.
        equation_list.append(self.downstream_eq())
        
        if self.active_storage:
            equation_list.append(self.storage_eq())
            
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

        # Declare a 2N by 2N matrix, where N is the number of nodes along the river,
        # and initialize all elements to zeros.
        if self.active_storage:
            matrix_shape = (2 * self.number_of_nodes + 1, 2 * self.number_of_nodes + 1)
        else:
            matrix_shape = (2 * self.number_of_nodes, 2 * self.number_of_nodes)
            
        jacobian_matrix = np.zeros(shape=matrix_shape)

        # Compute the derivatives of the upstream boundary condition with respect to A and Q
        # at the first node, and assign the computed values to the first 2 elements of the first row.
        jacobian_matrix[0, 0] = self.upstream_deriv_A(t)
        jacobian_matrix[0, 1] = self.upstream_deriv_Q()

        # The loop computes the derivatives of the continuity equation with respect to A and Q at each of the
        # ith and (i+1)th node, and stores their values in the next row The same is done for the momentum
        # equation. The derivatives are placed in their appropriate positions along the matrix diagonal.
        # Alternate between the continuity and momentum equation until the second to last row.
        for row in range(1, 2 * self.number_of_nodes - 1, 2):
            jacobian_matrix[row, row - 1] = self.continuity_deriv_Ai()
            jacobian_matrix[row, row + 0] = self.continuity_deriv_Qi()
            jacobian_matrix[row, row + 1] = self.continuity_deriv_Ai_plus1()
            jacobian_matrix[row, row + 2] = self.continuity_deriv_Qi_plus1()

            jacobian_matrix[row + 1, row - 1] = self.momentum_deriv_Ai( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 0] = self.momentum_deriv_Qi( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 1] = self.momentum_deriv_Ai_plus1( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 2] = self.momentum_deriv_Qi_plus1( (row - 1) // 2 )

        # Lastly, compute the derivatives of the downstream boundary condition with respect to A and Q
        # at the last node, and place their values in the last 2 places of the last row.
        if self.active_storage:
            jacobian_matrix[-2, -3] = self.downstream_deriv_A(t)
            jacobian_matrix[-2, -2] = self.downstream_deriv_Q()
            jacobian_matrix[-2, -1] = self.downstream_deriv_res_h()
            
            jacobian_matrix[-1, -3] = self.storage_eq_deriv_A()
            jacobian_matrix[-1, -2] = self.storage_eq_deriv_Q()
            jacobian_matrix[-1, -1] = self.storage_eq_deriv_res_h()
        else:
            jacobian_matrix[-1, -2] = self.downstream_deriv_A(t)
            jacobian_matrix[-1, -1] = self.downstream_deriv_Q()

        return jacobian_matrix

    def run(self, duration: int, tolerance=1e-4, verbose=3) -> None:
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
        for time in range(self.time_step, duration + self.time_step, self.time_step):
            if verbose >= 1:
                print('\n---------- Time = ' + str(time) + 's ----------\n')

            iteration = 0

            while True:
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
                    raise ValueError("Solution failed.")

                # Solve the equation J * delta = -F to compute the delta vector.
                delta = np.linalg.solve(J, -R)

                # Improve the trial values using the computed delta.
                self.unknowns += delta

                # Compute the cumulative error as the Manhattan norm of delta.
                error = Utility.euclidean_norm(delta)
                
                if iteration > 100:
                    print(R)
                    raise ValueError("")

                if verbose >= 3:
                    print("Error = " + str(error))
                    
                # End the loop and move to the next time step if the cumulative error is smaller
                # than the specified tolerance. Otherwise, repeat the solution using the updated values.
                if error < tolerance:
                    break
                                    
            # Save the final values of the solved time step.
            self.append_result()

            # Update the values of the previous time step.
            self.update_parameters()

        self.solved = True
        
        if verbose >= 1:
            print("Simulation completed successfully.")
    
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
            self.Sf_previous = [self.river.friction_slope(A, Q) for A, Q in zip(self.A_previous, self.Q_previous)]
        else:
            self.A_previous = [i for i in self.unknowns[::2]]
            self.Q_previous = [i for i in self.unknowns[1::2]]
            self.Sf_previous = [self.river.friction_slope(A, Q) for A, Q in zip(self.A_previous, self.Q_previous)]
        

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
            self.river.downstream_boundary.storage_stage = self.unknowns[-1]
        else:
            self.A_current = self.unknowns[::2]
            self.Q_current = self.unknowns[1::2]
        
    def upstream_eq(self, t) -> float:
        """
        Computes the residual of the upstream boundary condition equation.

        Parameters
        ----------
        t : float
            Current simulation time in seconds.

        Returns
        -------
        float
            The computed residual.

        """
        y = float(self.A_current[0]) / self.river.width
        q = self.Q_current[0]
        
        U = self.river.upstream_bc(t, y, q)

        return U

    def continuity_eq(self, i) -> float:
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
        C = (
                self.num_celerity * (self.A_current[i] + self.A_current[i + 1])
                + 2 * self.theta * (self.Q_current[i + 1] - self.Q_current[i])
                - (
                        self.num_celerity * (self.A_previous[i + 1] + self.A_previous[i])
                        - 2 * (1 - self.theta) * (self.Q_previous[i + 1] - self.Q_previous[i])
                )
        )

        return C

    def momentum_eq(self, i) -> float:
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
        Sf_i = self.river.friction_slope(self.A_current[i], self.Q_current[i])
        Sf_iplus1 = self.river.friction_slope(self.A_current[i + 1], self.Q_current[i + 1])
                
        M = (
                (g * self.theta / self.river.width) * (self.A_current[i + 1] ** 2 - self.A_current[i] ** 2)
                - self.spatial_step * g * self.theta * self.river.bed_slope * (self.A_current[i + 1] + self.A_current[i])
                + self.theta * g * self.spatial_step * (Sf_iplus1 * self.A_current[i+1] + Sf_i * self.A_current[i])
                + self.num_celerity * (self.Q_current[i + 1] + self.Q_current[i])
                + 2 * self.theta * (
                        self.Q_current[i + 1] ** 2 / self.A_current[i + 1] - self.Q_current[i] ** 2 / self.A_current[i]
                )
                - (
                        self.num_celerity * (self.Q_previous[i + 1] + self.Q_previous[i])
                        - 2 * (1 - self.theta) * (
                                self.Q_previous[i + 1] ** 2 / self.A_previous[i + 1]
                                + (0.5 * g / self.river.width) * self.A_previous[i + 1] ** 2
                                - self.Q_previous[i] ** 2 / self.A_previous[i]
                                - (0.5 * g / self.river.width) * self.A_previous[i] ** 2
                        )
                        + self.spatial_step * (1 - self.theta) * g * (
                                self.A_previous[i + 1] * (self.river.bed_slope - self.Sf_previous[i + 1])
                                + self.A_previous[i] * (self.river.bed_slope - self.Sf_previous[i])
                        )
                )
        )

        return M

    def downstream_eq(self) -> float:
        """
        Computes the residual of the downstream boundary condition equation.

        Returns
        -------
        float
            The computed residual.

        """
        depth = float(self.A_current[-1]) / self.river.width
        inflow = self.Q_current[-1]
        
        D = self.river.downstream_bc(depth=depth, discharge=inflow, time_step=self.time_step)
        
        return D

    
    def upstream_deriv_A(self, t) -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the upstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.river.upstream_bc_deriv_A(t, self.A_current[0])

        return d

    
    def upstream_deriv_Q(self) -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the upstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.river.upstream_bc_deriv_Q()

        return d

    def continuity_deriv_Ai_plus1(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.num_celerity

        return d

    def continuity_deriv_Ai(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.num_celerity

        return d

    def continuity_deriv_Qi_plus1(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = 2 * self.theta

        return d

    def continuity_deriv_Qi(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = -2 * self.theta

        return d

    def momentum_deriv_Ai_plus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        Sf = self.river.friction_slope(self.A_current[i + 1], self.Q_current[i + 1])
        dSf_dA = self.river.friction_slope_deriv_A(self.A_current[i + 1], self.Q_current[i + 1])
        
        d = (
                2 * g * self.theta / self.river.width * self.A_current[i + 1]
                - self.spatial_step * g * self.theta * self.river.bed_slope
                - 2 * self.theta * (self.Q_current[i + 1] / self.A_current[i + 1]) ** 2
                + self.theta * g * self.spatial_step * (Sf + self.A_current[i + 1] * dSf_dA)
        )

        return d

    def momentum_deriv_Ai(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        Sf = self.river.friction_slope(self.A_current[i], self.Q_current[i])
        dSf_dA = self.river.friction_slope_deriv_A(self.A_current[i], self.Q_current[i])
        
        d = (
                - 2 * g * self.theta / self.river.width * self.A_current[i]
                - self.spatial_step * g * self.theta * self.river.bed_slope
                + 2 * self.theta * (self.Q_current[i] / self.A_current[i]) ** 2
                + self.theta * g * self.spatial_step * (Sf + self.A_current[i] * dSf_dA)
        )

        return d

    def momentum_deriv_Qi_plus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dSf_dQ = self.river.friction_slope_deriv_Q(self.A_current[i + 1], self.Q_current[i + 1])
                
        d = (
                self.num_celerity
                + 4 * self.theta * self.Q_current[i + 1] / self.A_current[i + 1]
                + self.theta * g * self.spatial_step * self.A_current[i + 1] * dSf_dQ
        )

        return d

    def momentum_deriv_Qi(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dSf_dQ = self.river.friction_slope_deriv_Q(self.A_current[i], self.Q_current[i])
        
        d = (
                self.num_celerity
                - 4 * self.theta * self.Q_current[i] / self.A_current[i]
                + self.theta * g * self.spatial_step * self.A_current[i] * dSf_dQ
        )

        return d

    def downstream_deriv_A(self, t) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.river.downstream_bc_deriv_A(t, self.A_current[-1])

        return d

    
    def downstream_deriv_Q(self):
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.river.downstream_bc_deriv_Q()

        return d
    
    def downstream_deriv_res_h(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the storage depth.

        Returns
        -------
        float
            The computed derivative.

        """
        d = self.river.downstream_bc_deriv_res_h()

        return d
    
    def storage_eq(self):
        s = self.river.downstream_boundary.bed_level + self.A_current[-1] / self.river.width
        new_storage_stage = self.river.downstream_boundary.mass_balance(self.Q_current[-1], self.time_step, s)
        
        residual = self.river.downstream_boundary.storage_stage - new_storage_stage
        return residual
        
    
    def storage_eq_deriv_A(self):
        s = self.river.downstream_boundary.bed_level + self.A_current[-1] / self.river.width
        d_storage_depth_dh = self.river.downstream_boundary.mass_balance_deriv_h(self.Q_current[-1], self.time_step, s)
        
        d = 0 - d_storage_depth_dh * 1./ self.river.width
        return d
    
    def storage_eq_deriv_Q(self):
        s = self.river.downstream_boundary.bed_level + self.A_current[-1] / self.river.width
        d_storage_depth_dQ = self.river.downstream_boundary.mass_balance_deriv_Q(self.Q_current[-1], self.time_step, s)
        
        d = 0 - d_storage_depth_dQ
        return d
    
    def storage_eq_deriv_res_h(self):
        d = 1
        return d
    