from river import River
import numpy as np
from scipy.constants import g
from utility import Utility
from settings import USE_GVF


class PreissmannModel:
    """
    Implements the Preissmann implicit finite difference scheme to numerically
    solve the Saint-Venant equations.

    Attributes
    ----------
    river : River
        An instance of the `River` class, representing the river being modeled.
    theta : float
        The weighting factor of the Preissmann scheme (between 0.5 and 1).
    delta_t : float
        Time step for the simulation in seconds.
    delta_x : float
        Spatial step for the simulation in meters.
    celerity : float
        Ratio of spatial to time step, representing the wave celerity.
    n_nodes : int
        Number of spatial nodes along the river.
    A_previous : list of float
        Cross-sectional areas at the previous time step.
    Q_previous : list of float
        Discharges at the previous time step.
    Sf_previous : list of float
        Friction slopes at the previous time step.
    A_current : list of float
        Cross-sectional areas at the current iteration of the current time step.
    Q_current : list of float
        Discharges at the current iteration of the current time step.
    unknowns : list of float
        Vector of unknowns for the current iteration (alternating A and Q).
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
                 theta: int | float,
                 delta_t: int | float,
                 delta_x: int | float,
                 fit_delta_x = True):
        """
        Initializes the class.

        Parameters
        ----------
        river : River
            The River object on which the simulation is performed.
        theta : float
            The weighting factor of the Preissmann scheme.
        delta_t : float
            Time step for the simulation in seconds.
        delta_x : float
            Spatial step for the simulation in meters.
            
        """
        self.solved = False
        
        # Initialize the scheme discretization parameters.
        self.theta = theta
        self.delta_t, self.delta_x = delta_t, delta_x
        self.celerity = self.delta_x / float(self.delta_t)

        # Inizialize the river attributes.
        self.river = river
        self.W = self.river.width
        self.S_0 = self.river.bed_slope
        
        if fit_delta_x:
            self.n_nodes = int(round(self.river.total_length / self.delta_x) + 1)
            self.delta_x = self.river.total_length / (self.n_nodes - 1)
        else:
            self.n_nodes = int(self.river.total_length // self.delta_x + 1)

        # Declare empty lists for the flow variables at the previous time step, j.
        self.A_previous = []
        self.Q_previous = []
        self.Sf_previous = []

        # Declare empty lists for the flow variables at the advanced time step, j + 1.
        self.A_current = []
        self.Q_current = []

        # Declare an empty list to represent the vector of unknowns, X.
        self.unknowns = []

        # Declare an empty list to store the simulation results.
        self.computed_areas = []
        self.computed_flow_rates = []

        # Read the initial conditions of the river.
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
        if USE_GVF:
            self.river.initialize_conditions_gvh(self.n_nodes)
        else:
            self.river.initialize_conditions(self.n_nodes)

        # Read the values of A and Q from the 'River' object and assign
        # them to the lists of unknowns, as well as the lists of A and Q at
        # the previous (first) time step.

        H = []
        for A, Q in self.river.initial_conditions:
            self.unknowns += [A, Q]

            self.A_previous.append(A)
            self.Q_previous.append(Q)
            
            # Calculate the friction slope at all nodes at the previous (first) time step
            self.Sf_previous.append(self.river.friction_slope(A, Q))

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
        for i in range(self.n_nodes - 1):
            equation_list.append(self.continuity_eq(i))
            equation_list.append(self.momentum_eq(i))

        # Lastly, add the downstream boundary condition residuals.
        equation_list.append(self.downstream_eq(time))
            
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
        jacobian_matrix = np.zeros(shape=(2 * self.n_nodes, 2 * self.n_nodes))

        # Compute the derivatives of the upstream boundary condition with respect to A and Q
        # at the first node, and assign the computed values to the first 2 elements of the first row.
        jacobian_matrix[0, 0] = self.upstream_deriv_A(t)
        jacobian_matrix[0, 1] = self.upstream_deriv_Q()

        # The loop computes the derivatives of the continuity equation with respect to A and Q at each of the
        # ith and (i+1)th node, and stores their values in the next row The same is done for the momentum
        # equation. The derivatives are placed in their appropriate positions along the matrix diagonal.
        # Alternate between the continuity and momentum equation until the second to last row.
        for row in range(1, 2 * self.n_nodes - 1, 2):
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
        jacobian_matrix[-1, -2] = self.downstream_deriv_A(t)
        jacobian_matrix[-1, -1] = self.downstream_deriv_Q()

        return jacobian_matrix

    def solve(self, duration: int, tolerance=1e-4, verbose=3) -> None:
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
        for time in range(self.delta_t, duration + self.delta_t, self.delta_t):
            if verbose >= 1:
                print('\n---------- Time = ' + str(time) + 's ----------\n')

            iteration = 0

            while 1:
                iteration += 1
                if verbose >= 2:
                    print("--- Iteration #" + str(iteration) + ':')

                # Update the trial values for the unknown variables.
                self.update_guesses()

                # Compute the residual vector.
                R = self.compute_residual_vector(time)
                
                # Compute the Jacobian matrix.
                J = self.compute_jacobian(time)

                # Solve the equation J * delta = -F to compute the delta vector.
                delta = np.linalg.solve(J, -R)

                # Improve the trial values using the computed delta.
                self.unknowns += delta

                # Compute the cumulative error as the Manhattan norm of delta.
                error = Utility.euclidean_norm(delta)

                if verbose >= 3:
                    print("Error = " + str(error))
                    
                if np.isnan(R).any() or np.isnan(J).any():
                    raise ValueError("gr")

                # End the loop and move to the next time step if the cumulative error is smaller
                # than the specified tolerance. Otherwise, repeat the solution using the updated values.
                if error < tolerance:
                    break
                
            
            # Save the final values of the solved time step.
            self.append_results()

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
        y = float(self.A_current[0]) / self.W
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
                self.celerity * (self.A_current[i] + self.A_current[i + 1])
                + 2 * self.theta * (self.Q_current[i + 1] - self.Q_current[i])
                - (
                        self.celerity * (self.A_previous[i + 1] + self.A_previous[i])
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
                (g * self.theta / self.W) * (self.A_current[i + 1] ** 2 - self.A_current[i] ** 2)
                - self.delta_x * g * self.theta * self.S_0 * (self.A_current[i + 1] + self.A_current[i])
                + self.theta * g * self.delta_x * (Sf_iplus1 * self.A_current[i+1] + Sf_i * self.A_current[i])
                + self.celerity * (self.Q_current[i + 1] + self.Q_current[i])
                + 2 * self.theta * (
                        self.Q_current[i + 1] ** 2 / self.A_current[i + 1] - self.Q_current[i] ** 2 / self.A_current[i]
                )
                - (
                        self.celerity * (self.Q_previous[i + 1] + self.Q_previous[i])
                        - 2 * (1 - self.theta) * (
                                self.Q_previous[i + 1] ** 2 / self.A_previous[i + 1]
                                + (0.5 * g / self.W) * self.A_previous[i + 1] ** 2
                                - self.Q_previous[i] ** 2 / self.A_previous[i]
                                - (0.5 * g / self.W) * self.A_previous[i] ** 2
                        )
                        + self.delta_x * (1 - self.theta) * g * (
                                self.A_previous[i + 1] * (self.S_0 - self.Sf_previous[i + 1])
                                + self.A_previous[i] * (self.S_0 - self.Sf_previous[i])
                        )
                )
        )

        return M

    def downstream_eq(self, t) -> float:
        """
        Computes the residual of the downstream boundary condition equation.

        Returns
        -------
        float
            The computed residual.

        """
        y = float(self.A_current[-1]) / self.W
        q = self.Q_current[-1]
        
        D = self.river.downstream_bc(t, y, q)
        
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
        d = self.celerity

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
        d = self.celerity

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
                2 * g * self.theta / self.W * self.A_current[i + 1]
                - self.delta_x * g * self.theta * self.S_0
                - 2 * self.theta * (self.Q_current[i + 1] / self.A_current[i + 1]) ** 2
                + self.theta * g * self.delta_x * (Sf + self.A_current[i + 1] * dSf_dA)
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
                - 2 * g * self.theta / self.W * self.A_current[i]
                - self.delta_x * g * self.theta * self.S_0
                + 2 * self.theta * (self.Q_current[i] / self.A_current[i]) ** 2
                + self.theta * g * self.delta_x * (Sf + self.A_current[i] * dSf_dA)
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
                self.celerity
                + 4 * self.theta * self.Q_current[i + 1] / self.A_current[i + 1]
                + self.theta * g * self.delta_x * self.A_current[i + 1] * dSf_dQ
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
                self.celerity
                - 4 * self.theta * self.Q_current[i] / self.A_current[i]
                + self.theta * g * self.delta_x * self.A_current[i] * dSf_dQ
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


    def append_results(self):
        self.computed_areas.append(self.A_current.tolist())
        self.computed_flow_rates.append(self.Q_current.tolist())        
        
    def save_results(self, size: tuple, path: str = None) -> None:
        """
        Saves the results of the simulation in four .csv files, containing
        the computed cross-sectional flow area, discharge, flow depth, and velocity.
        The files are formatted with each row representing a time step and each
        column representing a spatial point.

        Parameters
        ----------
        size : tuple of float
            The number of temporal and spatial nodes to save.

        Returns
        -------
        None.

        """
        
        if path is None:
            path = 'Results//Preissmann'
            
        Utility.create_directory_if_not_exists(path)
        
        t_step = x_step = 1
        
        if size[0] > 1:
            t_step = (len(self.computed_areas) - 1) // (size[0] - 1)

        if size[1] > 1:
            x_step =  (self.n_nodes - 1) // (size[1] - 1)
                    
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
        depths = np.array(self.computed_areas) / self.W
        depths = depths.tolist()
        
        levels = []
        for sublist in depths:
            levels.append([sublist[i] + self.river.upstream_boundary.bed_level - self.S_0 * self.delta_x * i for i in range(len(sublist))])
            
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
        
        header = [self.river.upstream_boundary.chainage + x * self.delta_x for x in range(0, self.n_nodes, x_step)]
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
                
    
    def get_results(self, parameter: str, spatial_node: int = None, temporal_node: int = None) -> tuple:
        """
        Returns the results of the simulation.

        Returns
        -------
        tuple
            A tuple containing the computed cross-sectional flow area, discharge,
            velocity, and flow depth.

        """
        if not self.solved:
            raise ValueError("Not solved yet.")
        
        reqursted = None

        if parameter == 'a':
            reqursted = self.computed_areas
            
        elif parameter == 'q':
            reqursted = self.computed_flow_rates
            
        elif parameter == 'v':
            velocities = np.array(self.computed_flow_rates) / np.array(self.computed_areas)
            reqursted = velocities.tolist()
        
        elif parameter == 'h':
            depths = np.array(self.computed_areas) / self.W
            reqursted = depths.tolist()
        
        else:
            raise ValueError("Invalid parameter. Choose between 'a', 'q', 'v', or 'h'.")
        
        reqursted = np.array(reqursted)
        
        if spatial_node is not None:
            if temporal_node is not None:
                return reqursted[temporal_node, spatial_node]
            else:
                return reqursted[:, spatial_node]
        
        if temporal_node is not None:
            return reqursted[temporal_node, :]
            
        return reqursted