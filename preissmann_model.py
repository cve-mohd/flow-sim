from river import River
import numpy as np
from scipy.constants import g


class PreissmannModel:
    """
    Implements the Preissmann implicit finite difference scheme to numerically
    solve the Saint-Venant equations.

    Attributes
    ----------
    river : River
        An instance of the `River` class, representing the river being modeled.
    beta : float
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
    results : list of list of float
        Stores the simulation results over time.
    S_h : float
        Slope due to backwater effects.
        
    """

    def __init__(self,
                 river: River,
                 beta: int | float,
                 delta_t: int | float,
                 delta_x: int | float):
        """
        Initializes the class.

        Parameters
        ----------
        river : River
            The River object on which the simulation is performed.
        beta : float
            The weighting factor of the Preissmann scheme.
        delta_t : float
            Time step for the simulation in seconds.
        delta_x : float
            Spatial step for the simulation in meters.
            
        """

        # Initialize the scheme discretization parameters.
        self.beta = beta
        self.delta_t, self.delta_x = delta_t, delta_x
        self.celerity = self.delta_x / float(self.delta_t)

        # Inizialize the river attributes.
        self.river = river
        self.W = self.river.width
        self.n = self.river.manning_co
        self.S_h = 0
        self.S_0 = self.river.bed_slope
        self.n_nodes = self.river.length // self.delta_x + 1

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
        self.results = []

        # Read the initial conditions of the river.
        self.initialize_t0()

        # Compute the slope due to backwater effects.
        self.backwater_effects_calc()

    def initialize_t0(self) -> None:
        """
        Retrieves the values of the initial values of the flow variables
        from the initial conditions of the river.

        Returns
        -------
        None.

        """

        # Compute the initial conditions at all nodes in the 'River' object.
        self.river.initialize_conditions(self.n_nodes)

        # Read the values of A and Q from the 'River' object and assign
        # them to the lists of unknowns, as well as the lists of A and Q at
        # the previous (first) time step.
        for A, Q in self.river.initial_conditions:
            self.unknowns += [A, Q]

            self.A_previous.append(A)
            self.Q_previous.append(Q)

            # Calculate the friction slope at all nodes at the previous (first) time step
            self.Sf_previous.append(self.river.friction_slope(A, Q))

        # Convert the list of unknowns to a NumPy array.
        self.unknowns = np.array(self.unknowns)

        # Store the computed values of A and Q in the results list.
        self.results.append(self.unknowns.tolist())

    def compute_system_matrix(self, time) -> np.ndarray:
        """
        Constructs the system of equations, F.

        Parameters
        ----------
        time : float
            The current time of the simulation in seconds.

        Returns
        -------
        np.ndarray
            The vector of the residuals.
            
        """

        # Declare a list to store the equations and add the upstream boundary
        # condition equation as its 1st element.
        equation_list = [self.upstream_eq(time)]

        # Add the continuity and momentum equations for all reaches.
        for i in range(self.n_nodes - 1):
            equation_list.append(self.continuity_eq(i))
            equation_list.append(self.momentum_eq(i))

        # Lastly, add the downstream boundary condition equation.
        equation_list.append(self.downstream_eq())

        # Return the list as a NumPy array.
        return np.array(equation_list)

    def compute_jacobian(self) -> np.ndarray:
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
        jacobian_matrix[0, 0] = self.derivative_upstream_A()
        jacobian_matrix[0, 1] = self.derivative_upstream_Q()

        # The loop computes the derivatives of the continuity equation with respect to A and Q at each of the
        # ith and (i+1)th node, and stores their values in the next row The same is done for the momentum
        # equation. The derivatives are placed in their appropriate positions along the matrix diagonal.
        # Alternate between the continuity and momentum equation until the second to last row.
        for row in range(1, 2 * self.n_nodes - 1, 2):
            jacobian_matrix[row, row - 1] = self.derivative_c_A_i()
            jacobian_matrix[row, row + 0] = self.derivative_c_Q_i()
            jacobian_matrix[row, row + 1] = self.derivative_c_A_iplus1()
            jacobian_matrix[row, row + 2] = self.derivative_c_Q_iplus1()

            jacobian_matrix[row + 1, row - 1] = self.derivative_m_A_i( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 0] = self.derivative_m_Q_i( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 1] = self.derivative_m_A_iplus1( (row - 1) // 2 )
            jacobian_matrix[row + 1, row + 2] = self.derivative_m_Q_iplus1( (row - 1) // 2 )

        # Lastly, compute the derivatives of the downstream boundary condition with respect to A and Q
        # at the last node, and place their values in the last 2 places of the last row.
        jacobian_matrix[-1, -2] = self.derivative_downstream_A()
        jacobian_matrix[-1, -1] = self.derivative_downstream_Q()

        return jacobian_matrix

    def solve(self, duration: int, tolerance=1e-4) -> None:
        """
        Solves the system of equations using the Newton-Raphson method, and stores
        the obtained values of the flow variables.

        Parameters
        ----------
        duration : int
            The simulation duration.
        tolerance : float, optional
            The allowed tolerance for the iterative process. The simulation iterates until the cumulative error
            falls below this value. The default is 1e-4.

        Returns
        -------
        None.

        """

        # Loop through the time steps, incrementing the time by delta t every time.
        for time in range(self.delta_t, duration + 1, self.delta_t):
            print('\n---------- Time = ' + str(time) + 's ----------\n')

            iteration = 0

            while 1:
                iteration += 1
                print("--- Iteration #" + str(iteration) + ':')

                # Update the trial values for the unknown variables.
                self.update_guesses()

                # Compute the vector of residuals.
                F = self.compute_system_matrix(time)

                # Compute the Jacobian matrix.
                J = self.compute_jacobian()

                # Solve the equation J * delta = -F to compute the delta vector.
                delta = np.linalg.solve(J, -F)

                # Improve the trial values using the computed delta.
                self.unknowns += delta

                # Compute the cumulative error as the sum of the absolute values of delta.
                cumulative_error = np.sum(np.abs(delta))

                print("Error = " + str(cumulative_error))

                # End the loop and move to the next time step if the cumulative error is smaller
                # than the allowed tolerance. Otherwise, repeat the solution using the updated values.
                if cumulative_error < tolerance:
                    break

            # noinspection PyUnresolvedReferences
            # Save the final values of the solved time step.
            self.results.append(self.unknowns.tolist())

            # Update the values of the previous time step.
            self.update_parameters()

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
        self.Sf_previous = [self.river.friction_slope(self.A_previous[i], self.Q_previous[i]) for i in
                            range(self.n_nodes)]

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
        U = self.Q_current[0] - River.inflow_Q(t)

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
                + 2 * self.beta * (self.Q_current[i + 1] - self.Q_current[i])
                - (
                        self.celerity * (self.A_previous[i + 1] + self.A_previous[i])
                        - 2 * (1 - self.beta) * (self.Q_previous[i + 1] - self.Q_previous[i])
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
        M = (
                (g * self.beta / self.W) * (self.A_current[i + 1] ** 2 - self.A_current[i] ** 2)
                - self.delta_x * g * self.beta * (self.S_0 - self.S_h) * (self.A_current[i + 1] + self.A_current[i])
                + self.celerity * (self.Q_current[i + 1] + self.Q_current[i])
                + 2 * self.beta * (
                        self.Q_current[i + 1] ** 2 / self.A_current[i + 1] - self.Q_current[i] ** 2 / self.A_current[i]
                )
                + self.delta_x * g * self.beta * self.W ** (4. / 3) * self.n ** 2 * (
                        self.Q_current[i + 1] ** 2 / self.A_current[i + 1] ** (7. / 3)
                        + self.Q_current[i] ** 2 / self.A_current[i] ** (7. / 3)
                )
                - (
                        self.celerity * (self.Q_previous[i + 1] + self.Q_previous[i])
                        - 2 * (1 - self.beta) * (
                                self.Q_previous[i + 1] ** 2 / self.A_previous[i + 1]
                                + (0.5 * g / self.W) * self.A_previous[i + 1] ** 2
                                - self.Q_previous[i] ** 2 / self.A_previous[i]
                                - (0.5 * g / self.W) * self.A_previous[i] ** 2
                        )
                        + self.delta_x * (1 - self.beta) * g * (
                                self.A_previous[i + 1] * (self.S_0 - self.Sf_previous[i + 1] - self.S_h)
                                + self.A_previous[i] * (self.S_0 - self.Sf_previous[i] - self.S_h)
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
        D = self.A_current[-1] / self.W - 7.5

        return D

    @staticmethod
    def derivative_upstream_A() -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the upstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = 0

        return d

    @staticmethod
    def derivative_upstream_Q() -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the upstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = 1

        return d

    def derivative_c_A_iplus1(self) -> float:
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

    def derivative_c_A_i(self) -> float:
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

    def derivative_c_Q_iplus1(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = 2 * self.beta

        return d

    def derivative_c_Q_i(self) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = -2 * self.beta

        return d

    def derivative_m_A_iplus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = (
                2 * g * self.beta / self.W * self.A_current[i + 1]
                - self.delta_x * g * self.beta * (self.S_0 - self.S_h)
                - 2 * self.beta * (self.Q_current[i + 1] / self.A_current[i + 1]) ** 2
                - (7. / 3 * self.delta_x * g * self.beta * self.n ** 2 * self.W ** (4. / 3)
                   * self.Q_current[i + 1] ** 2 / self.A_current[i + 1] ** (10. / 3))
        )

        return d

    def derivative_m_A_i(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = (
                - 2 * g * self.beta / self.W * self.A_current[i]
                - self.delta_x * g * self.beta * (self.S_0 - self.S_h)
                + 2 * self.beta * (self.Q_current[i] / self.A_current[i]) ** 2
                - (7. / 3 * self.delta_x * g * self.beta * self.n ** 2 * self.W ** (4. / 3)
                   * self.Q_current[i] ** 2 / self.A_current[i] ** (10. / 3))
        )

        return d

    def derivative_m_Q_iplus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = (
                self.celerity
                + 4 * self.beta * self.Q_current[i + 1] / self.A_current[i + 1]
                + (2 * self.delta_x * g * self.beta * self.n ** 2 * self.W ** (4. / 3)
                   * self.Q_current[i + 1] / self.A_current[i + 1] ** (7. / 3))
        )

        return d

    def derivative_m_Q_i(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d = (
                self.celerity
                - 4 * self.beta * self.Q_current[i] / self.A_current[i]
                + (2 * self.delta_x * g * self.beta * self.n ** 2 * self.W ** (4. / 3)
                   * self.Q_current[i] / self.A_current[i] ** (7. / 3))
        )

        return d

    def derivative_downstream_A(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = 1 / self.W

        return d

    @staticmethod
    def derivative_downstream_Q() -> int:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        d = 0

        return d

    def backwater_effects_calc(self) -> None:
        """
        Computes the slope due to backwater effects.

        Returns
        -------
        None.

        """
        S_f = self.Sf_previous[0]
        self.S_h = self.S_0 - S_f

    def save_results(self, size: tuple) -> None:
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
        t_step = 1
        x_step = 2
        
        if size[0] > 1:
            t_step = (len(self.results) - 1) // (size[0] - 1)

        if size[1] > 1:
            x_step =  2 * (self.n_nodes - 1) // (size[1] - 1)

        A = [x[::x_step]
             for x in self.results[::t_step]]

        Q = [x[1::x_step]
             for x in self.results[::t_step]]

        y, V = [], []

        for i in range(len(A)):
            y.append([a / self.W for a in A[i]])
            V.append([q / a for a, q in zip(A[i], Q[i])])

        A, Q, y, V = str(A), str(Q), str(y), str(V)

        A = A.replace('], [', '\n')
        Q = Q.replace('], [', '\n')
        y = y.replace('], [', '\n')
        V = V.replace('], [', '\n')
        for c in "[]' ":
            A = A.replace(c, '')
            Q = Q.replace(c, '')
            y = y.replace(c, '')
            V = V.replace(c, '')

        with open('Results//Area.csv', 'w') as output_file:
            output_file.write(A)

        with open('Results//Discharge.csv', 'w') as output_file:
            output_file.write(Q)

        with open('Results//Depth.csv', 'w') as output_file:
            output_file.write(y)

        with open('Results//Velocity.csv', 'w') as output_file:
            output_file.write(V)
