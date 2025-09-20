from src.solver import Solver
from src.reach import Reach
import numpy as np
from scipy.constants import g
from src.utility import euclidean_norm


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
                 simulation_time: int,
                 enforce_physicality: bool = True,
                 fit_spatial_step: bool = True):
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
        super().__init__(reach, time_step, spatial_step, simulation_time, enforce_physicality, fit_spatial_step)
        
        self.theta = theta
        self.unknowns = None
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
        self.area[0, :] = self.reach.initial_conditions[:, 0]
        self.flow[0, :] = self.reach.initial_conditions[:, 1]
        self.unknowns = self.reach.initial_conditions.flatten()#.reshape(-1, 1)
        
        if self.active_storage:
            self.unknowns = np.append(self.unknowns, self.reach.downstream_boundary.lumped_storage.stage)
            _, self.outflow[0] = self.reach.downstream_boundary.lumped_storage.mass_balance(self.time_step,
                                                                                         self.flow[0, -1],
                                                                                         self.water_level_at(-1, 0))

    def compute_residual_vector(self) -> np.ndarray:
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
        time = self.time_level * self.time_step
        R = np.zeros(shape=(self.number_of_nodes*2, 1))
        
        R[0]  = self.upstream_residual(time)
        R[-1] = self.downstream_residual(time)
        
        for i in range(self.number_of_nodes-1):
            R[1+2*i] = self.continuity_residual(i)
            R[2+2*i] = self.momentum_residual(i)
        
        if self.active_storage:
            R = np.append(R, self.storage_residual())
            
        return R

    def compute_jacobian(self) -> np.ndarray:
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

        jacobian_matrix[0, 0] = self.dU_dA()
        jacobian_matrix[0, 1] = self.dU_dQ()

        for row in range(1, 2 * self.number_of_nodes - 1, 2):
            spatial_node = (row - 1) // 2
            
            jacobian_matrix[row, row - 1] = self.dC_dAi(i = spatial_node)
            jacobian_matrix[row, row + 0] = self.dC_dQi(i = spatial_node)
            jacobian_matrix[row, row + 1] = self.dC_dAiplus1(i = spatial_node)
            jacobian_matrix[row, row + 2] = self.dC_dQiplus1(i = spatial_node)

            jacobian_matrix[row + 1, row - 1] = self.dM_dAi(i = spatial_node)
            jacobian_matrix[row + 1, row + 0] = self.dM_dQi(i = spatial_node)
            jacobian_matrix[row + 1, row + 1] = self.dM_dAiplus1(i = spatial_node)
            jacobian_matrix[row + 1, row + 2] = self.dM_dQiplus1(i = spatial_node)
        
        if self.active_storage:        
            jacobian_matrix[-2, -3] = self.dD_dA()
            jacobian_matrix[-2, -2] = self.dD_dQ()
            jacobian_matrix[-2, -1] = self.dD_dYN()
            
            jacobian_matrix[-1, -2] = self.dSr_dQ()
            jacobian_matrix[-1, -1] = self.dSr_dYN()
        else:
            jacobian_matrix[-1, -2] = self.dD_dA()
            jacobian_matrix[-1, -1] = self.dD_dQ()

        return jacobian_matrix

    def run(self, tolerance=1e-4, verbose=3) -> None:
        """
        Run the simulation.
        """
        running = True
        
        while running:
            self.time_level += 1
            if self.time_level >= self.max_timelevels:
                running = False
                self.time_level -= 1
                break
            
            if verbose >= 1:
                print(f'\n> Time level #{self.time_level}')

            iteration = 0
            converged = False

            while not converged:
                iteration += 1
                if iteration - 1 >= 100:
                    raise ValueError(f'Convergence within {iteration - 1} iterations couldn\'t be achieved.')
                
                self.update_guesses()
                
                R = self.compute_residual_vector()
                J = self.compute_jacobian()
                
                if np.isnan(R).any() or np.isnan(J).any():
                    raise ValueError("NaN in system assembly")
                if np.linalg.cond(J) > 1e12:
                    raise ValueError("Jacobian is ill-conditioned (near singular).")
                                
                delta = np.linalg.solve(J, -R)
                self.unknowns += delta.flatten()
                error = euclidean_norm(delta)
                
                if verbose == 3:
                    print(f">> Iteration #{iteration}: Error = {error}")    
                if error < tolerance:
                    converged = True
                                
            if verbose==2:
                print(f'>> {iteration} iterations.')
                        
        super().finalize(verbose)
    
    def update_guesses(self) -> None:
        """
        Set the trial values of the unknowns to the ones last computed.

        Returns
        -------
        None.

        """
        if self.active_storage:
            self.area[self.time_level, :] = self.unknowns[ :-1:2]
            self.flow[self.time_level, :] = self.unknowns[1:-1:2]
            self.reach.downstream_boundary.lumped_storage.stage = float(self.unknowns[-1])
        else:
            self.area[self.time_level] = self.unknowns[ ::2]
            self.flow[self.time_level] = self.unknowns[1::2]
        
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
        A = self.area_at(i=0)
        h = self.depth_at(i=0)
        B = self.width_at(i=0)
        Q = self.flow_at(i=0)
        S_0 = self.bed_slope_at(0)
        n = self.reach.get_n(A=A, i=0)
                
        residual = self.reach.upstream_boundary.condition_residual(time=time,
                                                                   depth=h,
                                                                   width=B,
                                                                   flow_rate=Q,
                                                                   bed_slope=S_0,
                                                                   roughness=n)
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
        return self.time_diff(
            k0_i0 = self.area_at(i, -1),
            k0_i1 = self.area_at(i+1, -1),
            k1_i0 = self.area_at(i),
            k1_i1 = self.area_at(i+1)
            ) + self.spatial_diff(
                k0_i0 = self.flow_at(i, -1),
                k0_i1 = self.flow_at(i+1, -1),
                k1_i0 = self.flow_at(i),
                k1_i1 = self.flow_at(i+1)
            )

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
        return self.time_diff(
            k0_i0 = self.flow_at(i, -1),
            k0_i1 = self.flow_at(i+1, -1),
            k1_i0 = self.flow_at(i),
            k1_i1 = self.flow_at(i+1)
            ) + self.spatial_diff(
                k0_i0 = self.flow_at(i, -1) ** 2 / self.area_at(i, -1),
                k0_i1 = self.flow_at(i+1, -1) ** 2 / self.area_at(i+1, -1),
                k1_i0 = self.flow_at(i) ** 2 / self.area_at(i),
                k1_i1 = self.flow_at(i+1) ** 2 / self.area_at(i+1)
            ) + g * self.cell_avg(
                k0_i0=self.area_at(i, -1),
                k0_i1=self.area_at(i+1, -1),
                k1_i0=self.area_at(i),
                k1_i1=self.area_at(i+1)
            ) * (
                self.spatial_diff(
                    k0_i0=self.water_level_at(i, -1),
                    k0_i1=self.water_level_at(i+1, -1),
                    k1_i0=self.water_level_at(i),
                    k1_i1=self.water_level_at(i+1)
                ) + self.cell_avg(
                    k0_i0=self.Se_at(i, -1),
                    k0_i1=self.Se_at(i+1, -1),
                    k1_i0=self.Se_at(i),
                    k1_i1=self.Se_at(i+1)
                )
            )
    
    def downstream_residual(self, time) -> float:
        """
        Computes the residual of the downstream boundary condition equation.

        Returns
        -------
        float
            The computed residual.

        """
        A = self.area_at(-1)
        h = self.depth_at(-1)
        B = self.width_at(-1)
        n = self.reach.get_n(A=A, i=-1)
        S_0 = self.bed_slope_at(-1)
        Q = self.flow_at(-1)
        
        residual = self.reach.downstream_boundary.condition_residual(time=time,
                                                                     depth=h,
                                                                     width=B,
                                                                     flow_rate=Q,
                                                                     bed_slope=S_0,
                                                                     roughness=n)
        
        return residual

    def storage_residual(self):
        vol_in = 0.5 * (self.flow_at(-1) + self.flow_at(-1, -1)) * self.time_step
        Y_old = self.water_level_at(-1, -1)
        
        target_Y, vol_out = self.reach.downstream_boundary.lumped_storage.mass_balance(duration=self.time_step, vol_in=vol_in, Y_old=Y_old)
        self.outflow[self.time_level] = vol_out / self.time_step
        
        residual = self.reach.downstream_boundary.lumped_storage.stage - target_Y
        return residual
    
    def dU_dA(self) -> int:
        """
        Computes the derivative of the upstream BC residual w.r.t. flow area.

        Returns
        -------
        float
            dU/dA

        """
        A = self.area_at(0)
        h = self.depth_at(0)
        B = self.width_at(0)
        n = self.reach.get_n(A=A, i=0)
        S_0 = self.bed_slope_at(0)
                    
        dU_dn = self.reach.upstream_boundary.df_dn(depth=h, width=B, bed_slope=S_0, roughness=n)
        
        dU = self.reach.upstream_boundary.df_dA(area=A,
                                                width=B,
                                                bed_slope=S_0,
                                                roughness=n) + dU_dn * self.reach.dn_dA(A=A, i=0)
        if not self.regularization:
            return dU
        else:
            dU_dAreg = dU            
            dU_dQe = self.reach.upstream_boundary.df_dQ()
                        
            return dU_dAreg * self.dAreg_dA(i=0) + dU_dQe * self.dQe_dA(i=0)
    
    def dU_dQ(self) -> int:
        """
        Computes the derivative of the upstream BC residual w.r.t. flow rate.

        Returns
        -------
        float
            dU/dQ

        """
        dU = self.reach.upstream_boundary.df_dQ()
        
        if not self.regularization:
            return dU
        else:
            dU_dQe = dU
            return dU_dQe * self.dQe_dQ(0)

    def dC_dAiplus1(self, i) -> float:
        """
        Computes the derivative of the continuity residual w.r.t. flow area of the advanced spatial node.

        Returns
        -------
        float
            dC/dA_(i+1)

        """
        dC = 0.5 / self.time_step
        
        if not self.regularization:
            return dC
        else:
            dC_dAreg = dC            
            dC_dQe = self.dC_dQiplus1(i, eff=True)
            
            return dC_dAreg * self.dAreg_dA(i + 1) + dC_dQe * self.dQe_dA(i + 1)

    def dC_dAi(self, i) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dC = 0.5 / self.time_step
        
        if not self.regularization:
            return dC
        else:
            dC_dAreg = dC
            dC_dQe = self.dC_dQi(i, eff=True)
            
            return dC_dAreg * self.dAreg_dA(i) + dC_dQe * self.dQe_dA(i)

    def dC_dQiplus1(self, i, eff = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dC = self.theta / self.spatial_step
        
        if not self.regularization or eff:
            return dC
        else:
            dC_dQe = dC                        
            return dC_dQe * self.dQe_dQ(i + 1)
    
    def dC_dQi(self, i, eff = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        dC = - self.theta / self.spatial_step
        
        if not self.regularization or eff:
            return dC
        else:
            dC_dQe = dC
            return dC_dQe * self.dQe_dQ(i)

    def dM_dAiplus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(i+1)
        Q = self.flow_at(i+1)
        B = self.width_at(i+1)
        dSe_dA = self.reach.dSe_dA(A, Q, i+1)
        
        avg_A = self.cell_avg(
            k0_i0=self.area_at(i, -1),
            k0_i1=self.area_at(i+1, -1),
            k1_i0=self.area_at(i),
            k1_i1=self.area_at(i+1)
        )
        avg_Se = self.cell_avg(
            k0_i0=self.Se_at(i, -1),
            k0_i1=self.Se_at(i+1, -1),
            k1_i0=self.Se_at(i),
            k1_i1=self.Se_at(i+1)
        )
        dY_dx = self.spatial_diff(
            k0_i0=self.water_level_at(i, -1),
            k0_i1=self.water_level_at(i+1, -1),
            k1_i0=self.water_level_at(i),
            k1_i1=self.water_level_at(i+1)
        )
        
        # -----------------
        
        dM_dA = (
            - self.theta / self.spatial_step * (Q/A) ** 2
            + 0.5 * g * self.theta * (dY_dx + avg_Se)
            + g * self.theta * avg_A * (1. / (self.spatial_step * B) + 0.5 * dSe_dA)
        )
                
        if not self.regularization:
            return dM_dA
        else:
            dM_dQe = self.dM_dQiplus1(i, eff=True)
            return dM_dA * self.dAreg_dA(i+1) + dM_dQe * self.dQe_dA(i+1)

    def dM_dAi(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(i)
        Q = self.flow_at(i)
        B = self.width_at(i)
        dSe_dA = self.reach.dSe_dA(A, Q, i)
        
        avg_A = self.cell_avg(
            k0_i0=self.area_at(i, -1),
            k0_i1=self.area_at(i+1, -1),
            k1_i0=self.area_at(i),
            k1_i1=self.area_at(i+1)
        )
        avg_Se = self.cell_avg(
            k0_i0=self.Se_at(i, -1),
            k0_i1=self.Se_at(i+1, -1),
            k1_i0=self.Se_at(i),
            k1_i1=self.Se_at(i+1)
        )
        dY_dx = self.spatial_diff(
            k0_i0=self.water_level_at(i, -1),
            k0_i1=self.water_level_at(i+1, -1),
            k1_i0=self.water_level_at(i),
            k1_i1=self.water_level_at(i+1)
        )
        
        # -----------------
        
        dM_dA = (
            + self.theta / self.spatial_step * (Q/A) ** 2
            + 0.5 * g * self.theta * (dY_dx + avg_Se)
            + g * self.theta * avg_A * (-1. / (self.spatial_step * B) + 0.5 * dSe_dA)
        )
        
        if not self.regularization:
            return dM_dA
        else:
            return dM_dA * self.dAreg_dA(i) + self.dM_dQi(i, eff=True) * self.dQe_dA(i)

    def dM_dQiplus1(self, i, eff=False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(i+1)
        Q = self.flow_at(i+1)
        dSe_dQ = self.reach.dSe_dQ(A, Q, i+1)
        
        avg_A = self.cell_avg(
            k0_i0=self.area_at(i, -1),
            k0_i1=self.area_at(i+1, -1),
            k1_i0=self.area_at(i),
            k1_i1=self.area_at(i+1)
        )
        
        # -----------------
        
        dM_dQ = (
            + 0.5 / self.time_step
            + 2 * self.theta / self.spatial_step * Q / A
            + 0.5 * self.theta * g * avg_A * dSe_dQ
        )
        
        if not self.regularization or eff:
            return dM_dQ
        else:
            return dM_dQ * self.dQe_dQ(i+1)

    def dM_dQi(self, i, eff = False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(i)
        Q = self.flow_at(i)
        dSe_dQ = self.reach.dSe_dQ(A, Q, i)
        
        avg_A = self.cell_avg(
            k0_i0=self.area_at(i, -1),
            k0_i1=self.area_at(i+1, -1),
            k1_i0=self.area_at(i),
            k1_i1=self.area_at(i+1)
        )
        
        # -----------------
        
        dM_dQ = (
            + 0.5 / self.time_step
            - 2 * self.theta / self.spatial_step * Q / A
            + 0.5 * self.theta * g * avg_A * dSe_dQ
        )
        
        if not self.regularization or eff:
            return dM_dQ
        else:
            return dM_dQ * self.dQe_dQ(i)

    def dD_dA(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(-1)
        h = self.depth_at(-1)
        B = self.width_at(-1)
        n = self.reach.get_n(A=A, i=-1)
        S_0 = self.bed_slope_at(-1)
                
        dD_dn = self.reach.downstream_boundary.df_dn(depth=h, width=B, bed_slope=S_0, roughness=n)
        dn_dA = self.reach.dn_dA(A=A, i=-1)
        
        dD = self.reach.downstream_boundary.df_dA(area=A,
                                                  width=B,
                                                  bed_slope=S_0,
                                                  roughness=n) + dD_dn * dn_dA
        if not self.regularization:
            return dD
            
        else:
            dD_dAreg = dD
            dD_dQe = self.reach.downstream_boundary.df_dQ()
            
            return dD_dAreg * self.dAreg_dA(-1) + dD_dQe * self.dQe_dA(-1)

    def dD_dQ(self):
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        dD = self.reach.downstream_boundary.df_dQ()
        
        if not self.regularization:
            return dD
        else:
            dD_dQe = dD            
            return dD_dQe * self.dQe_dQ(-1)
    
    def dD_dYN(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the storage depth.

        Returns
        -------
        float
            The computed derivative.

        """
        return self.reach.downstream_boundary.df_dYN()
            
    def dSr_dQ(self):
        vol_in = 0.5 * (self.flow_at(-1) + self.flow_at(-1, -1)) * self.time_step
        Y_old = self.water_level_at(-1, -1)
                
        dSr_dvol = 0 - self.reach.downstream_boundary.lumped_storage.dY_new_dvol_in(self.time_step, vol_in, Y_old)
        dvol_dQ = 0.5 * self.time_step
        
        dSr = dSr_dvol * dvol_dQ
        
        if not self.regularization:
            return dSr
        else:
            dSr_dQe = dSr
            return dSr_dQe * self.dQe_dQ(-1)
        
    def dSr_dYN(self):
        return 1
            
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
        A_min = self.width_at(i) * h_min
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
        
        A_min = self.width_at(i) * h_min
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
        
        A_min = self.width_at(i) * h_min
        A_reg = self.area_at(i, True, True)
        
        chi = A_reg / (A_reg + A_min)
        
        return chi

    def time_diff(self, k1_i1, k1_i0, k0_i1, k0_i0):
        return (k1_i1 + k1_i0 - k0_i1 - k0_i0) / (2 * self.time_step)
    
    def spatial_diff(self, k1_i1, k1_i0, k0_i1, k0_i0):
        dx_k1 = (k1_i1 - k1_i0) / self.spatial_step
        dx_k0 = (k0_i1 - k0_i0) / self.spatial_step
        
        return self.theta * dx_k1 + (1 - self.theta) * dx_k0
        
    def cell_avg(self, k1_i1, k1_i0, k0_i1, k0_i0):
        k1 = (k1_i1 + k1_i0) * self.theta / 2
        k2 = (k0_i1 + k0_i0) * (1 - self.theta) / 2
        
        return k1 + k2
    