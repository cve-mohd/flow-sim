from src.solver import Solver
from src.channel import Channel
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
                 channel: Channel,
                 theta: int | float,
                 time_step: int | float,
                 spatial_step: int | float,
                 simulation_time: int,
                 fit_spatial_step: bool = True,
                 regularization: bool = True):
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
        super().__init__(channel=channel,
                         time_step=time_step,
                         spatial_step=spatial_step,
                         simulation_time=simulation_time,
                         regularization=regularization,
                         fit_spatial_step=fit_spatial_step)
        
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
        self.area[0, :] = self.channel.initial_conditions[:, 0]
        self.flow[0, :] = self.channel.initial_conditions[:, 1]
        self.unknowns = self.channel.initial_conditions.flatten()

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
        R = np.zeros(shape=(self.number_of_nodes*2))
        
        R[0]  = self.upstream_residual()
        R[-1] = self.downstream_residual()
        
        for i in range(self.number_of_nodes-1):
            R[1+2*i] = self.continuity_residual(i)
            R[2+2*i] = self.momentum_residual(i)
            
        return R

    def compute_jacobian(self) -> np.ndarray:
        """
        Constructs the Jacobian matrix J of the system of equations.

        Returns
        -------
        np.ndarray
            J
            
        """
        jacobian_matrix = np.zeros(shape=(2*self.number_of_nodes, 2*self.number_of_nodes))

        jacobian_matrix[0, 0] = self.dU_dA()
        jacobian_matrix[0, 1] = self.dU_dQ()

        for row in range(1, 2 * self.number_of_nodes - 1, 2):
            spatial_node = (row - 1) // 2
            
            jacobian_matrix[row, row - 1] = self.dC_dAi(i=spatial_node)
            jacobian_matrix[row, row + 0] = self.dC_dQi(i=spatial_node)
            jacobian_matrix[row, row + 1] = self.dC_dAiplus1(i=spatial_node)
            jacobian_matrix[row, row + 2] = self.dC_dQiplus1(i=spatial_node)

            jacobian_matrix[row + 1, row - 1] = self.dM_dAi(i=spatial_node)
            jacobian_matrix[row + 1, row + 0] = self.dM_dQi(i=spatial_node)
            jacobian_matrix[row + 1, row + 1] = self.dM_dAiplus1(i=spatial_node)
            jacobian_matrix[row + 1, row + 2] = self.dM_dQiplus1(i=spatial_node)
        
        jacobian_matrix[-1, -2] = self.dD_dA()
        jacobian_matrix[-1, -1] = self.dD_dQ()

        return jacobian_matrix

    def run(self, tolerance=1e-4, verbose=3) -> None:
        """
        Run the simulation.
        """
        running = True
        total_iterations = 0
        
        while running:
            self.time_level += 1
            if self.time_level >= self.max_timelevels:
                running = False
                self.time_level = self.max_timelevels-1
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
                self.unknowns += delta
                
                error = euclidean_norm(delta)
                
                if verbose == 3:
                    print(f">> Iteration #{iteration}: Error = {error}")    
                if error < tolerance:
                    converged = True
                                
            if verbose==2:
                print(f'>> {iteration} iterations.')
                
            total_iterations += iteration
            
        super().finalize(verbose)
    
    def update_guesses(self) -> None:
        """
        Set the trial values of the unknowns to the ones last computed.

        Returns
        -------
        None.

        """
        self.area[self.time_level] = self.unknowns[ ::2]
        self.flow[self.time_level] = self.unknowns[1::2]
        
    def upstream_residual(self) -> float:
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
        time = self.time_level * self.time_step
        
        return self.channel.upstream_boundary.condition_residual(time=time,
                                                                 depth=self.depth_at(i=0),
                                                                 width=self.width_at(i=0),
                                                                 flow_rate=self.flow_at(i=0),
                                                                 bed_slope=self.bed_slope_at(i=0),
                                                                 roughness=self.channel.get_n(A=self.area_at(i=0), i=0))

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
        dA_dt = self.time_diff(
            k_i=self.area_at(k=-1,i=i),
            k_i1=self.area_at(k=-1,i=i+1),
            k1_i=self.area_at(i=i),
            k1_i1=self.area_at(i=i+1)
            )

        dQ_dx = self.spatial_diff(
                k_i=self.flow_at(k=-1,i=i),
                k_i1=self.flow_at(k=-1,i=i+1),
                k1_i=self.flow_at(i=i),
                k1_i1=self.flow_at(i=i+1)
            )

        return dA_dt + dQ_dx

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
        dQ_dt = self.time_diff(
            k_i=self.flow_at(k=-1,i=i),
            k_i1=self.flow_at(k=-1,i=i+1),
            k1_i=self.flow_at(i=i),
            k1_i1=self.flow_at(i=i+1)
            )

        dQ2A_dx = self.spatial_diff(
            k_i=self.flow_at(k=-1,i=i) ** 2 / self.area_at(k=-1,i=i),
            k_i1=self.flow_at(k=-1,i=i+1) ** 2 / self.area_at(k=-1,i=i+1),
            k1_i=self.flow_at(i=i) ** 2 / self.area_at(i=i),
            k1_i1=self.flow_at(i=i+1) ** 2 / self.area_at(i=i+1)
            )

        avg_A = self.cell_avg(
            k_i=self.area_at(k=-1,i=i),
            k_i1=self.area_at(k=-1,i=i+1),
            k1_i=self.area_at(i=i),
            k1_i1=self.area_at(i=i+1)
            )

        dY_dx = self.spatial_diff(
            k_i=self.water_level_at(k=-1,i=i),
            k_i1=self.water_level_at(k=-1,i=i+1),
            k1_i=self.water_level_at(i=i),
            k1_i1=self.water_level_at(i=i+1)
            )
    
        avg_Se = self.cell_avg(
            k_i=self.Se_at(k=-1,i=i),
            k_i1=self.Se_at(k=-1,i=i+1),
            k1_i=self.Se_at(i=i),
            k1_i1=self.Se_at(i=i+1)
            )

        return dQ_dt + dQ2A_dx + g * avg_A * (dY_dx + avg_Se)
    
    def downstream_residual(self) -> float:
        """
        Computes the residual of the downstream boundary condition equation.

        Returns
        -------
        float
            The computed residual.

        """
        time = self.time_level * self.time_step
        
        return self.channel.downstream_boundary.condition_residual(time=time,
                                                                   depth=self.depth_at(i=-1),
                                                                   width=self.width_at(i=-1),
                                                                   flow_rate=self.flow_at(i=-1),
                                                                   bed_slope=self.bed_slope_at(i=-1),
                                                                   roughness=self.channel.get_n(A=self.area_at(i=-1), i=-1),
                                                                   duration=self.time_step,
                                                                   vol_in=0.5*(self.flow_at(k=-1, i=-1) + self.flow_at(i=-1))*self.time_step,
                                                                   Y_old=self.water_level_at(k=-1, i=-1))
            
    def dU_dA(self) -> int:
        """
        Computes the derivative of the upstream BC residual w.r.t. flow area.

        Returns
        -------
        float
            dU/dA

        """
        A = self.area_at(i=0)
        B = self.width_at(i=0)
        n = self.channel.get_n(A=A, i=0)
        S_0 = self.bed_slope_at(i=0)
                    
        dU_dn = self.channel.upstream_boundary.df_dn(depth=A/B, width=B, bed_slope=S_0, roughness=n)
        
        dU = self.channel.upstream_boundary.df_dA(area=A,
                                                  width=B,
                                                  bed_slope=S_0,
                                                  roughness=n) + dU_dn * self.channel.dn_dA(A=A, i=0)
        if not self.regularization:
            return dU
        else:
            dU_dAreg = dU            
            dU_dQe = self.channel.upstream_boundary.df_dQ()
                        
            return dU_dAreg * self.dAreg_dA(i=0) + dU_dQe * self.dQe_dA(i=0)
    
    def dU_dQ(self) -> int:
        """
        Computes the derivative of the upstream BC residual w.r.t. flow rate.

        Returns
        -------
        float
            dU/dQ

        """
        dU = self.channel.upstream_boundary.df_dQ()
        
        if not self.regularization:
            return dU
        else:
            dU_dQe = dU
            return dU_dQe * self.dQe_dQ(i=0)

    def dC_dAiplus1(self, i) -> float:
        """
        Computes the derivative of the continuity residual w.r.t. flow area of the advanced spatial node.

        Returns
        -------
        float
            dC/dA_(i+1)

        """
        d_dA_dt_dA = self.time_diff(k1_i1=1)
        d_dQ_dx_dA = 0

        if not self.regularization:
            return d_dA_dt_dA + d_dQ_dx_dA
        else:
            dC_dAreg = d_dA_dt_dA + d_dQ_dx_dA
            dC_dQe = self.dC_dQiplus1(i, eff=True)
            
            return dC_dAreg * self.dAreg_dA(i=i+1) + dC_dQe * self.dQe_dA(i=i+1)

    def dC_dAi(self, i) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the cross-sectional area of the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d_dA_dt_dA = self.time_diff(k1_i=1)
        d_dQ_dx_dA = 0

        if not self.regularization:
            return d_dA_dt_dA + d_dQ_dx_dA
        else:
            dC_dAreg = d_dA_dt_dA + d_dQ_dx_dA
            dC_dQe = self.dC_dQi(i, eff=True)
            
            return dC_dAreg * self.dAreg_dA(i=i) + dC_dQe * self.dQe_dA(i=i)

    def dC_dQiplus1(self, i, eff = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d_dA_dt_dQ = 0
        d_dQ_dx_dQ = self.spatial_diff(k1_i1=1)

        if not self.regularization or eff:
            return d_dA_dt_dQ + d_dQ_dx_dQ
        else:
            dC_dQe = d_dA_dt_dQ + d_dQ_dx_dQ
            return dC_dQe * self.dQe_dQ(i=i+1)
    
    def dC_dQi(self, i, eff = False) -> float:
        """
        Computes the derivative of the continuity equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        d_dA_dt_dQ = 0
        d_dQ_dx_dQ = self.spatial_diff(k1_i=1)

        if not self.regularization or eff:
            return d_dA_dt_dQ + d_dQ_dx_dQ
        else:
            dC_dQe = d_dA_dt_dQ + d_dQ_dx_dQ
            return dC_dQe * self.dQe_dQ(i=i)

    def dM_dAiplus1(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        # Flow variables:
        A = self.area_at(i=i+1)
        Q = self.flow_at(i=i+1)
        
        # Basic derivatives:
        dY_dA = 1.0/self.width_at(i=i+1)
        dSe_dA = self.channel.dSe_dA(A, Q, i+1)
        
        # Finite differences:
        avg_A = self.cell_avg(
            k_i=self.area_at(k=-1, i=i),
            k_i1=self.area_at(k=-1, i=i+1),
            k1_i=self.area_at(i=i),
            k1_i1=self.area_at(i=i+1)
            )
        
        dY_dx = self.spatial_diff(
            k_i=self.water_level_at(k=-1, i=i),
            k_i1=self.water_level_at(k=-1, i=i+1),
            k1_i=self.water_level_at(i=i),
            k1_i1=self.water_level_at(i=i+1)
            )
        
        avg_Se = self.cell_avg(
            k_i=self.Se_at(k=-1, i=i),
            k_i1=self.Se_at(k=-1, i=i+1),
            k1_i=self.Se_at(i=i),
            k1_i1=self.Se_at(i=i+1)
            )
        
        # Derivatives of finite differences:
        d_dQdt_dA = 0
        d_dQ2Adx_dA = -self.spatial_diff(k1_i1=1) * (Q/A) ** 2
        d_avgA_dA = self.cell_avg(k1_i1=1)
        d_dYdx_dA = self.spatial_diff(k1_i1=1) * dY_dA
        d_avgSe_dA = self.cell_avg(k1_i1=1) * dSe_dA
        
        # dM/dA:
        dM_dA = d_dQdt_dA + d_dQ2Adx_dA + g * (
            avg_A * (d_dYdx_dA + d_avgSe_dA) + d_avgA_dA * (dY_dx + avg_Se)
            )
        
        if not self.regularization:
            return dM_dA
        else:
            dM_dQe = self.dM_dQiplus1(i, eff=True)
            return dM_dA * self.dAreg_dA(i=i+1) + dM_dQe * self.dQe_dA(i=i+1)

    def dM_dAi(self, i) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the cross-sectional area at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        # Flow variables:
        A = self.area_at(i=i)
        Q = self.flow_at(i=i)
        
        # Basic derivatives:
        dY_dA = 1.0/self.width_at(i=i)
        dSe_dA = self.channel.dSe_dA(A, Q, i)
        
        # Finite differences:
        avg_A = self.cell_avg(
            k_i=self.area_at(k=-1, i=i),
            k_i1=self.area_at(k=-1, i=i+1),
            k1_i=self.area_at(i=i),
            k1_i1=self.area_at(i=i+1)
            )
        
        dY_dx = self.spatial_diff(
            k_i=self.water_level_at(k=-1, i=i),
            k_i1=self.water_level_at(k=-1, i=i+1),
            k1_i=self.water_level_at(i=i),
            k1_i1=self.water_level_at(i=i+1)
            )
        
        avg_Se = self.cell_avg(
            k_i=self.Se_at(k=-1, i=i),
            k_i1=self.Se_at(k=-1, i=i+1),
            k1_i=self.Se_at(i=i),
            k1_i1=self.Se_at(i=i+1)
            )
        
        # Derivatives of finite differences:
        d_dQdt_dA = 0
        d_dQ2Adx_dA = -self.spatial_diff(k1_i=1) * (Q/A) ** 2
        d_avgA_dA = self.cell_avg(k1_i=1)
        d_dYdx_dA = self.spatial_diff(k1_i=1) * dY_dA
        d_avgSe_dA = self.cell_avg(k1_i=1) * dSe_dA
        
        # dM/dA:
        dM_dA = d_dQdt_dA + d_dQ2Adx_dA + g * (
            avg_A * (d_dYdx_dA + d_avgSe_dA) + d_avgA_dA * (dY_dx + avg_Se)
            )
        
        if not self.regularization:
            return dM_dA
        else:
            return dM_dA * self.dAreg_dA(i=i) + self.dM_dQi(i, eff=True) * self.dQe_dA(i=i)

    def dM_dQiplus1(self, i, eff=False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the advanced spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        # Flow variables:
        A = self.area_at(i=i+1)
        Q = self.flow_at(i=i+1)
        
        # Basic derivatives:
        dSe_dQ = self.channel.dSe_dQ(A, Q, i+1)
        
        # Finite differences:
        avg_A = self.cell_avg(
            k_i=self.area_at(k=-1, i=i),
            k_i1=self.area_at(k=-1, i=i+1),
            k1_i=self.area_at(i=i),
            k1_i1=self.area_at(i=i+1)
            )
        
        dY_dx = self.spatial_diff(
            k_i=self.water_level_at(k=-1, i=i),
            k_i1=self.water_level_at(k=-1, i=i+1),
            k1_i=self.water_level_at(i=i),
            k1_i1=self.water_level_at(i=i+1)
            )
        
        avg_Se = self.cell_avg(
            k_i=self.Se_at(k=-1, i=i),
            k_i1=self.Se_at(k=-1, i=i+1),
            k1_i=self.Se_at(i=i),
            k1_i1=self.Se_at(i=i+1)
            )
        
        # Derivatives of finite differences:
        d_dQdt_dQ = self.time_diff(k1_i1=1)
        d_dQ2Adx_dQ = self.spatial_diff(k1_i1=1) * 2*Q/A
        d_avgA_dQ = 0
        d_dYdx_dQ = 0
        d_avgSe_dQ = self.cell_avg(k1_i1=1) * dSe_dQ
        
        # dM/dA:
        dM_dQ = d_dQdt_dQ + d_dQ2Adx_dQ + g * (
            avg_A * (d_dYdx_dQ + d_avgSe_dQ) + d_avgA_dQ * (dY_dx + avg_Se)
            )
    
        if not self.regularization or eff:
            return dM_dQ
        else:
            return dM_dQ * self.dQe_dQ(i=i+1)

    def dM_dQi(self, i, eff = False) -> float:
        """
        Computes the derivative of the momentum equation with respect to
        the discharge at the current spatial point.

        Returns
        -------
        float
            The computed derivative.

        """
        # Flow variables:
        A = self.area_at(i=i)
        Q = self.flow_at(i=i)
        
        # Basic derivatives:
        dSe_dQ = self.channel.dSe_dQ(A, Q, i)
        
        # Finite differences:
        avg_A = self.cell_avg(
            k_i=self.area_at(k=-1, i=i),
            k_i1=self.area_at(k=-1, i=i+1),
            k1_i=self.area_at(i=i),
            k1_i1=self.area_at(i=i+1)
            )
        
        dY_dx = self.spatial_diff(
            k_i=self.water_level_at(k=-1, i=i),
            k_i1=self.water_level_at(k=-1, i=i+1),
            k1_i=self.water_level_at(i=i),
            k1_i1=self.water_level_at(i=i+1)
            )
        
        avg_Se = self.cell_avg(
            k_i=self.Se_at(k=-1, i=i),
            k_i1=self.Se_at(k=-1, i=i+1),
            k1_i=self.Se_at(i=i),
            k1_i1=self.Se_at(i=i+1)
            )
        
        # Derivatives of finite differences:
        d_dQdt_dQ = self.time_diff(k1_i=1)
        d_dQ2Adx_dQ = self.spatial_diff(k1_i=1) * 2*Q/A
        d_avgA_dQ = 0
        d_dYdx_dQ = 0
        d_avgSe_dQ = self.cell_avg(k1_i=1) * dSe_dQ
        
        # dM/dA:
        dM_dQ = d_dQdt_dQ + d_dQ2Adx_dQ + g * (
            avg_A * (d_dYdx_dQ + d_avgSe_dQ) + d_avgA_dQ * (dY_dx + avg_Se)
            )
    
        if not self.regularization or eff:
            return dM_dQ
        else:
            return dM_dQ * self.dQe_dQ(i=i)

    def dD_dA(self) -> float:
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the cross-sectional area of the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        A = self.area_at(i=-1)
        B = self.width_at(i=-1)
        n = self.channel.get_n(A=A, i=-1)
        S_0 = self.bed_slope_at(i=-1)
                
        dD_dn = self.channel.downstream_boundary.df_dn(depth=A/B, width=B, bed_slope=S_0, roughness=n)
        dn_dA = self.channel.dn_dA(A=A, i=-1)
        
        dD = self.channel.downstream_boundary.df_dA(area=A,
                                                    width=B,
                                                    bed_slope=S_0,
                                                    roughness=n) + dD_dn * dn_dA
        if not self.regularization:
            return dD
            
        else:
            dD_dAreg = dD
            dD_dQe = self.channel.downstream_boundary.df_dQ()
            
            return dD_dAreg * self.dAreg_dA(i=-1) + dD_dQe * self.dQe_dA(i=-1)

    def dD_dQ(self):
        """
        Computes the derivative of the downstream boundary condition equation
        with respect to the discharge at the downstream node.

        Returns
        -------
        float
            The computed derivative.

        """
        dD = self.channel.downstream_boundary.df_dQ(
            duration=self.time_step,
            vol_in=0.5*(self.flow_at(k=-1, i=-1) + self.flow_at(i=-1))*self.time_step,
            Y_old=self.water_level_at(k=-1, i=-1)
        )
        
        if not self.regularization:
            return dD
        else:
            dD_dQe = dD            
            return dD_dQe * self.dQe_dQ(i=-1)
      
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
        A_min = self.width_at(i=i) * h_min
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
        
        A_min = self.width_at(i=i) * h_min
        A_reg = self.area_at(i, 1, True)
        Q = self.flow_at(i, 1, False)
        
        return A_min * Q * self.dAreg_dA(i=i) / (A_reg + A_min) ** 2
        
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
        
        A_min = self.width_at(i=i) * h_min
        A_reg = self.area_at(i, True, True)
        
        chi = A_reg / (A_reg + A_min)
        
        return chi

    def time_diff(self, k1_i1 = 0, k1_i = 0, k_i1 = 0, k_i = 0):
        return (k1_i1 + k1_i - k_i1 - k_i) / (2 * self.time_step)
    
    def spatial_diff(self, k1_i1 = 0, k1_i = 0, k_i1 = 0, k_i = 0):
        dx_k1 = (k1_i1 - k1_i) / self.spatial_step
        dx_k  = (k_i1  - k_i)  / self.spatial_step
        return self.theta * dx_k1 + (1 - self.theta) * dx_k
        
    def cell_avg(self, k1_i1 = 0, k1_i = 0, k_i1 = 0, k_i = 0):
        k1 = 0.5 * self.theta * (k1_i1 + k1_i)
        k2 = 0.5 * (1 - self.theta) * (k_i1 + k_i)
        return k1 + k2
    