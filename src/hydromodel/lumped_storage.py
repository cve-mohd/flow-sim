from .rating_curve import RatingCurve
from . import hydraulics
import numpy as np
from scipy.constants import g
from scipy.optimize import brentq

class LumpedStorage:
    def __init__(self, solution_boundaries: tuple, surface_area: float = None, min_stage: float = None, rating_curve: RatingCurve = None):
        self.rating_curve = rating_curve
        self.surface_area = surface_area
        self.min_stage = min_stage
        
        self.stage_hydrograph = []
        self.area_curve = None
        self.reservoir_length = None
        self.capture_losses = False
        self.Cc = 0.5    # Cc ~ 0.2-0.7
        self.K_q = 0
        
        if solution_boundaries is not None:
            self.Y_min = solution_boundaries[0]
            self.Y_max = solution_boundaries[1]
    
    def mass_balance(self, duration, vol_in, Y_old = None, time = None):
        def f(Y_new):
            Q_out = 0.5 * (self.rating_curve.discharge(Y_old, time) + self.rating_curve.discharge(Y_new, time)) if self.rating_curve else 0.0
            target_vol = vol_in - Q_out * duration
            return self.net_vol_change(Y_old, Y_new) - target_vol
        
        Y_target = brentq(f, self.Y_min, self.Y_max)
                
        if Y_target < self.min_stage:
            Y_target = self.min_stage

        return Y_target

    def dY_new_dvol_in(self, duration, vol_in, Y_old, time = None) -> float:
        """
        d(Y_new)/d(vol_in)
        """
        Y_new = self.mass_balance(duration, vol_in, Y_old, time)
        if Y_new <= self.min_stage:
            return 0.0
        
        return 1/self.area_at(Y_new)
    
    def energy_loss(self, A_ent, Q, n, R, A_str = None):
        if not self.capture_losses:
            return 0
                
        hf = self.friction_loss(A_ent=A_ent, Q=Q, n=n, R=R)
        h_exp = self.expansion_loss(A_ent=A_ent, A_str=A_str, Q=Q)
        h_emp = self.empirical_loss(A_ent=A_ent, Q=Q)
        
        head_loss = hf + h_exp + h_emp
        return head_loss

    def friction_loss(self, A_ent, Q, n, R):
        Sf = hydraulics.Sf(A=A_ent, Q=Q, n=n, R=R)
        return Sf * self.reservoir_length

    def expansion_loss(self, A_ent, Q, A_str = None):
        if A_str is None:
            return 0
        
        K = (1 - A_ent/A_str)**2
        V = Q / A_ent
            
        return K * V**2 / (2*g)
    
    def empirical_loss(self, Q, A_ent):
        V = Q/A_ent
        return self.K_q * V**2 / (2*g)
    
    def dhl_dn(self, A_ent, Q, n, R):
        if not self.capture_losses:
            return 0
                
        dSf_dn = hydraulics.dSf_dn(A=A_ent, Q=Q, n=n, R=R)
        return dSf_dn * self.reservoir_length

    def dhf_dA(self, A_ent, Q, n, R, dR_dA):
        dSf_dA = hydraulics.dSf_dA(A=A_ent, Q=Q, n=n, R=R, dR_dA=dR_dA)
        return dSf_dA * self.reservoir_length
    
    def d_h_exp_dA(self, A_ent, Q, A_str = None):
        if A_str is None:
            return 0
        
        K = (1 - A_ent/A_str)**2
        V = Q / A_ent
        
        dK_dA = 2*(1 - A_ent/A_str) * (-1/A_str)
        dV_dA = -Q / A_ent**2
            
        return (K * 2*V*dV_dA + V**2 * dK_dA) / (2*g)
    
    def d_h_emp_dA(self, A_ent, Q):
        V = Q/A_ent
        dV_dA = -Q/(A_ent**2)
        
        return self.K_q * 2*V*dV_dA / (2*g)

    def dhl_dA(self, A_ent, Q, n, R, dR_dA, A_str = None):
        if not self.capture_losses:
            return 0
                
        dhf_dA = self.dhf_dA(A_ent=A_ent, Q=Q, n=n, R=R, dR_dA=dR_dA)
        d_h_exp_dA = self.d_h_exp_dA(A_ent=A_ent, Q=Q, A_str=A_str)
        d_h_emp_dA = self.d_h_emp_dA(A_ent=A_ent, Q=Q)
        
        return dhf_dA + d_h_exp_dA + d_h_emp_dA

    def dhf_dQ(self, A_ent, Q, n, R):
        dSf_dQ = hydraulics.dSf_dQ(A=A_ent, Q=Q, n=n, R=R)
        return dSf_dQ * self.reservoir_length

    def d_h_exp_dQ(self, A_ent, Q, A_str = None):
        if A_str is None:
            return 0
        
        K = (1 - A_ent/A_str)**2
        V = Q / A_ent
        
        dV_dQ = -Q / A_ent**2
            
        return K * 2*V*dV_dQ / (2*g)
    
    def d_h_emp_dQ(self, A_ent, Q):
        V = Q/A_ent
        dV_dQ = 1./A_ent
        
        return self.K_q * 2*V*dV_dQ / (2*g)
            
    def dhl_dQ(self, A_ent, Q, n, R, A_str = None):
        if not self.capture_losses:
            return 0
        
        dhf_dQ = self.dhf_dQ(A_ent=A_ent, Q=Q, n=n, R=R)
        d_h_exp_dQ = self.d_h_exp_dQ(A_ent=A_ent, Q=Q, A_str=A_str)
        d_h_emp_dQ = self.d_h_emp_dQ(A_ent=A_ent, Q=Q)
        
        return dhf_dQ + d_h_exp_dQ + d_h_emp_dQ
            
    def set_area_curve(self, table, alpha=1, beta=0, update_solution_boundaries = True):
        self.alpha = alpha
        self.beta = beta
        self.area_curve = np.asarray(table, dtype=np.float64)
        self.area_gradient = np.gradient(self.area_curve[:, 1], self.area_curve[:, 0])
        
        if update_solution_boundaries:
            self.Y_min = np.min(self.area_curve[:, 0])
            self.Y_max = np.max(self.area_curve[:, 0])
    
    def area_at(self, stage):
        if self.area_curve is None:
            return self.surface_area
        else:
            a = self.alpha * np.interp(stage+self.beta, self.area_curve[:, 0], self.area_curve[:, 1])
            return a

    def dA_dY(self, stage):
        if self.area_curve is None:
            return 0
        else:
            return self.alpha * np.interp(stage, self.area_curve[:, 0], self.area_gradient)
        
    def net_vol_change(self, Y1, Y2):
        if self.area_curve is None:
            return (Y2 - Y1) * self.surface_area
        else:
            step =  np.min(np.abs(self.area_curve[1:, 0] - self.area_curve[:-1, 0]))
            n = int(abs(Y2-Y1)/step)
            if n > 2:
                ys = np.linspace(Y1, Y2, n)
                areas = [self.area_at(y) for y in ys]
                return np.trapezoid(areas, ys)
            else:
                return 0.5 * (self.area_at(Y2) + self.area_at(Y1)) * (Y2 - Y1)
