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
        self.widths = None
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
    
    def energy_loss(self, Q, n, h):
        if not self.capture_losses:
            return 0
                
        hf = self.friction_loss(Q, h, n)
        h_exp = self.expansion_loss(Q, h)
        h_emp = self.empirical_loss(Q, h)
        
        head_loss = hf + h_exp + h_emp
        return head_loss

    def friction_loss(self, Q, h, n):
        A_ent = h * self.widths[0]
        Sf = hydraulics.Sf(A=A_ent, Q=Q, n=n, B=self.widths[0])
        return Sf * self.reservoir_length

    def expansion_loss(self, Q, h):
        h_trans = 0.0
        for i in range(len(self.widths)-1):
            A_up = (h-h_trans) * self.widths[i]; A_down = (h-h_trans) * self.widths[i+1]
            if A_down > A_up:
                K = (1 - A_up/A_down)**2
                V = Q / A_up
            else:
                # contraction
                K = self.Cc * (1 - A_down/A_up)
                V = Q / A_down
            h_trans += K * V**2 / (2*g)
            
        return h_trans
    
    def empirical_loss(self, Q, h):
        A_ent = h * self.widths[0]
        V = Q/A_ent
        return self.K_q * V**2 / (2*g)
    
    def dhl_dn(self, Q, h, n):
        if not self.capture_losses:
            return 0
                
        A_ent = h * self.widths[0]
        dSf_dn = hydraulics.dSf_dn(A=A_ent, Q=Q, n=n, B=self.widths[0])
        return dSf_dn * self.reservoir_length

    def dhf_dA(self, Q, h, n):
        A_ent = h * self.widths[0]
        dSf_dA = hydraulics.dSf_dA(A=A_ent, Q=Q, n=n, B=self.widths[0])
        return dSf_dA * self.reservoir_length
    
    def d_h_exp_dA(self, Q, h):
        dh_exp_dh = 0.0
        h_trans = 0.0
        for i in range(len(self.widths)-1):
            A_up = (h-h_trans) * self.widths[i]; A_down = (h-h_trans) * self.widths[i+1]
            if A_down > A_up:
                K = (1 - A_up/A_down)**2
                V = Q / A_up
                dV_dA_up = -Q / A_up**2
                dA_up_dh = self.widths[i]
                dV_dh = dV_dA_up * dA_up_dh
            else:
                # contraction
                K = self.Cc * (1 - A_down/A_up)
                V = Q / A_down
                dV_dA_down = -Q / A_down**2
                dA_down_dh = self.widths[i+1]
                dV_dh = dV_dA_down * dA_down_dh
                
            h_trans += K * V**2 / (2*g)
            dh_exp_dh += K * 2*V*dV_dh / (2*g)
            
        dh_dA = 1/self.widths[0]
        return dh_exp_dh * dh_dA
    
    def d_h_emp_dA(self, Q, h):
        A_ent = h * self.widths[0]
        V = Q/A_ent
        dV_dA = -Q/(A_ent**2)
        
        return self.K_q * 2*V*dV_dA / (2*g)

    def dhl_dA(self, Q, h, n):
        if not self.capture_losses:
            return 0
                
        dhf_dA = self.dhf_dA(Q, h, n)
        d_h_exp_dA = self.d_h_exp_dA(Q, h)
        d_h_emp_dA = self.d_h_emp_dA(Q, h)
        
        return dhf_dA + d_h_exp_dA + d_h_emp_dA

    def dhf_dQ(self, Q, h, n):
        A_ent = self.widths[0] * h
        dSf_dQ = hydraulics.dSf_dQ(A=A_ent, Q=Q, n=n, B=self.widths[0])
        return dSf_dQ * self.reservoir_length

    def d_h_exp_dQ(self, Q, h):
        dh_exp_dQ = 0.0
        h_trans = 0.0
        for i in range(len(self.widths)-1):
            A_up = (h-h_trans) * self.widths[i]; A_down = (h-h_trans) * self.widths[i+1]
            if A_down > A_up:
                K = (1 - A_up/A_down)**2
                V = Q / A_up
            else:
                # contraction
                K = self.Cc * (1 - A_down/A_up)
                V = Q / A_down
                
            h_trans += K * V**2 / (2*g)
            dV_dQ = V / Q
            dh_exp_dQ += K * 2*V*dV_dQ / (2*g)
            
        return dh_exp_dQ
    
    def d_h_emp_dQ(self, Q, h):
        A_ent = h * self.widths[0]
        V = Q/A_ent
        dV_dQ = 1./A_ent
        
        return self.K_q * 2*V*dV_dQ / (2*g)
            
    def dhl_dQ(self, Q, h, n):
        if not self.capture_losses:
            return 0
        
        dhf_dQ = self.dhf_dQ(Q, h, n)
        d_h_exp_dQ = self.d_h_exp_dQ(Q, h)
        d_h_emp_dQ = self.d_h_emp_dQ(Q, h)
        
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
