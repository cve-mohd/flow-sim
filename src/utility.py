import os
import numpy as np
from scipy.constants import g
    
def create_directory_if_not_exists(directory):
    """
    Checks if a directory exists and creates it if it doesn't.

    Attributes
    ----------
    directory : str
        The path to the directory to check.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def manhattan_norm(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return np.sum(np.abs(vector))

def euclidean_norm(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return np.sum(np.square(vector))**0.5

def seconds_to_hms(seconds: int):
    if seconds < 0:
        return "0:00:00"
    
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60
    
    return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

def compute_radii_curv(x_coords, y_coords):
    x_coords, y_coords = np.asarray(x_coords, dtype=np.float64), np.asarray(y_coords, dtype=np.float64)
    # Arc length parameterization
    ds = np.hypot(np.diff(x_coords), np.diff(y_coords))
    s = np.insert(np.cumsum(ds), 0, 0.0)

    dx = np.gradient(x_coords, s)
    dy = np.gradient(y_coords, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    # Curvature κ = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    radii = np.where(kappa != 0, 1/kappa, np.inf)
    
    return kappa, radii

class RatingCurve:
    def __init__(self):
        self.function = None
        self.derivative = None
        
        self.defined = False
        self.type = None    
    
    def set(self, type, a, b, c=None, stage_shift=None):
        if stage_shift is None:
            self.stage_shift = 0
            
        if type == 'polynomial':
            if c is None:
                raise ValueError("Insufficient arguments. c must be specified.")
            else:
                self.a, self.b, self.c = a, b, c
            
        elif type == 'power':
            self.a, self.b = a, b
            
        else:
            raise ValueError("Invalid type.")
        
        self.function = None
        self.derivative = None
        self.defined = True
        self.type = type
        
    def discharge(self, stage):
        """
        Computes the discharge for a given stage using
        the rating curve equation.

        Parameters
        ----------
        stage : float
            The stage or water level.

        Returns
        -------
        discharge : float
            The computed discharge in cubic meters per second.

        """
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        if self.function is not None:
            return self.function(stage)
        
        else:
            x = stage + self.stage_shift
            
            if self.type == 'polynomial':
                discharge = self.a * x**2 + self.b * x + self.c
                
            else:
                discharge = self.a * x**self.b
                    
            return discharge

    def stage(self, discharge: float, trial_stage: float = None, tolerance: float = 1e-2, rate=1) -> float:
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        if trial_stage is None:
            trial_stage = - self.stage_shift * 1.05
        
        q = self.discharge(stage=trial_stage)
        
        while abs(q - discharge) > tolerance:
            func = q - discharge
            deriv = self.derivative_wrt_stage(stage=trial_stage)
            
            delta = - rate * func / deriv
            trial_stage += delta
            q = self.discharge(stage=trial_stage)
        
        return trial_stage
    
    def fit(self, discharges: list, stages: list, stage_shift: float=0, type: str='polynomial', scale=True, degree: int=2):
        self.type = type
        discharges = np.asarray(discharges, dtype=np.float64)
        stages = np.asarray(stages, dtype=np.float64)
        
        if discharges.size < 3:
            raise ValueError("Need at least 3 points.")
        
        if discharges.shape != stages.shape:
            raise ValueError("Q and Y lists should have the same lengths.")
        
        self.stage_shift = stage_shift
        shifted_stages = stages + self.stage_shift
                
        if any(shifted_stages <= 0):
            raise ValueError("All (stage - base) values must be positive for power-law fitting.")

        if type == 'polynomial':
            if scale:
                self.function = np.polynomial.polynomial.Polynomial.fit(x=shifted_stages, y=discharges, deg=degree)
                self.derivative = self.function.deriv()
            
            else:
                if degree != 2:
                    print("WARNING: Polynomial degree defaults to 2 for unscaled fitting.")
                    
                a, b, c = np.polyfit(shifted_stages, discharges, deg=2)
                
                self.a = float(a)
                self.b = float(b)
                self.c = float(c)
            
        elif type == 'power':
            log_Y = np.log(shifted_stages)
            log_Q = np.log(discharges)

            # Fit: log(Q) = b * log(shifted_stages) + log(a)
            b, log_a = np.polyfit(log_Y, log_Q, deg=1)
            a = np.exp(log_a)

            self.a = float(a)
            self.b = float(b)
            
        else:
            raise ValueError("Invalid rating curve type.")
        
        self.defined = True
        
    def derivative_wrt_stage(self, stage):
        Y_ = stage + self.stage_shift
        
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        if self.type == 'polynomial':
            if self.function is not None:
                d = self.derivative(Y_)
            else:
                d = self.a * 2 * Y_ + self.b
        
        else:    
            d = self.a * self.b * Y_**(self.b - 1)
            
        return d
    
    def tostring(self):
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        if self.type == 'polynomial':
            if self.function is not None:
                return str(self.function)
            else:
                equation = str(self.a) + ' (Y+' + str(self.stage_shift) + ')^2 + ' + str(self.b) + ' (Y+' + str(self.stage_shift) + ') + ' + str(self.c)
        
        else:
            equation = str(self.a) + ' (Y+' + str(self.stage_shift) + ')^' + str(self.b)
            
        return equation

class Hydrograph:
    def __init__(self, function = None):
        if function is not None:
            self.used_function = function
        else:
            self.used_function = self.interpolate_hydrograph
    
    def interpolate_hydrograph(self, time):
        if self.values is None:
            raise ValueError("Hydrograph is not defined.")
        
        if time > self.times[-1]:
            return self.values[-1]
                
        return float(np.interp(time, self.times, self.values))

    def get_at(self, time):
        return self.used_function(time)
        
    def set_values(self, times, values):
        times  = np.asarray(times,   dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if times.shape != values.shape:
            raise ValueError("Times and values must have the same length.")
        
        self.times, self.values = times, values
        
    def load_csv(self, path):
        import pandas as pd
        hydrograph_file = pd.read_csv(path, thousands=',')
        
        self.times  = np.asarray(hydrograph_file.iloc[:,0], dtype=np.float64)
        self.values = np.asarray(hydrograph_file.iloc[:,1], dtype=np.float64)
                    
    def set_function(self, func):
        self.used_function = func
    
class Hydraulics:
    def normal_flow(A, S_0, n, B):
        R = Hydraulics.R(A, B)
        
        Q = A * R**(2/3) * np.abs(S_0)**0.5 / n
        if S_0 < 0:
            Q = -Q
                        
        return Q
        
    def normal_area(Q, A_guess, S_0, n, B, tolerance = 1e-3):
        Q_guess = Hydraulics.normal_flow(A_guess, S_0, n, B)
            
        while abs(Q_guess - Q) >= tolerance:
            error = (Q_guess - Q) / Q
            A_guess -= 0.1 * error * A_guess
            Q_guess = Hydraulics.normal_flow(A_guess, S_0, n, B)
            
        return A_guess
    
    def effective_roughness(depth: float, steepness, roughness, dry_roughness, wet_depth):
        transition_depth = steepness * wet_depth
        
        if depth <= wet_depth:
            return roughness
        if transition_depth == 0 or depth - wet_depth > transition_depth:
            return dry_roughness
        else:
            return roughness + (dry_roughness - roughness) * (depth - wet_depth) / transition_depth
        
    def Sf(A: float, Q: float, n: float, B: float) -> float:
        """
        Computes the friction slope using Manning's equation.
        
        Parameters
        ----------
        A : float
            The cross-sectional flow area.
        Q : float
            The discharge.
            
        Returnsr
        -------
        float
            The computed friction slope.
            
        """
        R = Hydraulics.R(A, B)
        return n**2 * A**-2 * R**(-4/3) * Q * np.abs(Q)
    
    def Sc(A: float, Q: float, n: float, B: float, rc: float) -> float:
        """
        Computes the energy gradient due to transverse circulation.
        
        Parameters
        ----------
        A : float
            The cross-sectional flow area.
        Q : float
            The discharge.
            
        Returns
        -------
        float
            The computed friction slope.
            
        """
        h = A/B
        Fr = Hydraulics.froude_num(A, Q, B)
        R = Hydraulics.R(A, B)
        C = R**(1/6) / n
        f = 8 * g / C**2
        
        numerator = (2.86 * np.sqrt(f) + 2.07 * f) * h**2 * Fr**2
        denominator = (0.565 + np.sqrt(f)) * rc**2
        Sc = numerator/denominator
        return Sc
    
    def dSc_dA(A, Q, n, B, rc):
        h = A / B
        Fr = Hydraulics.froude_num(A, Q, B)
        R = Hydraulics.R(A, B)
        dR_dA = Hydraulics.dR_dA(A, B)
        C = R**(1/6) / n
        f = 8 * g / C**2
        
        dh_dA = 1.0 / B
        dFr_dA = -1.5 * Fr / A
        df_dA = -(8.0/3.0) * g * n**2 * R**(-4.0/3.0) * dR_dA
        
        sqrtf = np.sqrt(f)
        num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
        den = (0.565 + sqrtf) * rc**2
        
        dnum_dA = (2.86/(2*sqrtf)*df_dA + 2.07*df_dA) * h**2 * Fr**2 \
                + (2.86*sqrtf + 2.07*f) * (2*h*dh_dA * Fr**2 + h**2 * 2*Fr*dFr_dA)
        dden_dA = (1.0/(2*sqrtf) * df_dA) * rc**2
        
        return (dnum_dA*den - num*dden_dA) / (den**2)

    def dSc_dQ(A, Q, n, B, rc):
        h = A / B
        Fr = Hydraulics.froude_num(A, Q, B)
        R = Hydraulics.R(A, B)
        C = R**(1/6) / n
        f = 8 * g / C**2
        
        dFr_dQ = Fr / Q
        
        sqrtf = np.sqrt(f)
        num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
        den = (0.565 + sqrtf) * rc**2
        
        dnum_dQ = (2.86*sqrtf + 2.07*f) * h**2 * 2*Fr*dFr_dQ
        dden_dQ = 0.0
        
        return (dnum_dQ*den - num*dden_dQ) / (den**2)

    def dSc_dn(A, Q, n, B, rc):
        h = A / B
        Fr = Hydraulics.froude_num(A, Q, B)
        R = Hydraulics.R(A, B)
        C = R**(1/6) / n
        f = 8 * g / C**2
        
        df_dn = 16.0 * g * n / R**(1/3)
        
        sqrtf = np.sqrt(f)
        num = (2.86*sqrtf + 2.07*f) * h**2 * Fr**2
        den = (0.565 + sqrtf) * rc**2
        
        dnum_dn = (2.86/(2*sqrtf)*df_dn + 2.07*df_dn) * h**2 * Fr**2
        dden_dn = (1.0/(2*sqrtf) * df_dn) * rc**2
        
        return (dnum_dn*den - num*dden_dn) / (den**2)
    
    def froude_num(A: float, Q: float, B: float):
        V = Q / A
        h = A / B

        return V / np.sqrt(g*h)
    
    def dSf_dA(A: float, Q: float, n: float, B: float) -> float:
        """Computes the partial derivative of Sf w.r.t. A.

        Args:
            A (float): Cross-sectional flow area
            Q (float): Flow rate
            n (float): Manning's coefficient
            B (float): Cross-sectional width
            approx_R (bool, optional): Whether P (wetted perimeter) is approximated to B. Defaults to False.

        Returns:
            float: dSf/dA
        """
        R = Hydraulics.R(A, B)
        dSf_dA = -2 * n**2 * A**-3 * R**(-4/3) * Q * abs(Q)
        dSf_dR = (-4/3) * n**2 * A**-2 * R**(-4/3 - 1) * Q * abs(Q)

        return dSf_dA + dSf_dR * Hydraulics.dR_dA(A, B)
    
    def dSf_dQ(A: float, Q: float, n: float, B: float, approx_R = False) -> float:
        """Computes the partial derivative of Sf w.r.t. Q.

        Args:
            A (float): Cross-sectional flow area
            Q (float): Flow rate
            n (float): Manning's coefficient
            B (float): Cross-sectional width
            approx_R (bool, optional): Whether P (wetted perimeter) is approximated to B. Defaults to False.

        Returns:
            float: dSf/dQ
        """
        R = Hydraulics.R(A, B)
        
        d_Sf = 2 * abs(Q) * (n / (A * R**(2/3)))**2
        
        return d_Sf
    
    def R(A, B, approx = False):
        if approx:
            P = B
        else:
            P = B + 2 * A/B
                    
        return A / P
    
    def dR_dA(A, B, approx = False):
        """
        dR/dA where R = A / P and P = B + 2A/B (unless approx=True, then P=B).
        Correct formula: dR/dA = (P - A*dP/dA) / P**2
        """
        if approx:
            P = B
            dP_dA = 0.0
        else:
            P = B + 2.0 * A / B
            dP_dA = 2.0 / B
    
        return (P - A * dP_dA) / (P**2)
        
    def dSf_dn(A, Q, n, B):
        R = Hydraulics.R(A, B)
        return 2 * n * A**-2 * R**(-4/3) * Q * abs(Q)
    
    def dn_dh(depth: float, steepness: float, roughness: float, dry_roughness: float, wet_depth: float):
        transition_depth = steepness * wet_depth
        
        if depth <= wet_depth or depth - wet_depth > transition_depth:
            return 0
        else:
            return (dry_roughness - roughness) / transition_depth
        
    def dQn_dA(A, S, n, B):
        R = Hydraulics.R(A, B)
        dQn_dR = (2/3) * A * R**(2/3 - 1) * abs(S)**0.5 / n
        
        dQn_dA = R**(2/3) * abs(S)**0.5 / n + dQn_dR * Hydraulics.dR_dA(A, B)
        if S < 0:
            dQn_dA = -dQn_dA
            
        return dQn_dA
    
    def dQn_dn(A, S_0, n, B):
        R = Hydraulics.R(A, B)
        
        dQn_dn = -1 * A * R**(2/3) * abs(S_0)**0.5 * n**-2
        
        if S_0 < 0:
            dQn_dn = -dQn_dn
            
        return dQn_dn
    
class LumpedStorage:
    def __init__(self, surface_area: float, min_stage: float, solution_boundaries: tuple, rating_curve: RatingCurve = None):
        self.rating_curve = rating_curve
        self.surface_area = surface_area
        self.min_stage = min_stage
        self.stage = None
        self.area_curve = None
        
        self.Y_min = solution_boundaries[0]
        self.Y_max = solution_boundaries[1]
    
    def mass_balance(self, duration, vol_in, Y_old=None):
        from scipy.optimize import brentq
        
        if Y_old is None:
            Y_old = self.stage

        def f(Y_new):
            Q_out = 0.5 * (self.rating_curve.discharge(Y_old) + self.rating_curve.discharge(Y_new)) if self.rating_curve else 0.0
            target_vol = vol_in - Q_out * duration
            return self.net_vol_change(Y_old, Y_new) - target_vol

        Y_new = brentq(f, self.Y_min, self.Y_max)
        if Y_new < self.min_stage:
            Y_new = self.min_stage

        # update outflow based on actual storage change achieved
        vol_out = vol_in - self.net_vol_change(Y_old, Y_new) if self.rating_curve else 0.0
        return Y_new, vol_out

    def dY_new_dvol_in(self, duration, vol_in, Y_old) -> float:
        """
        d(Y_new)/d(vol_in)
        """
        Y_new, _ = self.mass_balance(duration, vol_in, Y_old)
        if Y_new <= self.min_stage:
            return 0.0
        
        return 1/self.area_at(Y_new)
            
    def set_area_curve(self, curve, alpha=1, beta=0, update_solution_boundaries = True):
        self.alpha = alpha
        self.beta = beta
        self.area_curve = np.asarray(curve, dtype=np.float64)
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
