import os
from numpy import sum, abs, square
from numpy import min, polyfit, array, log, exp
from numpy.polynomial.polynomial import Polynomial
import numpy as np


class Utility:
    def __init__():
        pass
    
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
        return sum(abs(vector))
    
    def euclidean_norm(vector):
        return sum(square(vector)) ** 0.5
    
    def seconds_to_hms(seconds):
        if seconds < 0:
            return "0:00:00"
        
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

        
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
                discharge = self.a * x ** 2 + self.b * x + self.c
                
            else:
                discharge = self.a * x ** self.b
                    
            return float(discharge)
    
    
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
            #print(trial_stage)
            
            q = self.discharge(stage=trial_stage)
        
        return trial_stage
    
    
    def fit(self, discharges: list, stages: list, stage_shift: float=0, type: str='polynomial', scale=True, degree: int=2):
        self.type = type
        
        if len(discharges) < 3:
            raise ValueError("Need at least 3 points.")
        
        if len(discharges) != len(stages):
            raise ValueError("Q and Y lists should have the same lengths.")
        
        discharges = array(discharges, dtype=float)
        stages = array(stages, dtype=float)
            
        self.stage_shift = stage_shift
        shifted_stages = stages + self.stage_shift
                
        if any(shifted_stages <= 0):
            raise ValueError("All (stage - base) values must be positive for power-law fitting.")

        if type == 'polynomial':
            if scale:
                self.function = Polynomial.fit(x=shifted_stages, y=discharges, deg=degree)
                self.derivative = self.function.deriv()
            
            else:
                if degree != 2:
                    print("WARNING: Polynomial degree defaults to 2 for unscaled fitting.")
                    
                a, b, c = polyfit(shifted_stages, discharges, deg=2)
                
                self.a = float(a)
                self.b = float(b)
                self.c = float(c)
            
        elif type == 'power':
            log_Y = log(shifted_stages)
            log_Q = log(discharges)

            # Fit: log(Q) = b * log(shifted_stages) + log(a)
            b, log_a = polyfit(log_Y, log_Q, deg=1)
            a = exp(log_a)

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
            d = self.a * self.b * Y_ ** (self.b - 1)
            
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
                
        from numpy import interp        
        return float(interp(time, self.times, self.values))

    def get_at(self, time):
        return self.used_function(time)
        
    def set_values(self, times, values):
        if len(times) != len(values):
            raise ValueError("Times and values must have the same length.")
        
        self.times, self.values = times, values
        
    def load_csv(self, path):
        import pandas as pd
        hydrograph_file = pd.read_csv(path, thousands=',')
        
        self.times = hydrograph_file.iloc[:,0].astype(float).tolist()
        self.values = hydrograph_file.iloc[:,1].astype(float).tolist()
                    
    def set_function(self, func):
        self.used_function = func
    
class Hydraulics:
    def normal_flow(A, S, n, B):
        R = Hydraulics.R(A, B)
        
        Q = A * R ** (2./3) * abs(S) ** 0.5 / n
        if S < 0:
            Q = -Q
            
        return Q
        
    def normal_area(Q, A_guess, S_0, n, B, tolerance = 1e-3):
        Q_guess = Hydraulics.normal_flow(A_guess, S_0, n, B)
            
        while abs(Q_guess - Q) >= tolerance:
            error = (Q_guess - Q) / Q
            A_guess -= 0.1 * error * A_guess
            Q_guess = Hydraulics.normal_flow(A_guess, S_0, n, B)
            
        return A_guess
    
    def effective_roughness(depth: float, steepness, channel_roughness, floodplain_roughness, bankful_depth):
        if floodplain_roughness is None or bankful_depth is None:
            return channel_roughness
        
        transition_depth = steepness * bankful_depth
        
        if depth <= bankful_depth:
            return channel_roughness
        if transition_depth == 0 or depth - bankful_depth > transition_depth:
            return floodplain_roughness
        else:
            return channel_roughness + (floodplain_roughness - channel_roughness) * (depth - bankful_depth) / transition_depth
        
    def Sf(A: float, Q: float, n: float, B: float, approx_R = False) -> float:
        """
        Computes the friction slope using Manning's equation.
        
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
        R = Hydraulics.R(A, B)
        return n ** 2 * A ** -2 * R ** (-4. / 3) * Q * abs(Q)
    
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
        
        dSf_dA = -2 * n ** 2 * A ** -3 * R ** (-4. / 3) * Q * abs(Q)
        dSf_dR = (-4./3) * n ** 2 * A ** -2 * R ** (-4./3 - 1) * Q * abs(Q)

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
        
        d_Sf = 2 * abs(Q) * (n / (A * R ** (2. / 3))) ** 2
        
        return d_Sf
    
    def R(A, B, approx = False):
        if approx:
            P = B
        else:
            P = B + 2. * A / B
                    
        return A / P
    
    def dR_dA(A, B, approx = False):
        if approx:
            P = B
            dP_dA = 0
        else:
            P = B + 2. * A / B
            dP_dA = 2. / B
                
        dR_dP = - A / P ** 2
        
        return dR_dP * dP_dA
        
    def dSf_dn(A, Q, n, B):
        R = Hydraulics.R(A, B)
        return 2 * n * A ** -2 * R ** (-4. / 3) * Q * abs(Q)
    
    def dn_dh(depth: float, steepness, channel_roughness, floodplain_roughness, bankful_depth):
        if floodplain_roughness is None or bankful_depth is None:
            return 0
        
        transition_depth = steepness * bankful_depth
        
        if depth <= bankful_depth or depth - bankful_depth > transition_depth:
            return 0
        else:
            return (floodplain_roughness - channel_roughness) / transition_depth
        
    def dQn_dA(A, S, n, B):
        R = Hydraulics.R(A, B)
        dQn_dR = (2./3) * A * R ** (2./3 - 1) * abs(S) ** 0.5 / n
        
        dQn_dA = R ** (2./3) * abs(S) ** 0.5 / n + dQn_dR * Hydraulics.dR_dA(A, B)
        if S < 0:
            dQn_dA = -dQn_dA
            
        return dQn_dA
    