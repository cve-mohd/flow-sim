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
    
    
    def stage(self, discharge: float, trial_stage: float = None, tolerance: float = 1e-3, rate=1) -> float:
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