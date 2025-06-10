import os
from numpy import sum, abs, square


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
        self.defined = False
        self.type = None    
    
    
    def set(self, type, a, b, c=0, base=0):
        if type not in ['power', 'quadratic']:
            raise ValueError("Invalid rating curve type.")
        
        self.a, self.b, self.c, self.base_stage = a, b, c, base
        self.defined = True
        self.type = type
        
    
    def discharge(self, stage):
        """
        Computes the discharge for a given flow depth using
        the rating curve equation of the upstream boundary.

        Parameters
        ----------
        flow_depth : float
            The flow depth at the upstream boundary in meters.

        Returns
        -------
        discharge : float
            The computed discharge in cubic meters per second.

        """
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        x = stage - self.base_stage
        
        if self.type == 'quadratic':
            discharge = self.a * x ** 2 + self.b * x + self.c
            
        else:
            discharge = self.a * x ** self.b
                
        return float(discharge)
    
    
    def stage(self, discharge: float, tolerance: float = 1e-3) -> float:
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        trial_stage = self.base_stage * 1.05
        
        q = self.discharge(stage=trial_stage)
        
        while abs(q - discharge) > tolerance:
            error = (q - discharge) / discharge
            trial_stage -= 0.1 * error * trial_stage
            q = self.discharge(stage=trial_stage)
        
        return trial_stage
    
    
    def fit(self, discharges: list, stages: list, base: float=0, type='quadratic'):
        self.type = type
        from numpy import min, polyfit, array, log, exp
        
        if len(discharges) < 3:
            raise ValueError("Need at least 3 points.")
        
        if len(discharges) != len(stages):
            raise ValueError("Q and Y lists should have the same lengths.")
        
        discharges = array(discharges, dtype=float)
        stages = array(stages, dtype=float)
            
        self.base_stage = base
        Y_shifted = stages - self.base_stage
                
        if any(Y_shifted <= 0):
            raise ValueError("All (stage - base) values must be positive for power-law fitting.")

        if type == 'quadratic':
            a, b, c = polyfit(Y_shifted, discharges, deg=2)
            
            self.a = float(a)
            self.b = float(b)
            self.c = float(c)
            
        elif type == 'power':
            log_Y = log(Y_shifted)
            log_Q = log(discharges)

            # Fit: log(Q) = b * log(Y_shifted) + log(a)
            b, log_a = polyfit(log_Y, log_Q, deg=1)
            a = exp(log_a)

            self.a = float(a)
            self.b = float(b)
            
        else:
            raise ValueError("Invalid rating curve type.")
        
        self.defined = True
        
        
    def discharge_derivative(self, stage, width):
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        if self.type == 'quadratic':
            d = self.a * 2 * (stage - self.base_stage) * (1. / width) + self.b * (1. / width)
        else:    
            d = self.a * self.b * (stage - self.base_stage) ** (self.b - 1) * (1. / width)
            
        return d
    
    
    def tostring(self):
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        if self.type == 'quadratic':
            equation = str(self.a) + ' (Y-' + str(self.base_stage) + ')^2 + ' + str(self.b) + ' (Y-' + str(self.base_stage) + ') + ' + str(self.c)
        else:
            equation = str(self.a) + ' (Y-' + str(self.base_stage) + ')^' + str(self.b)
            
        return equation