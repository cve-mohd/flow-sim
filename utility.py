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
    def __init__(self, a, b, c, base):
        self.a, self.b, self.c, self.base = a, b, c, base
        self.defined = True
        
    
    def __init__(self):
        self.defined = False
    
    
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
        
        x = stage - self.base
        discharge = self.a * x ** 2 + self.b * x + self.c
                
        return discharge
    
    
    def stage(self, discharge: float, tolerance: float = 1e-3) -> float:
        if not self.defined:
            raise ValueError("Rating curve is undefined.")
        
        trial_stage = self.base * 1.05
        
        q = self.discharge(stage=trial_stage)
        
        while abs(q - discharge) > tolerance:
            error = (q - discharge) / discharge
            trial_stage -= 0.1 * error * trial_stage
            q = self.discharge(stage=trial_stage)
        
        return trial_stage
    
    
    def fit(self, discharges, stages):
        from numpy import min, polyfit
        
        if len(discharges) < 3:
            raise ValueError("Need at least 3 points.")
        
        if len(discharges) != len(stages):
            raise ValueError("Q and Y lists should have the same lengths.")
        
        self.base = min(stages) * 0.9

        Y_shifted = stages - self.base

        self.a, self.b, self.c = polyfit(Y_shifted, discharges, deg=2)

        self.defined = True
        
        
    def discharge_derivative(self, stage, width):
        return self.a * 2 * (stage - self.base) * (1. / width) + self.b * (1. / width)
    