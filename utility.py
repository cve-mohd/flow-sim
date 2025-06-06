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
    
    
    def rating_curve(stage: float, eq_parameters) -> float:
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
        discharge = (
            eq_parameters["coefficients"][0]
            + eq_parameters["coefficients"][1] * (stage - eq_parameters["base"])
            + eq_parameters["coefficients"][2] * (stage - eq_parameters["base"]) ** 2
            )
                
        return discharge
    
    
    def inverse_rating_curve(discharge: float, eq_parameters, stage_guess: float, tolerance: float = 1e-3) -> float:
        trial_stage = stage_guess
        
        q = Utility.rating_curve(trial_stage, eq_parameters)
        
        while abs(q - discharge) > tolerance:
            error = (q - discharge) / discharge
            trial_stage -= 0.1 * error * trial_stage
            q = Utility.rating_curve(trial_stage, eq_parameters) 
        
        return trial_stage
