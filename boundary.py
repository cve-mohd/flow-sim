from settings import *
import numpy as np

    
class Upstream:
    initial_depth = US_INIT_DEPTH
    initial_discharge = US_INIT_DISCHARGE   
    initial_stage = US_INIT_STAGE 
    
    @staticmethod
    def rating_curve(water_depth: float) -> float:
        discharge = rating_curve(water_depth, US_RATING_CURVE, US_INIT_STAGE - US_INIT_DEPTH)
        return discharge
    
    @staticmethod
    def inflow_hydrograph(time: float | int) -> float:
        """
        Computes the discharge through the upstream boundary at a
        given time using a discharge hydrograph.

        Parameters
        ----------
        time : float | int
            The time in seconds.

        Returns
        -------
        float
            The computed discharge in cubic meters per second.

        """
        
        angle = 0.5 * np.pi * time / (PEAK_HOUR * 3600)

        discharge = US_INIT_DISCHARGE + (PEAK_DISCHARGE - US_INIT_DISCHARGE) * np.sin(angle)
        
        if time >= 3600 * PEAK_HOUR * 2:
            discharge = US_INIT_DISCHARGE

        return max(discharge, US_INIT_DISCHARGE)


class Downstream:
    initial_depth = DS_INIT_DEPTH
    initial_discharge = DS_INIT_DISCHARGE  
    initial_stage = DS_INIT_STAGE 
        
    @staticmethod
    def rating_curve(water_depth: float) -> float:
        discharge = rating_curve(water_depth, DS_RATING_CURVE, DS_INIT_STAGE - DS_INIT_DEPTH)
        return discharge
        
        
def rating_curve(water_depth: float, eq_parameters, bed_stage) -> float:
        """
        Computes the discharge for a given water depth using
        the rating curve equation of the upstream boundary.

        Parameters
        ----------
        water_depth : float
            The water depth at the upstream boundary in meters.

        Returns
        -------
        discharge : float
            The computed discharge in cubic meters per second.

        """
        stage = bed_stage + water_depth
        
        discharge = (
            eq_parameters["coefficients"][0]
            + eq_parameters["coefficients"][1] * (stage - eq_parameters["base"])
            + eq_parameters["coefficients"][2] * (stage - eq_parameters["base"]) ** 2
            )
        
        return discharge
    