from settings import *
import numpy as np


class Upstream:
    initial_depth = US_INIT_DEPTH
    initial_discharge = US_INIT_DISCHARGE   
    initial_stage = US_INIT_STAGE 
    
    @staticmethod
    def rating_curve(self, water_depth: float) -> float:
        """
        Computes the discharge for a given water depth using
        the rating curve equation of the upstream boundary.

        Parameters
        ----------
        water_depth : float
            The water depth at the upstream boundary in meters.

        Returns
        -------
        Q : float
            The computed discharge in cubic meters per second.

        """
        stage = US_INIT_STAGE + (water_depth - US_INIT_DEPTH)
        
        Q = (US_RATING_CURVE_COEFF[0] + US_RATING_CURVE_COEFF[1] * stage
            + US_RATING_CURVE_COEFF[2] * stage ** 2)
        
        return Q
    
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
        """
        Computes the discharge for a given water depth using
        the rating curve equation of the downstream boundary.

        Parameters
        ----------
        water_depth : float
            The water depth at the downstream boundary in meters.

        Returns
        -------
        Q : float
            The computed discharge in cubic meters per second.

        """
        stage = DS_INIT_STAGE + (water_depth - DS_INIT_DEPTH)
        
        Q = (DS_RATING_CURVE_COEFF[0] + DS_RATING_CURVE_COEFF[1] * stage
            + DS_RATING_CURVE_COEFF[2] * stage ** 2)
        
        return Q
        