from settings import US_INIT_DEPTH, US_INIT_DISCHARGE, US_INIT_STAGE, US_RATING_CURVE
from settings import PEAK_HOUR, PEAK_DISCHARGE
from settings import custom_hydrograph, CUSTOM_INFLOW

from settings import DS_INIT_DEPTH, DS_INIT_DISCHARGE, DS_INIT_STAGE, DS_RATING_CURVE
from settings import DS_CONDITION

import numpy as np

    
class Upstream:
    initial_depth = US_INIT_DEPTH
    initial_discharge = US_INIT_DISCHARGE   
    initial_stage = US_INIT_STAGE 
    
    @staticmethod
    def rating_curve(flow_depth: float) -> float:
        discharge = rating_curve(flow_depth, US_RATING_CURVE, US_INIT_STAGE - US_INIT_DEPTH)
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
        
        if CUSTOM_INFLOW:
            return custom_hydrograph(time)
        
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
    def rating_curve(flow_depth: float) -> float:
        discharge = rating_curve(flow_depth, DS_RATING_CURVE, DS_INIT_STAGE - DS_INIT_DEPTH)
        return discharge
    
    @staticmethod
    def condition_residual(A, Q):
        from settings import WIDTH as B
        h = A/B
        if DS_CONDITION == 'fixed_depth':
            return h - Downstream.initial_depth
        
        if DS_CONDITION == 'normal_depth':
            from settings import BED_SLOPE as S_0, MANNING_COEFF as n
            P = B + 2 * h
            manning_Q = (A ** (5./3) * S_0 ** 0.5) / (n * P ** (2./3))
            return Q - manning_Q
        
        if DS_CONDITION == 'rating_curve':
            return Q - Downstream.rating_curve(A/B)
    
    
    @staticmethod
    def derivative_wrt_A(A):
        from settings import WIDTH as B, BED_SLOPE as S_0, MANNING_COEFF as n
        if DS_CONDITION == 'fixed_depth':
            return 1./B
        
        if DS_CONDITION == 'normal_depth':
            P = B + 2 * A/B
            R = A/P
            return - S_0 ** 0.5 / n * (
                -4. / (3*B) * R ** (5./3) + 5./3 * R ** (2./3)
                )
        
        if DS_CONDITION == 'rating_curve':
            stage = DS_INIT_STAGE - DS_INIT_DEPTH + A/B
            return -(DS_RATING_CURVE["coefficients"][1] / B
            + DS_RATING_CURVE["coefficients"][2] * 2 * (stage - DS_RATING_CURVE["base"]) / B)
        
    @staticmethod
    def derivative_wrt_Q():
        if DS_CONDITION == 'fixed_depth':
            return 0
        
        if DS_CONDITION == 'normal_depth':
            return 1
        
        if DS_CONDITION == 'rating_curve':
            return 1
        
        
def rating_curve(flow_depth: float, eq_parameters, bed_stage) -> float:
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
        stage = bed_stage + flow_depth
        
        discharge = (
            eq_parameters["coefficients"][0]
            + eq_parameters["coefficients"][1] * (stage - eq_parameters["base"])
            + eq_parameters["coefficients"][2] * (stage - eq_parameters["base"]) ** 2
            )
                
        return discharge 
    