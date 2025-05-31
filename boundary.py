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
    bed_level = initial_stage - initial_depth
    
    max_service_stage = 493
    max_q = None
    base_q = None
    q_out = None
    
    fail = False
    fixed_depth = DS_INIT_DEPTH
    reservoir_area = 318880455 - 250 * 15000
        
    @staticmethod
    def rating_curve(stage: float) -> float:
        discharge = rating_curve(stage, DS_RATING_CURVE)
        return discharge
    
    @staticmethod
    def inverse_rating_curve(discharge: float) -> float:
        stage = inverse_rating_curve(discharge, DS_RATING_CURVE, 490)
        return stage
    
    @staticmethod
    def condition_residual(A, Q):
        from settings import WIDTH as B
        h = A/B[-1]
        if DS_CONDITION == 'fixed_depth':
            return h - Downstream.fixed_depth
        
        if DS_CONDITION == 'normal_depth':
            from settings import BED_SLOPE as S_0, MANNING_COEFF as n
            P = B[-1] + 2 * h
            manning_Q = (A ** (5./3) * S_0[-1] ** 0.5) / (n[-1] * P ** (2./3))
            return Q - manning_Q
        
        if DS_CONDITION == 'rating_curve':
            return Q - Downstream.rating_curve(Downstream.bed_level + A/B[-1])
    
    
    def update_fixed_depth(q_in, delta_t):
        """
        Updates the water depth at the boundary in case of fixed-depth boundary condition.

        Parameters
        ----------
        q_in : float
            Rate of flow at the boundary.
            
        delta_t : float
            Temporal step.

        Returns
        -------
        None.

        """
        
        if Downstream.max_q is None:
            Downstream.max_q = Downstream.rating_curve(Downstream.max_service_stage)
            Downstream.base_q = Downstream.rating_curve(Downstream.initial_stage)
            Downstream.q_out = Downstream.rating_curve(Downstream.initial_stage)
        
        current_stage = Downstream.fixed_depth + Downstream.bed_level
        
        if q_in > Downstream.q_out or current_stage > Downstream.initial_stage:
            added_vol = (q_in - Downstream.q_out) * delta_t
            Downstream.fixed_depth += added_vol / Downstream.reservoir_area
            
            if Downstream.fixed_depth < Downstream.initial_depth:
                Downstream.fixed_depth = Downstream.initial_depth
                
            current_stage = Downstream.fixed_depth + Downstream.bed_level
            
            Downstream.q_out = Downstream.rating_curve(current_stage)
            
            Downstream.fail = (current_stage > Downstream.max_service_stage)
            
        else:
            Downstream.fixed_depth = Downstream.initial_depth
    
    @staticmethod
    def derivative_wrt_A(A):
        from settings import WIDTH as B, BED_SLOPE as S_0, MANNING_COEFF as n
        if DS_CONDITION == 'fixed_depth':
            return 1./B[-1]
        
        if DS_CONDITION == 'normal_depth':
            P = B[-1] + 2 * A/B[-1]
            R = A/P
            return - S_0[-1] ** 0.5 / n[-1] * (
                -4. / (3*B[-1]) * R ** (5./3) + 5./3 * R ** (2./3)
                )
        
        if DS_CONDITION == 'rating_curve':
            stage = Downstream.bed_level + A/B[-1]
            return -(DS_RATING_CURVE["coefficients"][1] / B[-1]
            + DS_RATING_CURVE["coefficients"][2] * 2 * (stage - DS_RATING_CURVE["base"]) / B[-1])
        
    @staticmethod
    def derivative_wrt_Q():
        if DS_CONDITION == 'fixed_depth':
            return 0
        
        if DS_CONDITION == 'normal_depth':
            return 1
        
        if DS_CONDITION == 'rating_curve':
            return 1
        
        
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
    
    q = rating_curve(trial_stage, eq_parameters)
    
    while abs(q - discharge) > tolerance:
        error = (q - discharge) / discharge
        trial_stage -= 0.1 * error * trial_stage
        q = rating_curve(trial_stage, eq_parameters) 
    
    return trial_stage
