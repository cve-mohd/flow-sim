from initial_conditions import us_level, us_y
import numpy as np


def rating_curve(water_depth: float) -> float:
    """
    Computes the discharge for a given water depth using
    the rating curve equation of the GERD.

    Parameters
    ----------
    water_depth : float
        The water depth at the GERD in meters.

    Returns
    -------
    Q : float
        The computed discharge in cubic meters per second.

    """
    # Compute the water level.

    water_level = water_depth + (us_level - us_y)
    
    Q = (17376701.003616 - 69833.35989796935 * water_level
         + 70.16036993338325 * water_level**2)
    
    return Q
    

def inflow_hydrograph(time_in_seconds: float | int) -> float:
    """
    Computes the discharge through the GERD at a given time
    using a discharge hydrograph.

    Parameters
    ----------
    time_in_seconds : float | int
        The time in seconds.

    Returns
    -------
    float
        The computed discharge in cubic meters per second.

    """
    initial_discharge = 1562.5
    peak_discharge = 10000
    peak_hour = 6
    
    time = time_in_seconds/3600.
    angle = 0.5 * np.pi * time / peak_hour

    discharge = initial_discharge + (peak_discharge - initial_discharge) * np.sin(angle)
    
    if time >= peak_hour * 2:
        discharge = initial_discharge

    return max(discharge, initial_discharge)
