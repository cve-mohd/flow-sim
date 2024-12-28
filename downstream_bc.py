from initial_conditions import ds_level, ds_y


def rating_curve(water_depth: float) -> float:
    """
    Computes the discharge for a given water depth using
    the rating curve equation of the Roseires Dam.

    Parameters
    ----------
    water_depth : float
        The water depth at the Roseires Dam reservoir in meters.

    Returns
    -------
    Q : float
        The computed discharge in cubic meters per second.

    """

    # Compute the water level.
    water_level = water_depth + (ds_level - ds_y)
    
    Q = (-786548.06 + 2936.794642859 * water_level
         - 2.643543956 * water_level ** 2)
    
    return Q
    