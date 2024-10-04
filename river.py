from math import pi, sin, cos


class River:
    def __init__(self, bed_slope: float, mannings_co: float, width: int | float, length: int | float):
        self.bed_slope = bed_slope        
        self.mannings_co = mannings_co
        self.width = width
        self.length = length

    def friction_slope(self, v: int | float, A: int | float):
        P = self.width + 2*(A/self.width)
        R = A/P
        return self.mannings_co**2 * v**2 / R**(4./3)
        
    def inflow_Q(self, t_hours: int | float):
        Qb, Qp = 100, 200
        tb, tp = 15, 5
    
        if t_hours <= tp:
            return Qp/2 * sin(pi*t_hours/tp - pi/2) + Qp/2 + Qb
        elif t_hours <= tb:
            return Qp/2 * cos(pi * (t_hours-tp)/(tb-tp)) + Qp/2 + Qb
        else:
            return Qb