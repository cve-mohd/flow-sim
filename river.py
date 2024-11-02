from math import pi, sin, cos


class River:
    def __init__(self, bed_slope: float, mannings_co: float, width: int | float, length: int | float):
        self.bed_slope = bed_slope        
        self.mannings_co = mannings_co
        self.width = width
        self.length = length
        self.initial_conditions = []

    def friction_slope(self, A: int | float, Q: int | float):
        return (Q * self.mannings_co * (self.width + 2*A/self.width)**(2./3) / A**(5./3))**2
        
    def inflow_Q(self, t_hours: int | float):
        Qb, Qp = 100, 200
        tb, tp = 15, 5
    
        if t_hours <= tp:
            return Qp/2 * sin(pi*t_hours/tp - pi/2) + Qp/2 + Qb
        elif t_hours <= tb:
            return Qp/2 * cos(pi * (t_hours-tp)/(tb-tp)) + Qp/2 + Qb
        else:
            return Qb
        
        
    def initialize_(self, delta_x):
        n_nodes = int(self.length / delta_x + 1)
        
        A = 0.863797858 * self.width # calc'ed using inflow hydrograph + Manning's eq
        Q = self.inflow_Q(0)
        
        self.initial_conditions = [(A, Q) for i in range(n_nodes)]
        