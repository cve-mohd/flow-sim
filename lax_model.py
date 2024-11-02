import numpy as np
from river import River

g = 9.81

class LaxModel:
    def __init__(self,
                 river: River,
                 delta_t: int | float,
                 delta_x: int | float,
                 duration: int | float):
        
        
        self.river = river
        self.delta_t, self.delta_x = delta_t, delta_x
        self.celerity = self.delta_x / float(self.delta_t)
        
        self.n_nodes = int(self.river.length / self.delta_x + 1)
        self.duration = duration 
    
        self.A_prev = []
        self.Q_prev = []
        
        self.A_current = []
        self.Q_current = []
        
        self.resultsA = []
        self.resultsQ = []
        
        self.initialize_t0()
        
        
    def initialize_t0(self):
        self.river.initialize_(self.delta_x)
        
        for x, (A, Q) in enumerate(self.river.initial_conditions):
            self.A_prev.append(A)
            self.Q_prev.append(Q)
            
            self.A_current.append(0)
            self.Q_current.append(0)
            
        self.resultsA.append(self.A_prev)
        self.resultsQ.append(self.Q_prev)
    
    
    def solve(self):
        W = self.river.width
        n = self.river.mannings_co
        S_0 = self.river.bed_slope
        
        for time in range(self.delta_t, self.duration + 1, self.delta_t):            
            for i in range(1, self.n_nodes - 1):
                self.A_current[i] = 0.5 * (self.A_prev[i+1] + self.A_prev[i-1]) - (0.5/self.celerity) * (self.Q_prev[i+1] - self.Q_prev[i-1])
                self.Q_current[i] = (-g/(4*W*self.celerity) * (self.A_prev[i+1]**2 - self.A_prev[i-1]**2)
                                     + 0.5*g*S_0*self.delta_t * (self.A_prev[i+1] + self.A_prev[i-1])
                                     + 0.5 * (self.Q_prev[i+1] + self.Q_prev[i-1])
                                     - (0.5/self.celerity) * (self.Q_prev[i+1]**2/self.A_prev[i+1] - self.Q_prev[i-1]**2/self.A_prev[i-1])
                                     - 0.5*g*W**(4/3.)*n**2*self.delta_t * (self.Q_prev[i+1]**2/self.A_prev[i+1]**(7/3.) + self.Q_prev[i-1]**2/self.A_prev[i-1]**(7/3.)))


            self.Q_current[0] = self.river.inflow_Q(time/3600.)
            self.A_current[0] = self.A_prev[0] - (1/self.celerity) * (self.Q_prev[1] - self.Q_prev[0])
            #nodes_for_extrapo = 5
            #self.A_current[0] = self.quadratic_extrapolation([i for i in range(2, 2+nodes_for_extrapo)], self.A_current[1:1+nodes_for_extrapo], 1) # quad
            
            self.A_current[-1] = self.A_prev[-1] - (1/self.celerity) * (self.Q_prev[-1] - self.Q_prev[-2])
            #self.A_current[-1] = self.quadratic_extrapolation([i for i in range(self.n_nodes-nodes_for_extrapo, self.n_nodes)], self.A_current[-1-nodes_for_extrapo:-1], self.n_nodes) # quad
            self.Q_current[-1] = ((1/n) * self.A_current[-1] ** (5./3) * self.river.friction_slope(self.A_current[-2], self.Q_current[-2]) ** 0.5
            / (W + 2*self.A_current[-1]/W) ** (2./3))


            self.A_prev = [a for a in self.A_current]
            self.Q_prev = [q for q in self.Q_current]

            self.resultsA.append(self.A_prev)
            self.resultsQ.append(self.Q_prev)


    def quadratic_extrapolation(self, indices, values, target_index):
        # Ax = B
        A = [np.array(indices)**i for i in range(len(indices)-1, 0, -1)]
                        
        A.append(np.ones(len(indices)))
        A = np.vstack(A).T
        B = np.array(values)
    
        coeffs = np.linalg.solve(A, B)
    
        target_value = 0
        for i in range(len(indices)):
            target_value += coeffs[i] * target_index ** (len(indices) - i - 1)
            
        return target_value        
        
        
    def save_results(self, time_steps_to_save):
        A = self.resultsA[::len(self.resultsA) // (time_steps_to_save-1)]
        Q = self.resultsQ[::len(self.resultsQ) // (time_steps_to_save-1)]
        
        A, Q = str(A), str(Q)
        
        A = A.replace('], [', '\n')
        Q = Q.replace('], [', '\n')
        for c in "[]' ":
            A = A.replace(c, '')
            Q = Q.replace(c, '')
            
        with open('area.csv', 'w') as output_file:
            output_file.write(A)
            
        with open('discharge.csv', 'w') as output_file:
            output_file.write(Q)
            