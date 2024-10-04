from sympy import symbols, Matrix
from river import River
import numpy as np


g = 9.81

class Model:
    def __init__(self,
                 river: River,
                 beta: int | float,
                 delta_t: int | float,
                 delta_x: int | float,
                 duration: int | float):
        
        self.beta = beta  # (b/w 0.5 and 1)
        
        self.river = river
        self.delta_t, self.delta_x = delta_t, delta_x
        self.duration = duration 
    
        # Flow variables (area and velocity):
        self.A_previous, self.V_previous, self.Sf_previous = [], [], []
        self.A_guess, self.V_guess = [], []
        self.results = []
        self.x0 = []
        self.timestep_parameters_values = {}
        self.guesses_values = {}
        self.Sf_ds, self.Q_in = symbols('Sf_ds Q_in')
        
        self.system = self.build_system()
        self.jacobian_matrix = self.build_jacobian()
        
    def build_system(self):
        n_nodes = int(self.river.length / self.delta_x + 1)
        B = self.river.width
        n = self.river.mannings_co
        S_0 = self.river.bed_slope
        
        for i in range(n_nodes):
            self.A_previous.append(symbols('A_prev' + str(i)))
            self.V_previous.append(symbols('V_prev' + str(i)))
            self.Sf_previous.append(symbols('Sf_prev' + str(i)))
        
            self.A_guess.append(symbols('self.A_guess' + str(i)))
            self.V_guess.append(symbols('self.V_guess' + str(i)))
        
        equations_list = [
            self.V_guess[0] * self.A_guess[0] - self.Q_in  # Eq. 25
            ]
        
        for i in range(n_nodes - 1):
            equations_list.append(
                self.A_guess[i+1] + self.A_guess[i]
                + 2 * self.delta_t / self.delta_x * (
                    self.beta * (
                        self.V_guess[i+1] * self.A_guess[i+1] - self.V_guess[i] * self.A_guess[i]
                    )
                    + (1 - self.beta) * (
                        self.V_previous[i+1] * self.A_previous[i+1] - self.V_previous[i] * self.A_previous[i]
                    )
                )
                - self.A_previous[i+1] - self.A_previous[i]
            )  # Eq. 26
        
            equations_list.append(
                self.V_guess[i+1] * self.A_guess[i+1]
                + self.V_guess[i] * self.A_guess[i]
                + 2 * self.delta_t / self.delta_x * (
                    self.beta * (
                        self.V_guess[i+1]**2 * self.A_guess[i+1] - self.V_guess[i]**2 * self.A_guess[i]
                        + g * (self.A_guess[i+1]**2 - self.A_guess[i]**2) / (2*B)
                    )
                    + (1 - self.beta) * (
                            self.V_previous[i+1]**2 * self.A_previous[i+1] - self.V_previous[i]**2 * self.A_previous[i]
                            + g * (self.A_previous[i+1]**2 - self.A_previous[i]**2) / (2*B)
                    )
                )
                + self.delta_t * (
                    self.beta * (
                        - g * self.A_guess[ i ] * (S_0 - n**2 * self.V_guess[ i ]**2 * (B + 2*self.A_guess[ i ]/B)**(4./3) / self.A_guess[ i ]**(4./3))
                        - g * self.A_guess[i+1] * (S_0 - n**2 * self.V_guess[i+1]**2 * (B + 2*self.A_guess[i+1]/B)**(4./3) / self.A_guess[i+1]**(4./3))
                    )
                    + (1 - self.beta) * (
                        - g * self.A_previous[ i ] * (S_0 - self.Sf_previous[ i ])
                        - g * self.A_previous[i+1] * (S_0 - self.Sf_previous[i+1])
                    )
                )
                - self.V_previous[i+1] * self.A_previous[i+1] - self.V_previous[i] * self.A_previous[i]
            )  # Eq. 27
        
        equations_list.append(
            self.V_guess[-1] - (1/n) * self.A_guess[-1]**(2./3)
            * self.Sf_ds**0.5 / (B + 2*self.A_guess[-1]/B)**(2./3)
        )  # Eq. 28
        
        return Matrix(equations_list)
        
    
    def build_jacobian(self):
        variables = []
        for i in range(len(self.A_guess)):
            variables.append(self.V_guess[i])
            variables.append(self.A_guess[i])
        
        variables = Matrix(variables)
        return self.system.jacobian(variables)
    

    def set_t0_values(self):
        y = 0.863797858 # calc'ed using inflow hydrograph + Manning's eq
        for x in range(len(self.A_guess)):
            self.timestep_parameters_values[self.V_previous[x]] = self.river.inflow_Q(0) / (self.river.width*y)
            self.timestep_parameters_values[self.A_previous[x]] = self.river.width*y
            self.timestep_parameters_values[self.Sf_previous[x]] = self.river.friction_slope(self.timestep_parameters_values[self.V_previous[x]], self.timestep_parameters_values[self.A_previous[x]])
        
        
    def update_parameters(self):
        for x in range(len(self.A_guess)):
            self.timestep_parameters_values[self.V_previous[x]] = self.x0[ 2*x ]
            self.timestep_parameters_values[self.A_previous[x]] = self.x0[2*x+1]
            self.timestep_parameters_values[self.Sf_previous[x]] = self.river.friction_slope(self.timestep_parameters_values[self.V_previous[x]], self.timestep_parameters_values[self.A_previous[x]])
                    
        
    def initialize_x0(self):
        x0 = []
        for x in range(len(self.V_previous)):
            x0.append(self.timestep_parameters_values[self.V_previous[x]])
            x0.append(self.timestep_parameters_values[self.A_previous[x]])
        
        return Matrix(x0)
        
        
    def update_guesses(self):
        for x in range(len(self.A_guess)):
            self.guesses_values[self.V_guess[x]] = self.x0[ 2*x ]            
            self.guesses_values[self.A_guess[x]] = self.x0[2*x+1]
                    
        self.guesses_values[self.Sf_ds] = self.river.friction_slope(self.guesses_values[self.V_guess[-1]], self.guesses_values[self.A_guess[-1]])
        
        
    def run(self):
        self.set_t0_values()
        self.x0 = self.initialize_x0()
        self.results.append(self.x0.tolist())
    
        for time in range(self.delta_t, self.duration + 1, self.delta_t):
            
            print('\n---------- Time = ' + str(time) + 's ----------\n')
                    
            
            self.timestep_parameters_values[self.Q_in] = self.river.inflow_Q(time/3600.)
            
            jacobian = self.jacobian_matrix.subs(self.timestep_parameters_values)
            Fx = self.system.subs(self.timestep_parameters_values)
                    
            iteration = 0
            tolerance = 10**-4
            cumulative_error = tolerance
            
            while cumulative_error >= tolerance:
                iteration += 1
                print("--- Iteration #" + str(iteration) + ':')
                
                self.update_guesses()
                
                j = jacobian.subs(self.guesses_values)
                f = Fx.subs(self.guesses_values)
                
                j_inverse = j.inv()
                
                change = -j_inverse * f
                self.x0 = self.x0 + change
                
                change = [abs(i) for i in change]
                cumulative_error = np.sum(np.array(change))
                
                print("Error = " + str(cumulative_error))
        
            self.results.append(self.x0.tolist())
            self.update_parameters()
            
            
    def save_results(self):
        y, v = [], []
        for x in self.results:
            yx, vx = [], []
            for i in range(0, len(x), 2):
                vx.append(x[i][0])
                yx.append(x[i+1][0] / self.river.width)
                
            y.append(yx)
            v.append(vx)
                
        y, v = str(y), str(v)
        
        y = y.replace('], [', '\n')
        v = v.replace('], [', '\n')
        for c in "[]' ":
            y = y.replace(c, '')
            v = v.replace(c, '')
            
        with open('depth.csv', 'w') as output_file:
            output_file.write(y)
            
        with open('velocity.csv', 'w') as output_file:
            output_file.write(v)
            