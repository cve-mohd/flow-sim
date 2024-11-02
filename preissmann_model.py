from sympy import symbols, Matrix
from river import River
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


g = 9.81


class PreissmannModel:
    def __init__(self,
                 river: River,
                 beta: int | float,
                 delta_t: int | float,
                 delta_x: int | float,
                 duration: int | float):
        
        self.beta = beta  # (b/w 0.5 and 1)
        
        self.river = river
        self.delta_t, self.delta_x = delta_t, delta_x
        self.celerity = self.delta_x / float(self.delta_t)
        self.n_nodes = int(self.river.length / self.delta_x + 1)
        self.duration = duration 
    
        self.A_previous = [symbols('A_prev' + str(i)) for i in range(self.n_nodes)]
        self.Q_previous = [symbols('Q_prev' + str(i)) for i in range(self.n_nodes)]
        self.A_guess = [symbols('A_guess' + str(i)) for i in range(self.n_nodes)]
        self.Q_guess = [symbols('Q_guess' + str(i)) for i in range(self.n_nodes)]
        
        self.results = []
        
        self.x0 = None
        self.timestep_subs = {}
        self.iteration_subs = {}
        self.Sf_ds, self.Q_in = symbols('Sf_ds Q_in')
        self.Sf_previous = [symbols('Sf_prev' + str(i)) for i in range(self.n_nodes)]
        
        self.system = self.build_system()
        self.initialize_t0()
        self.symbolic_jacobian = None
        self.jacobian_sparse_indices = []
        self.build_jacobian()
        
        
    def build_system(self):
        W = self.river.width
        n = self.river.mannings_co
        S_0 = self.river.bed_slope
        
        equation_list = []
        """
        equation_list = [
            self.Q_guess[0] * self.A_guess[0] - self.Q_in  # Eq. 25
            ]
        
        for i in range(self.n_nodes - 1):
            equations_list.append(
                self.A_guess[i+1] + self.A_guess[i]
                + 2 * self.delta_t / self.delta_x * (
                    self.beta * (
                        self.Q_guess[i+1] * self.A_guess[i+1] - self.Q_guess[i] * self.A_guess[i]
                    )
                    + (1 - self.beta) * (
                        self.Q_previous[i+1] * self.A_previous[i+1] - self.Q_previous[i] * self.A_previous[i]
                    )
                )
                - self.A_previous[i+1] - self.A_previous[i]
            )  # Eq. 26
        
            equation_list.append(
                self.Q_guess[i+1] * self.A_guess[i+1]
                + self.Q_guess[i] * self.A_guess[i]
                + 2 * self.delta_t / self.delta_x * (
                    self.beta * (
                        self.Q_guess[i+1]**2 * self.A_guess[i+1] - self.Q_guess[i]**2 * self.A_guess[i]
                        + g * (self.A_guess[i+1]**2 - self.A_guess[i]**2) / (2*B)
                    )
                    + (1 - self.beta) * (
                            self.Q_previous[i+1]**2 * self.A_previous[i+1] - self.Q_previous[i]**2 * self.A_previous[i]
                            + g * (self.A_previous[i+1]**2 - self.A_previous[i]**2) / (2*B)
                    )
                )
                + self.delta_t * (
                    self.beta * (
                        - g * self.A_guess[ i ] * (S_0 - n**2 * self.Q_guess[ i ]**2 * (B + 2*self.A_guess[ i ]/B)**(4./3) / self.A_guess[ i ]**(4./3))
                        - g * self.A_guess[i+1] * (S_0 - n**2 * self.Q_guess[i+1]**2 * (B + 2*self.A_guess[i+1]/B)**(4./3) / self.A_guess[i+1]**(4./3))
                    )
                    + (1 - self.beta) * (
                        - g * self.A_previous[ i ] * (S_0 - self.Sf_previous[ i ])
                        - g * self.A_previous[i+1] * (S_0 - self.Sf_previous[i+1])
                    )
                )
                - self.Q_previous[i+1] * self.A_previous[i+1] - self.Q_previous[i] * self.A_previous[i]
            )  # Eq. 27
        
        equation_list.append(
            self.V_guess[-1] - (1/n) * self.A_guess[-1] ** (2./3)
            * self.Sf_ds ** 0.5 / (B + 2*self.A_guess[-1]/B) ** (2./3)
        )  # Eq. 28
        """
        equation_list.append(
            self.Q_guess[0] - self.Q_in
        )
        
        for i in range(self.n_nodes - 1):
            equation_list.append(
                self.celerity * (self.A_guess[ i ] + self.A_guess[i+1])
                + 2*self.beta * (self.Q_guess[i+1] - self.Q_guess[ i ])
                - (
                    self.celerity * (self.A_previous[i+1] + self.A_previous[ i ])
                    - 2*(1-self.beta) * (self.Q_previous[i+1] - self.Q_previous[ i ])
                  )
            )
            
            equation_list.append(
                (g*self.beta/W) * (self.A_guess[i+1]**2 - self.A_guess[i]**2)
                - self.delta_x*g*self.beta*S_0 * (self.A_guess[i+1] + self.A_guess[i])
                + self.celerity * (self.Q_guess[i+1] + self.Q_guess[i])
                + 2*self.beta * (self.Q_guess[i+1]**2 / self.A_guess[i+1] - self.Q_guess[i]**2 / self.A_guess[i])
                + self.delta_x*self.beta*g*W**(4./3)*n**2 * (self.Q_guess[i+1]**2 / self.A_guess[i+1]**(7./3) + self.Q_guess[i]**2 / self.A_guess[i]**(7./3))
                - (
                    self.celerity * (self.Q_previous[i+1] + self.Q_previous[i])
                    - 2*(1-self.beta) * (self.Q_previous[i+1]**2 / self.A_previous[i+1] + (0.5*g/W)*self.A_previous[i+1]**2
                                       - self.Q_previous[ i ]**2 / self.A_previous[ i ] - (0.5*g/W)*self.A_previous[ i ]**2)
                    + self.delta_x*(1-self.beta)*g * (self.A_previous[i+1] * (S_0 - self.Sf_previous[i+1])
                                                    + self.A_previous[ i ] * (S_0 - self.Sf_previous[ i ]))
                  )
            )
            
        
        equation_list.append(
            self.Q_guess[-1] - (1/n) * self.A_guess[-1] ** (5./3)
            * self.Sf_ds ** 0.5 / (W + 2*self.A_guess[-1]/W) ** (2./3)
        )
        
        return Matrix(equation_list)
        
    
    def build_jacobian(self):
        unknowns = [val for pair in zip(self.A_guess, self.Q_guess)
                    for val in pair]           
        unknowns = Matrix(unknowns)
        
        self.symbolic_jacobian = self.system.jacobian(unknowns)
        
        for i in range(self.symbolic_jacobian.shape[0]):
            for j in range(self.symbolic_jacobian.shape[1]):
                if self.symbolic_jacobian[i, j] != 0:
                    self.jacobian_sparse_indices.append((i, j))
    
    
    def sparse_jacobian(self):
        jacobian_sparse = lil_matrix((2*self.n_nodes, 2*self.n_nodes))

        for i, j in self.jacobian_sparse_indices:
            jacobian_sparse[i, j] = self.symbolic_jacobian[i, j].subs(self.iteration_subs)

        return csr_matrix(jacobian_sparse)


    def optimized_solve(self, tolerance = 1e-4):
        for time in range(self.delta_t, self.duration+1, self.delta_t):
            print('\n---------- Time = ' + str(time) + 's ----------\n')
            
            self.timestep_subs[self.Q_in] = self.river.inflow_Q(time/3600.)
            Fx = self.system.subs(self.timestep_subs)
            
            iteration = 0
            cumulative_error = tolerance
            
            #while iteration < 6:
            while cumulative_error >= tolerance:
                iteration += 1
                print("--- Iteration #" + str(iteration) + ':')
                
                self.update_guesses()
                
                j_sparse = self.sparse_jacobian()
                
                f = Fx.subs(self.iteration_subs)
                f = np.array(f).astype(float).flatten()
                
                delta = spsolve(j_sparse, -f)
                self.x0 = self.x0 + np.reshape(delta, (60, 1))
                
                delta = [abs(i) for i in delta]
                cumulative_error = np.sum(np.array(delta))
                
                print("Error = " + str(cumulative_error))
        
            self.results.append(self.x0.tolist())
            self.update_parameters()
        
        
    def solve(self, tolerance = 1e-4):
        for time in range(self.delta_t, self.duration + 1, self.delta_t):
            print('\n---------- Time = ' + str(time) + 's ----------\n')
            
            self.timestep_subs[self.Q_in] = self.river.inflow_Q(time/3600.)
            Fx = self.system.subs(self.timestep_subs)
                    
            iteration = 0
            cumulative_error = tolerance
            
            while cumulative_error >= tolerance:
                iteration += 1
                print("--- Iteration " + str(iteration) + ':')
                
                self.update_guesses()
                
                j = self.symbolic_jacobian.subs(self.iteration_subs)
                f = Fx.subs(self.iteration_subs)
                
                j_inverse = j.inv()
                
                delta = -j_inverse * f
                self.x0 = self.x0 + delta
                
                delta = [abs(i) for i in delta]
                cumulative_error = np.sum(np.array(delta))
                
                print("Error = " + str(cumulative_error))
                
            self.results.append(self.x0.tolist())
            self.update_parameters()
        
        
    def save_results(self):
        A, Q = [], []
        for x in self.results:
            Ax, Qx = [], []
            for i in range(0, len(x), 2):
                Ax.append(x[ i ][0])
                Qx.append(x[i+1][0])
                
            A.append(Ax)
            Q.append(Qx)
                
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
            
    
    def initialize_t0(self):
        self.river.initialize_(self.delta_x)
        x0 = []
        
        for x, (A, Q) in enumerate(self.river.initial_conditions):
            x0 += [[A],
                   [Q]]
            
            self.timestep_subs[self.A_previous[x]] = A
            self.timestep_subs[self.Q_previous[x]] = Q
            self.timestep_subs[self.Sf_previous[x]] = self.river.friction_slope(A, Q)
            
        self.results.append(x0)
        self.x0 = Matrix(x0)
        
        
    def update_parameters(self):
        for x in range(self.n_nodes):
            A, Q = self.x0[2*x], self.x0[2*x+1]
            self.timestep_subs[self.A_previous[x]] = A
            self.timestep_subs[self.Q_previous[x]] = Q
            self.timestep_subs[self.Sf_previous[x]] = self.river.friction_slope(A, Q)
                    
        
    def update_guesses(self):
        for x in range(self.n_nodes):
            A, Q = self.x0[2*x], self.x0[2*x+1]
            self.iteration_subs[self.A_guess[x]] = A            
            self.iteration_subs[self.Q_guess[x]] = Q
                    
        self.iteration_subs[self.Sf_ds] = self.river.friction_slope(self.iteration_subs[self.A_guess[-1]], self.iteration_subs[self.Q_guess[-1]])
            