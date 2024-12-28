from sympy import Matrix
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


class NewtonRaphsonSolver:
    def __init__(self):
        self.f = None
        self.unknowns = None
        self.initial_guesses = None
        
        self.substitutions = None
        
        self.jacobian_sparse_indices = []
        self.symbolic_jacobian = None
        self.jacobian_sparse = lil_matrix((len(self.unknowns), len(self.unknowns)))
    
    
    def solve(self,
              system: Matrix,
              unknowns: list,
              initial_guesses: list,
              tolerance = 1e-4,
              reconstruct_jacobian = True):
                
        self.substitutions = list(zip(unknowns, initial_guesses))
        
        if self.symbolic_jacobian is None or reconstruct_jacobian:
            self.symbolic_jacobian = self.build_jacobian()

        iteration = 0
        cumulative_error = tolerance
        x0 = Matrix(self.initial_guesses)
        
        while cumulative_error >= tolerance:
            
            iteration += 1
            print("--- Iteration #" + str(iteration) + ':')
            
            if iteration > 1:
                self.update_guesses(x0)
            
            j_sparse = self.subs_sparse_jacobian()
            
            f = system.subs(dict(self.substitutions))
                
            f = np.array(f).astype(float).flatten()
            
            delta = spsolve(j_sparse, -f)
            x0 = x0 + np.reshape(delta, (-1, 1))
            
            cumulative_error = np.sum(np.array([abs(i) for i in delta]))
            
        return x0
            
            
    def update_guesses(self, x0):
        for i in range(len(self.substitutions)):
            self.substitutions[i] = (self.substitutions[i][0], x0[i])
                    
        
    def build_jacobian(self, system: Matrix, unknowns: list):
        symbolic_jacobian = system.jacobian(unknowns)
        
        for i in range(symbolic_jacobian.shape[0]):
            for j in range(symbolic_jacobian.shape[1]):
                if symbolic_jacobian[i, j] != 0:
                    self.jacobian_sparse_indices.append((i, j))
                    
        return symbolic_jacobian
    
    
    def subs_sparse_jacobian(self):
        for i, j in self.jacobian_sparse_indices:
            self.jacobian_sparse[i, j] = self.symbolic_jacobian[i, j].subs(dict(self.substitutions))

        return csr_matrix(self.jacobian_sparse)
    
