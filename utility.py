import os
from numpy import sum, abs, square


class Utility:
    def __init__():
        pass
    
    
    def create_directory_if_not_exists(directory):
        """
        Checks if a directory exists and creates it if it doesn't.

        Attributes
        ----------
        directory : str
            The path to the directory to check.
        """
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            
        
    def manhattan_norm(vector):
        return sum(abs(vector))
    
    
    def euclidean_norm(vector):
        return sum(square(vector)) ** 0.5