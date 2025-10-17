import numpy as np

class Hydrograph:
    def __init__(self, function = None, table: np.ndarray = None):
        """Creates a hydrograph object.

        Args:
            function (_type_, optional): f(t). Defaults to None.
            table (np.ndarray, optional): Should contain time (in seconds) in the first column
            and corresponding values in the second. Defaults to None.
        """
        self.table = table
        self.used_function = self.interpolate_hydrograph if function is None else function
            
    def interpolate_hydrograph(self, time):
        if self.table is None:
            raise ValueError("Hydrograph is not defined.")
                        
        return float(np.interp(time, self.table[:, 0], self.table[:, 1]))

    def get_at(self, time):
        return self.used_function(time)
        
    def set_table(self, table: np.ndarray):
        """Time (in seconds) in the first column and corresponding values in the second.

        Args:
            table (np.ndarray): NumPy array containing the hydrograph.
        """
        self.table = table
       
    def set_function(self, func):
        self.used_function = func
