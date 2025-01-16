discharge = [0   , 1562.5, 3850, 6000, 10000, 14000, 21000]  # in m³/s
elevation = [500 , 502.5 , 505 , 507 , 510  , 512  , 515]         # in m

import numpy as np

coefficients = np.polynomial.polynomial.Polynomial.fit(x=elevation, y=discharge, deg=2, domain=[])

print(coefficients)