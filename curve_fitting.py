import numpy as np


discharge = [0   , 1562.5, 3850, 6000, 10000, 14000, 21000]
elevation = [500 , 502.5 , 505 , 507 , 510  , 512  , 515]

elevation = [i - 500 for i in elevation]

coefficients = np.polynomial.polynomial.Polynomial.fit(x=elevation, y=discharge, deg=2, domain=[])

print(coefficients)

print([coefficients(i) for i in elevation])