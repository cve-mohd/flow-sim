import numpy as np


discharge = [14034, 14438, 14830, 15220, 15598, 15971, 16342, 16703, 17062, 17409, 17763, 18110, 18450, 18785]
elevation = [480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493]

elevation = [i - 466.7 for i in elevation]

coefficients = np.polynomial.polynomial.Polynomial.fit(x=elevation, y=discharge, deg=2, domain=[])

print(coefficients)

print([coefficients(i) for i in elevation])