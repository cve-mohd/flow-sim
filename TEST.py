import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([1, 4, 3], dtype=float)

a = np.vstack((a, b))
c=a.flatten()

a[0,1] = 6
print(a)
print(c)
