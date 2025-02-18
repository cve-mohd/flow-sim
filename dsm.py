#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g

# ---------------------
# Given parameters
# ---------------------
n = 0.027                      # Manning's coefficient
b = 250.0                      # channel width (m) for rectangular channel
Q = 1562.5                     # discharge (m^3/s)
S0 = 2.3145/15000                # bed slope
L = 15000.0                    # length of reach (m)
dx = 100.0                     # step size (m)

# ---------------------
# Functions
# ---------------------
def friction_slope(h):
    """
    Compute the friction slope S_f at depth h using Manning's formula.
    For a rectangular channel: A = b*h, P = b + 2*h, and R = A/P.
    """
    A = b * h
    P = b + 2 * h
    R = A / P
    return (Q * n / (A * (R ** (2/3))))**2

def energy(h):
    """
    Compute the specific energy (water depth plus velocity head)
    at a section with water depth h.
    """
    A = b * h
    v = Q / A
    # Note: since channel is rectangular, velocity V = Q/A,
    # so velocity head = V^2/(2g) = Q^2/(2g*A^2)
    return h + v**2 / (2 * g)

def f(h_up, h_down):
    """
    Function whose zero gives the upstream depth h_up.
    h_down is the known downstream depth.
    The energy equation over one step is:
      energy(h_up) + (S0 + S_f(h_up))*dx = energy(h_down)
    Rearranging, we define:
      f(h_up) = energy(h_up) + (S0 + S_f(h_up))*dx - energy(h_down)
    """
    A_up = b * h_up
    term_up = h_up + Q**2 / (2 * g * (A_up**2))
    return term_up + (S0 + friction_slope(h_up)) * dx - (h_down + Q**2 / (2 * g * (b * h_down)**2))

def solve_hup(h_down):
    """
    Solve f(h_up) = 0 for h_up using a Newton iteration.
    """
    tol = 1e-6
    maxit = 100
    h_guess = h_down  # initial guess: assume h_up is similar to h_down
    for i in range(maxit):
        f_val = f(h_guess, h_down)
        # approximate derivative using a finite difference
        dh = 1e-6
        f_prime = (f(h_guess + dh, h_down) - f_val) / dh
        if f_prime == 0:
            break
        h_new = h_guess - f_val / f_prime
        if abs(h_new - h_guess) < tol:
            return h_new
        h_guess = h_new
    return h_guess

# ---------------------
# Main program
# ---------------------

# Ask the user for the downstream water depth (in meters)
h_down_input = 7.5#input("Enter the downstream water depth (m): ")
try:
    h_down = float(h_down_input)
except ValueError:
    print("Please enter a valid number for the water depth.")
    exit()

# Build the backwater profile by marching upstream
nsteps = int(L / dx)
x_vals = [0]         # x = 0 at the downstream end
h_vals = [h_down]    # downstream water depth

# March upstream one step at a time
for i in range(nsteps):
    h_new = solve_hup(h_vals[-1])
    h_vals.append(h_new)
    x_vals.append((i + 1) * dx)

# Convert lists to numpy arrays for plotting
x_vals = np.array(x_vals)
h_vals = np.array(h_vals)

# Plot the backwater profile.
plt.figure(figsize=(8, 5))
plt.plot(x_vals, h_vals, marker='o')
plt.xlabel("Distance upstream (m)")
plt.ylabel("Water depth (m)")
plt.title("Backwater Profile using the Direct Step Method")
# In many backwater profiles, the downstream end is the lower x-value.
# If desired, reverse the x-axis:
plt.gca().invert_xaxis()
plt.grid(True)
plt.show()

print(h_vals[-1])

np.savetxt('output.csv', h_vals, delimiter=',')