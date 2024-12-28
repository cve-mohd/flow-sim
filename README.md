# Simplified Solution of Flood Wave Propagation in Open Channel

## Overview
This project numerically solves the Saint-Venant equations using the finite difference method.

## Features
- **Preissmann Implicit Scheme**: Implements the Preissmann scheme for solving the Saint-Venant equations.
- **Newton-Raphson Method**: Used for solving the nonlinear system of equations.
- **Customizable Parameters**: Allows users to define river characteristics, boundary conditions, and initial conditions.

## Project Structure
The program consists of the following components:

1. **`River`**: 
   - Contains river characteristics such as:
     - Width
     - Bed slope
     - Initial and boundary conditions
     - Manning's coefficient
     - Length

2. **`PreissmannModel`**: 
   - Implements the Preissmann implicit scheme.
   - Handles the numerical solution of the Saint-Venant equations using the Newton-Raphson method.

3. **`LaxModel`**: 
   - Provides an alternative solution using the Lax method for comparison and validation.

4. **`settings.py`**:
   - Centralized configuration file for defining:
     - River attributes (e.g., bed slope, Manning's coefficient, width, length)
     - Simulation parameters (e.g., time step, spatial step, duration, tolerance)
     - Initial and boundary conditions (e.g., rating curves, initial depth, discharge, stage)

## Requirements
- **Python** (version 3.8 or later)
- Required Python packages:
  - `numpy`
  - `scipy`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mohammed0r217/1d_flow_simulator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd 1d_flow_simulator
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Define the river and simulation parameters in `settings.py`.
2. Run the program.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
Special thanks to Prof. Mohammed Akode Osman for supervising this project.
