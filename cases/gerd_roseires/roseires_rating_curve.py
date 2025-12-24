import numpy as np
from src.hydromodel.rating_curve import RatingCurve
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.optimize import brentq
from .settings import OPEN_TIMING, CLOSE_TIMING

HYDROPOWER_Q = 63.0 * 1e6 / (24*3600)
NUM_SLUICE_GATES = 5
NUM_SPILLWAYS = 7
MAX_SPILLWAY_OPENING = 13
MIN_STAGE = 466.7
MAX_STAGE = 492
TAIL_WATER_LEVEL_RANGE = [440, 455]
    
class RoseiresRatingCurve(RatingCurve):
    def __init__(
        self, initial_stage = None,
        initial_flow = None,
        initially_open = False,
        jammed_spillways = 0,
        jammed_sluice_gates = 0,
        max_cooldown = 3600 * 5,
        smooth = True,
        buffer = 0.5,
        deep_sluices_active = True
    ):       
        self.fit_models()
        
        self.initial_stage = initial_stage
        
        if self.initial_stage > MAX_STAGE or self.initial_stage < MIN_STAGE:
            raise ValueError(f"Roseires water stage must be between {MIN_STAGE} m and {MAX_STAGE} m.")
        
        self.current_stage = initial_stage
        self.smooth = smooth
        self.open = True
        self.buffer = buffer
        
        self.jammed_spillways = jammed_spillways
        self.jammed_sluice_gates = jammed_sluice_gates if deep_sluices_active else NUM_SLUICE_GATES
        
        self.open_state = [MAX_SPILLWAY_OPENING] * (NUM_SPILLWAYS - self.jammed_spillways) + [0] * self.jammed_spillways,\
                           NUM_SLUICE_GATES - self.jammed_sluice_gates
        self.closed_state = None
        
        self.spillway_openings = None
        self.open_sluices_num = None
                           
        self.max_cooldown = max_cooldown
        self.cooldown = 0
        self.prev_time = None
        
        self.tail_water_level = np.average(TAIL_WATER_LEVEL_RANGE)
        
        self.calc_closed_state(initial_flow)
        
        if initially_open:
            self.open_gates()
        else:
            self.close_gates()
 
    def discharge(self, stage, time = None, update_stage = True, update_gate_state = True, smooth = None) -> float:
        if smooth is None:
            smooth = self.smooth
            
        if not smooth:
            if update_gate_state:
                self.gate_control(time=time)
            
            discharge = self.total_release(stage)
            
            if update_stage:
                self.current_stage = stage
            
        else:
            alpha = self.alpha_smooth(stage)
            discharge = self.effective_release(stage, alpha)
            
        return discharge

    def total_release(self, stage):
        sluice_releases = self.sluice_Q(stage) * self.open_sluices_num
        spillway_releases = sum([self.spillway_Q(stage, opening) if opening > 0 else 0 for opening in self.spillway_openings])
        return spillway_releases + sluice_releases + HYDROPOWER_Q

    def effective_release(self, stage, alpha):
        self.open_gates()
        high_Q = self.total_release(stage)
            
        self.close_gates()
        low_Q = self.total_release(stage)
            
        return (1.0 - alpha) * low_Q + alpha * high_Q

    def alpha_smooth(self, stage, buffer = None):
        if buffer is None:
            buffer = self.buffer
            
        if stage >= self.initial_stage + buffer:
            alpha = 1.0
        elif stage <= self.initial_stage:
            alpha = 0.0
        else:
            s = (stage - self.initial_stage) / buffer
            alpha = 3 * s**2 - 2 * s**3
        return alpha

    def gate_control(self, time):
        if self.prev_time is not None:
            self.cooldown = max(0, self.cooldown - (time - self.prev_time))
            
        self.prev_time = time
        if self.cooldown > 0:
            return
        
        if self.open_condition(time) and not self.open:
            self.cooldown = self.max_cooldown
            self.open_gates()
        elif self.close_condition(time) and self.open:
            self.cooldown = self.max_cooldown
            self.close_gates()
            
    def open_condition(self, time):
        return self.current_stage >= self.initial_stage + 0.5
    
    def close_condition(self, time):
        return self.current_stage <= self.initial_stage - 1

    def open_gates(self):
        self.set_state(self.open_state)
        self.open = True

    def set_state(self, state):
        self.spillway_openings, self.open_sluices_num = state

    def close_gates(self):
        self.set_state(self.closed_state)
        self.open = False

    def calc_closed_state(self, initial_flow):
        current_status = self.spillway_openings, self.open_sluices_num
        
        # First, determine how many sluices need to be open
        self.spillway_openings = [MAX_SPILLWAY_OPENING] * (NUM_SPILLWAYS - self.jammed_spillways) # All spillways open except jammed ones
        for i in range(NUM_SLUICE_GATES + 1 - self.jammed_sluice_gates):
            self.open_sluices_num = i
            if self.discharge(stage=self.initial_stage, update_gate_state=False, smooth=False) > initial_flow:
                self.open_sluices_num = max(i-1, 0)
                break
        
        # Second, determine how many spillways need to be fully open
        fully_o = 0
        for i in range(NUM_SPILLWAYS + 1 - self.jammed_spillways):
            self.spillway_openings = [MAX_SPILLWAY_OPENING] * i + [0] * (NUM_SPILLWAYS - i)
            if self.discharge(stage=self.initial_stage, update_gate_state=False, smooth=False) > initial_flow:
                fully_o = i - 1
                break
                
        def f(partial_opening):
            self.spillway_openings = [MAX_SPILLWAY_OPENING] * fully_o + [partial_opening] + [0] * (NUM_SPILLWAYS - fully_o - 1)
            return initial_flow - self.discharge(stage=self.initial_stage, update_gate_state=False, smooth=False)
        
        partial_opening = round(brentq(f, 0, MAX_SPILLWAY_OPENING), 1)
        partially_o = 1 if partial_opening > 0 else 0
        
        if fully_o + partially_o > NUM_SPILLWAYS - self.jammed_spillways:
            raise ValueError("wttf")
        
        self.spillway_openings = [MAX_SPILLWAY_OPENING] * fully_o + [partial_opening] + [0] * (NUM_SPILLWAYS - fully_o - 1)
        
        # ------------ #
        
        self.closed_state = self.spillway_openings, self.open_sluices_num
        self.spillway_openings, self.open_sluices_num = current_status

    def spillway_Q(self, stage: float, opening: float = None) -> float:
        opening = MAX_SPILLWAY_OPENING if opening is None else opening
        
        if self.spillway_model is None:
            raise RuntimeError("Model not fitted.")
        return float(self.spillway_model.predict([[stage, opening]])[0])
    
    def sluice_Q(self, stage, tail_water_level = None) -> float:
        """Computes the discharge through one sluice gate for a given stage.

        Args:
            stage (float): Current water stage

        Returns:
            float: Discharge through one sluice gate.
        """
        if self.sluice_model is None:
            raise RuntimeError("Model not fitted.")
        
        twl = self.tail_water_level if tail_water_level is None else tail_water_level
        return float(self.sluice_model.predict([[stage, twl]])[0])

    def dQ_dz(self, stage, time = float, dY=0.001) -> float:
        f_plus = self.discharge(stage + dY, time=time, update_stage=False, update_gate_state=False)
        f_minus = self.discharge(stage - dY, time=time, update_stage=False, update_gate_state=False)
        derivative = (f_plus - f_minus) / (2 * dY)
        
        #self.current_stage = stage
        return derivative

    def fit_models(self):
            # Spillway:
            df = read_csv("cases\\gerd_roseires\\data\\roseires_spillway_releases.csv", index_col=0)
            df.index.name = "stage"

            stages = df.index.to_numpy(dtype=float)
            openings = df.columns.to_numpy(dtype=float)

            X, y = [], []
            for i, stage in enumerate(stages):
                for j, opening in enumerate(openings):
                    discharge = df.iloc[i, j]
                    if not np.isnan(discharge):
                        X.append([stage, opening])
                        y.append(discharge)

            X = np.array(X)
            y = np.array(y)

            self.spillway_model = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linreg", LinearRegression())
            ])
            self.spillway_model.fit(X, y)
            
            # Sluice:
            df = read_csv("cases\\gerd_roseires\\data\\roseires_deep_sluice_releases.csv", index_col=0)
            df.index.name = "stage"

            stages = df.index.to_numpy(dtype=float)
            tail_water_levels = df.columns.to_numpy(dtype=float)

            X, y = [], []
            for i, stage in enumerate(stages):
                for j, twl in enumerate(tail_water_levels):
                    discharge = df.iloc[i, j]
                    if not np.isnan(discharge):
                        X.append([stage, twl])
                        y.append(discharge)

            X = np.array(X)
            y = np.array(y)

            self.sluice_model = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("linreg", LinearRegression())
            ])
            self.sluice_model.fit(X, y)

if __name__ == '__main__':
    rc = RoseiresRatingCurve(initial_stage=486, initial_flow=1562.5, deep_sluices_active=False)
    print(rc.discharge(stage=487))
"""

Y = [y for y in range(480, 493)]
rc = RoseiresRatingCurve(initial_stage=487, initial_flow=1562.5, initially_open=False)

Q = [rc.discharge(stage=y, update_stage=False, update_gate_state=False) for y in Y]

import csv
with open("low_release_rating_curve.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Y", "Q"])  # header
    for y_val, q_val in zip(Y, Q):
        writer.writerow([y_val, q_val])
        
# High release
for j1 in [0, 1]:
    for j2 in [0, 1]:
        rc = RoseiresRatingCurve(initial_stage=487, initial_flow=1562.5, initially_open=True,
                                 jammed_spillways=j1, jammed_sluice_gates=j2)
        
        Q = [rc.discharge(stage=y, update_stage=False, update_gate_state=False) for y in Y]

        with open(f"high_release_rating_curve_{j1}{j2}.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Y", "Q"])  # header
            for y_val, q_val in zip(Y, Q):
                writer.writerow([y_val, q_val])
"""

# py -m cases.gerd_roseires.roseires_rating_curve
