import numpy as np
from src.hydromodel.rating_curve import RatingCurve
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.optimize import brentq
from .settings import OPEN_TIMING, CLOSE_TIMING, JAMMED_SPILLWAYS, JAMMED_SLUICEGATES

HYDROPOWER_Q = 63.0 * 1e6 /(24*3600)
NUM_SLUICE_GATES = 5
NUM_SPILLWAYS = 7
MAX_SPILLWAY_OPENING = 13
MIN_STAGE = 466.7
MAX_STAGE = 492
TAIL_WATER_LEVEL_RANGE = [440, 455]
OPEN_STATUS = [MAX_SPILLWAY_OPENING] * (NUM_SPILLWAYS - JAMMED_SPILLWAYS) + [0] * JAMMED_SPILLWAYS, NUM_SLUICE_GATES - JAMMED_SLUICEGATES
MAX_COOLDOWN = 3600 * 5
    
class RoseiresRatingCurve(RatingCurve):
    def __init__(self, initial_stage = None, initial_flow = None):       
        self.fit_models()
        
        self.initial_stage = initial_stage
        
        if self.initial_stage > MAX_STAGE or self.initial_stage < MIN_STAGE:
            raise ValueError(f"Roseires water stage must be between {MIN_STAGE} m and {MAX_STAGE} m.")
        
        self.current_stage = initial_stage
        self.open = True        
        self.closed_status = None
        
        self.cooldown = 0
        self.prev_time = None
        
        self.spillway_opening = [MAX_SPILLWAY_OPENING] * NUM_SPILLWAYS
        self.open_sluices = NUM_SLUICE_GATES
        
        self.tail_water_level = TAIL_WATER_LEVEL_RANGE[0]
        
        self.calc_closed_status(initial_flow)
        self.close_gates()
 
    def discharge(self, stage, time = None, update_stage = True, check_gate_state = True) -> float:
        if check_gate_state:
            self.gate_control(time=time)
        
        sluice_releases = self.sluice_Q(stage) * self.open_sluices
        spillway_releases = sum([self.spillway_Q(stage, opening) if opening > 0 else 0 for opening in self.spillway_opening])
        
        discharge = spillway_releases + sluice_releases + HYDROPOWER_Q
        
        if update_stage:
            self.current_stage = stage
            
        return discharge

    def spillway_Q(self, stage: float, opening: float = None) -> float:
        opening = MAX_SPILLWAY_OPENING if opening is None else opening
        
        if self.spillway_model is None:
            raise RuntimeError("Model not fitted.")
        return float(self.spillway_model.predict([[stage, opening]])[0])
    
    def sluice_Q(self, stage) -> float:
        """Computes the discharge through one sluice gate for a given stage.

        Args:
            stage (float): Current water stage

        Returns:
            float: Discharge through one sluice gate.
        """
        if self.sluice_model is None:
            raise RuntimeError("Model not fitted.")
        return float(self.sluice_model.predict([[stage, self.tail_water_level]])[0])

    def dQ_dz(self, stage, time = float, dx=0.1) -> float:
        f_plus = self.discharge(stage + dx, time=time, update_stage=False, check_gate_state=False)
        f_minus = self.discharge(stage - dx, time=time, update_stage=False, check_gate_state=False)
        derivative = (f_plus - f_minus) / (2 * dx)
        
        self.current_stage = stage
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
            
    def gate_control(self, time):
        if self.prev_time is not None:
            self.cooldown = max(0, self.cooldown - (time - self.prev_time))
            
        self.prev_time = time
        if self.cooldown > 0:
            return
        
        if self.open_condition(time) and not self.open:
            self.cooldown = MAX_COOLDOWN
            self.open_gates()
        elif self.close_condition(time) and self.open:
            self.cooldown = MAX_COOLDOWN
            self.close_gates()
            
    def open_condition(self, time):
        return self.current_stage >= self.initial_stage + 0.5
        return time < OPEN_TIMING or time > CLOSE_TIMING
    
    def close_condition(self, time):
        return self.current_stage <= self.initial_stage - 1

    def open_gates(self):
        self.spillway_opening, self.open_sluices = OPEN_STATUS
        self.open = True

    def close_gates(self):    
        self.spillway_opening, self.open_sluices = self.closed_status
        self.open = False

    def calc_closed_status(self, initial_flow):
        current_status = self.spillway_opening, self.open_sluices
        
        # First, determine how many sluices need to be open
        self.spillway_opening = [MAX_SPILLWAY_OPENING] * (NUM_SPILLWAYS - JAMMED_SPILLWAYS)
        for i in range(NUM_SLUICE_GATES + 1 - JAMMED_SLUICEGATES):
            self.open_sluices = i
            if self.discharge(stage=self.initial_stage, check_gate_state=False) > initial_flow:
                self.open_sluices = max(i-1, 0)
                break
        
        # Second, determine how many spillways need to be fully open
        fully_o = 0
        for i in range(NUM_SPILLWAYS + 1 - JAMMED_SPILLWAYS):
            self.spillway_opening = [MAX_SPILLWAY_OPENING] * i + [0] * (NUM_SPILLWAYS - i)
            if self.discharge(stage=self.initial_stage, check_gate_state=False) > initial_flow:
                fully_o = i - 1
                break
                
        def f(partial_opening):
            self.spillway_opening = [MAX_SPILLWAY_OPENING] * fully_o + [partial_opening] + [0] * (NUM_SPILLWAYS - fully_o - 1)
            return initial_flow - self.discharge(stage=self.initial_stage, check_gate_state=False)
        
        partial_opening = round(brentq(f, 0, MAX_SPILLWAY_OPENING), 1)
        partially_o = 1 if partial_opening > 0 else 0
        
        if fully_o + partially_o > NUM_SPILLWAYS - JAMMED_SPILLWAYS:
            raise ValueError("wttf")
        
        self.spillway_opening = [MAX_SPILLWAY_OPENING] * fully_o + [partial_opening] + [0] * (NUM_SPILLWAYS - fully_o - 1)
        
        # ------------ #
        
        self.closed_status = self.spillway_opening, self.open_sluices
        self.spillway_opening, self.open_sluices = current_status
    