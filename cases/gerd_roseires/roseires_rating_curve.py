import numpy as np
from hydromodel.utility import RatingCurve
from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
    
class RoseiresRatingCurve(RatingCurve):
    HYDROPOWER_Q = 63.0 * 1e6 /(24*3600)
    NUM_SLUICE_GATES = 5
    NUM_SPILLWAYS = 7
    
    def __init__(self):       
        self.fit_models()
        
        self.spillway_opening = [13] * RoseiresRatingCurve.NUM_SPILLWAYS # max is 13 m
        self.open_sluices = 5
        self.tail_water_level = 455 # Usually ranges between 440 and 455 m.
 
    def discharge(self, stage, time = None) -> float:
        self.gate_state(time=time)
        
        sluice_releases = self.sluice_Q(stage) * self.open_sluices
        spillway_releases = sum([self.spillway_Q(stage, opening) if opening > 0 else 0 for opening in self.spillway_opening])
        
        discharge = spillway_releases + sluice_releases + RoseiresRatingCurve.HYDROPOWER_Q
        return discharge

    def spillway_Q(self, stage: float, opening: float = 13) -> float:
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
        f_plus = self.discharge(stage + dx, time=time)
        f_minus = self.discharge(stage - dx, time=time)
        derivative = (f_plus - f_minus) / (2 * dx)
        return derivative

    def fit_models(self):
            # Spillway:
            df = read_csv("cases\\gerd_roseires\\input\\roseires_spillway_releases.csv", index_col=0)
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
            df = read_csv("cases\\gerd_roseires\\input\\roseires_deep_sluice_releases.csv", index_col=0)
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
            
    def gate_state(self, time):
        OPEN_AT = 6
        CLOSE_AT = 55
        t = time/3600.
        
        if t < OPEN_AT or t > CLOSE_AT:
            self.spillway_opening = [6.5] + [0] * (RoseiresRatingCurve.NUM_SPILLWAYS-1)
            self.open_sluices = 0
            
        else:
            self.spillway_opening = [13] * RoseiresRatingCurve.NUM_SPILLWAYS
            self.open_sluices = 5
