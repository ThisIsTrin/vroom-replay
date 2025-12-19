import pandas as pd
import numpy as np


class TelemetryData:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        t = df["Time"].astype(np.float64).values / 1000.0
        lon = df["imu.long"].astype(np.float32).values
        lat = df["imu.lat"].astype(np.float32).values

        valid = ~(np.isnan(lon) | np.isnan(lat))
        self.time = t[valid]
        self.long_g = lon[valid] / 9.8
        self.lat_g = lat[valid] / 9.8

    def clamp_time(self, t: float) -> float:
        return max(self.time[0], min(t, self.time[-1]))
