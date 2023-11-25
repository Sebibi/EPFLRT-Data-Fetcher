from typing import List
from collections import deque
import numpy as np
from src.state_estimation.config.vehicle_params import VehicleParams


class MeasurementPreprocessing:
    delta: int  # delta time to compute the accelerations in iterations
    wheel_acc_history: deque

    def __init__(self):
        self.delta = 20
        self.wheel_acc_history = deque(maxlen=self.delta)
        self.wheel_acc_history.extend([np.zeros(4) for _ in range(self.delta)])


    def get_delta_wheel(self, steering_angle: float) -> np.ndarray:
        delta_wheels = np.zeros(2)  # Delta L - Delta R
        res0 = 0.165 * steering_angle - 9.5e-4 * steering_angle ** 2
        res1 = 0.207 * steering_angle + 1.02e-4 * steering_angle ** 2
        res0 *= np.pi / 180.0
        res1 *= np.pi / 180.0
        delta_wheels[0] = res0 if steering_angle > 0 else res1
        delta_wheels[1] = res0 if steering_angle < 0 else res1
        return delta_wheels


    def get_wheel_speeds(self, motor_speeds: np.ndarray) -> np.ndarray:
        return motor_speeds * np.pi / (30.0 * VehicleParams.gear_ratio)

    def get_wheel_acc(self, wheel_speeds: np.ndarray) -> np.ndarray:
        old_wheel_speeds = self.wheel_acc_history[0]
        self.wheel_acc_history.append(wheel_speeds)
        return (wheel_speeds - old_wheel_speeds) / (self.delta * 0.01)
    
    
    def get_longitudonal_forces(self, torques: np.ndarray, bps: np.ndarray, wheel_speeds: np.ndarray, wheel_acc: np.ndarray) -> np.ndarray:
        l_forces = np.zeros(4)
        for i, (tau, Pb, w, dw) in enumerate(zip(torques, bps, bps, wheel_speeds, wheel_acc)):
            if abs(w) > 0.1 or abs(dw) > 0.1:
                l_forces[i] = (tau - VehicleParams.kd * w - VehicleParams.ks - VehicleParams.lw * dw) / VehicleParams.Rw
        return l_forces