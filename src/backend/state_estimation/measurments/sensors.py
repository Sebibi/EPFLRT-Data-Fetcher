from typing import TypedDict, Sequence

import numpy as np
import pandas as pd

from src.backend.state_estimation.config.vehicle_params import VehicleParams


class Sensors(TypedDict):
    ins: np.ndarray  # ax
    bps: np.ndarray
    torques: np.ndarray
    motor_speeds: np.ndarray
    steering_angle: float


def get_sensors_from_data(data: pd.DataFrame) -> list[Sensors]:
    sensors_list: list[Sensors] = [
        Sensors(
            ins=np.array([row['sensors_accX'], row['sensors_accY'], row['sensors_accZ'], row['sensors_gyroX'], row['sensors_gyroY'], row['sensors_gyroZ']], dtype=float),
            bps=np.array([row['sensors_brake_pressure_L'] for _ in range(4)], dtype=float),
            torques=np.array([row[f'VSI_TrqFeedback_{wheel}'] for wheel in VehicleParams.wheel_names], dtype=float),
            motor_speeds=np.array([row[f'VSI_Motor_Speed_{wheel}'] for wheel in VehicleParams.wheel_names], dtype=float),
            steering_angle=row['sensors_steering_angle'],
        )
        for _, row in data.iterrows()]
    return sensors_list
