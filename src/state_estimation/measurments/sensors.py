from typing import TypedDict, Sequence

import numpy as np
import pandas as pd


class Sensors(TypedDict):
    ins: np.ndarray  # ax
    bps: np.ndarray
    torques: np.ndarray
    motor_speeds: np.ndarray
    steering_angle: float


def get_sensors(inputs: pd.DataFrame) -> Sensors:
    INS_index = [0, 1, 2]
    BP_index = [3, 4]
    TorqueFeedBack_index = [5, 6, 7, 8]
    MotorSpeeds_index = [9, 10, 11, 12]
    angle_steering_index = 13
    ins = inputs[INS_index]
    bps = inputs[BP_index]
    torques = inputs[TorqueFeedBack_index]
    motor_speeds = inputs[MotorSpeeds_index]
    steering_angle = inputs[angle_steering_index]
    return dict(
        ins=ins,
        bps=bps,
        torques=torques,
        motor_speeds=motor_speeds,
        steering_angle=steering_angle
    )
