from typing import TypedDict, Sequence

import pandas as pd


class Inputs(TypedDict):
    ins: Sequence[float]
    bps: Sequence[float]
    torques: Sequence[float]
    motor_speeds: Sequence[int]
    steering_angle: float


def get_inputs(inputs: pd.DataFrame) -> Inputs:
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
