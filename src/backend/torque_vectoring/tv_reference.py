import numpy as np

from src.backend.state_estimation.measurments.measurement_transformation import measure_delta_wheel_angle
from src.backend.torque_vectoring.config_tv import TVParams
def tv_reference(vx_est: float, steering_wheel: float) -> float:
    delta_wheels = measure_delta_wheel_angle(steering_wheel)
    delta_wheel_mean = np.mean(delta_wheels) * 2 # Mean between FL and FR

    l = TVParams.wheel_base
    yaw_ref = vx_est/(l + TVParams.K_understeer*l*vx_est**2) * delta_wheel_mean
    return yaw_ref


def tv_references(vx_est: np.ndarray, steering_wheel: np.ndarray) -> np.ndarray:
    return np.array([tv_reference(vx, steering) for vx, steering in zip(vx_est, steering_wheel)])