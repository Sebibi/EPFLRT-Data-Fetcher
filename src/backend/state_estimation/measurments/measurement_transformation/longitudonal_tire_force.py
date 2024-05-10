import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams


def measure_tire_longitudinal_force(torque: float, bp: float, wheel_speed: float, wheel_acc: float) -> float:
    if abs(wheel_speed) > 0.1 or abs(wheel_acc) > 0.1:
        return (torque - VehicleParams.kd * wheel_speed - VehicleParams.kb * bp - VehicleParams.Iw * wheel_acc - VehicleParams.ks) / VehicleParams.Rw
    else:
        return 0.0


def measure_tire_longitudinal_forces(torques: np.ndarray, bps: np.ndarray, wheel_speeds: np.ndarray,
                                     wheel_acc: np.ndarray) -> np.ndarray:
    l_forces = np.zeros(4)
    for i, (tau, bp, w, dw) in enumerate(zip(torques, bps, wheel_speeds, wheel_acc)):
        if abs(w) > 0.1 or abs(dw) > 0.1:
            l_forces[i] = (tau - VehicleParams.kd * w - VehicleParams.kb * bp - VehicleParams.Iw * dw - VehicleParams.ks) / VehicleParams.Rw
    return l_forces


def simple_measure_tire_longitudinal_force(torque: float, bp: float) -> float:
    return (torque - VehicleParams.kb * bp) / VehicleParams.Rw






