import numpy as np

from src.state_estimation.config.vehicle_params import VehicleParams


def measure_tire_longitudinal_force(torque: float, bp: float, wheel_speed: float, wheel_acc: float) -> float:
    if abs(wheel_speed) > 0.1 or abs(wheel_acc) > 0.1:
        return (torque - VehicleParams.kd * wheel_speed - VehicleParams.ks - VehicleParams.Iw * wheel_acc) / VehicleParams.Rw
    else:
        return 0.0


def measure_tire_longitudinal_forces(torques: np.ndarray, bps: np.ndarray, wheel_speeds: np.ndarray,
                                     wheel_acc: np.ndarray) -> np.ndarray:
    l_forces = np.zeros(4)
    for i, (tau, Pb, w, dw) in enumerate(zip(torques, bps, wheel_speeds, wheel_acc)):
        if abs(w) > 0.1 or abs(dw) > 0.1:
            l_forces[i] = (tau - VehicleParams.kd * w - VehicleParams.ks - VehicleParams.Iw * dw) / VehicleParams.Rw
    return l_forces



