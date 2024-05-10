import numpy as np
import pandas as pd

from src.backend.state_estimation.kalman_filters.MKF import MKF
from src.backend.state_estimation.measurments.sensors import Sensors

from src.backend.state_estimation.measurments.measurement_transformation.wheel_speed import measure_wheel_speeds
from src.backend.state_estimation.measurments.measurement_transformation.wheel_acceleration import \
    measure_wheel_acceleration
from src.backend.state_estimation.measurments.measurement_transformation.steering_to_wheel_angle import \
    measure_delta_wheel_angle

from src.backend.state_estimation.measurments.measurement_transformation.wheel_acceleration import \
    measure_wheel_acceleration


class StateEstimatorApp:
    mkf: MKF
    x: np.ndarray
    P: np.ndarray

    def __init__(self, independent_updates: bool = False):
        self.mkf = MKF(independent_updates=independent_updates)
        self.x = np.zeros(self.mkf.dim_x)  # initial state (location and velocity)
        self.P = np.eye(self.mkf.dim_x) * 1e-6  # initial uncertainty

        # reset wheel_acc:
        for i in range(30):
            measure_wheel_acceleration(wheel_speeds=np.array([0, 0, 0, 0], dtype=float))

    def run(self, sensors: Sensors) -> tuple[np.ndarray, np.ndarray]:
        wheel_speeds = measure_wheel_speeds(motor_speeds=sensors['motor_speeds'])
        wheel_acc = measure_wheel_acceleration(wheel_speeds=wheel_speeds)
        steering_deltas = measure_delta_wheel_angle(steering_angle=sensors['steering_angle'])

        # Predict step (EKF)
        self.x, self.P = self.mkf.predict(self.x, self.P, wheel_speeds, wheel_acc)

        # Update step (LKF, UKF)
        self.x, self.P = self.mkf.update(
            x=self.x,
            P=self.P,
            ins=sensors['ins'],
            wheel_speeds=wheel_speeds,
            wheel_acc=wheel_acc,
            steering_deltas=steering_deltas,
            torques=sensors['torques'],
            bp=sensors['bps'],
        )
        return self.x.copy(), np.diag(self.P)


if __name__ == '__main__':
    sensors = Sensors(
        ins=np.array([0, 0, 0, 0, 0, 0], dtype=float),
        motor_speeds=np.array([0, 0, 0, 0], dtype=float),
        torques=np.array([0, 0, 0, 0], dtype=float),
        bps=np.array([0, 0, 0, 0], dtype=float),
        steering_angle=0,
    )

    app = StateEstimatorApp()
    x, P = app.run(sensors)
    print(x)
