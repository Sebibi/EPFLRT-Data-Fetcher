import numpy as np

from src.state_estimation.kalman_filters.MKF import MKF
from src.state_estimation.measurments.sensors import Sensors

from src.state_estimation.measurments.measurement_transformation.wheel_speed import measure_wheel_speeds
from src.state_estimation.measurments.measurement_transformation.wheel_acceleration import measure_wheel_acceleration
from src.state_estimation.measurments.measurement_transformation.steering_to_wheel_angle import \
    measure_delta_wheel_angle


class StateEstimatorApp:
    mkf: MKF
    x: np.ndarray
    P: np.ndarray

    def __init__(self):
        self.mkf = MKF()
        self.x = np.zeros((self.mkf.dim_x, 1))  # initial state (location and velocity)
        self.P = np.eye(self.mkf.dim_x)  # initial uncertainty

    def run(self, sensors: Sensors) -> tuple[np.ndarray, np.ndarray]:
        wheel_speeds = measure_wheel_speeds(motor_speeds=sensors['motor_speeds'])
        wheel_acc = measure_wheel_acceleration(wheel_speeds=wheel_speeds)
        steering_deltas = measure_delta_wheel_angle(steering_angle=sensors['steering_angle'])

        # Predict step (EKF)
        self.x, self.P = self.mkf.predict(self.x, self.P)

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
        return self.x.copy(), self.P.copy()
