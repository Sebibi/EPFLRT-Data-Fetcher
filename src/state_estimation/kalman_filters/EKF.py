import numpy as np


class EKF:
    dt = 0.01
    dim_x = 9
    dim_z = 3
    state_cov = [0.1, 0.1, 0.1, 0.1, 0.01] + [0.1 for _ in range(4)]

    def __init__(self):
        B = np.eye(self.dim_x)
        self.Q = np.diag(self.state_cov)
        self.N = B @ self.Q @ B.T * self.dt

    def predict(self, x: np.array, P: np.ndarray, vx, vy, yaw_rate):
        A = np.zeros((self.dim_x, self.dim_x))
        A[0, 1] = yaw_rate
        A[0, 2] = 1
        A[0, 4] = vy
        A[1, 0] = -yaw_rate
        A[1, 3] = 1
        A[1, 4] = -vx
        phi = A * self.dt
        print(phi[:5, :5])
        x += phi @ x
        P = phi @ P @ phi.T + self.Q
        return x, P


if __name__ == '__main__':
    ekf = EKF()
    x = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=float).reshape(-1, 1)
    P = np.eye(ekf.dim_x)
    vx = 1
    vy = 1
    yaw_rate = 0
    print(x)
    x, P = ekf.predict(x, P, vx, vy, yaw_rate)
    print(x)
    print(P[:4, :4])
    x, P = ekf.predict(x, P, vx, vy, yaw_rate)
    print(x)
