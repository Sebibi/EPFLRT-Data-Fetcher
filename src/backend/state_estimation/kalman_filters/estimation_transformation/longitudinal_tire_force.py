import numpy as np

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.kalman_filters.estimation_transformation.normal_forces import estimate_normal_force, \
    estimate_normal_forces


def estimate_longitudinal_tire_force(x: np.array, wheel_id: int, use_traction_ellipse: bool = True) -> np.ndarray:
    normal_force = estimate_normal_force(x, wheel_id=wheel_id)
    mux = traction_ellipse(x) if use_traction_ellipse else None
    mu = VehicleParams.magic_formula(slip_ratio=x[5 + wheel_id], mux=mux)
    return np.array([normal_force * mu])


def estimate_longitudinal_tire_forces(x: np.array, use_traction_ellipse: bool = True) -> np.ndarray:
    normal_forces = estimate_normal_forces(x)
    mux = traction_ellipse(x) if use_traction_ellipse else None
    mu = np.array([VehicleParams.magic_formula(s, mux=mux) for s in x[5:9]])
    return normal_forces * mu


def traction_ellipse(x: np.array) -> float:
    # 1 = (ax / ax_max)^2 + (ay / ay_max)^2
    # 1 = (mux / mux_max)^2 + (ay * (Fz/g) / muy_max * Fz)^2
    # mux = mux_max * sqrt(1 - (ay / muy_max * g)^2)
    mux_max = VehicleParams.D
    muy_max = VehicleParams.D - 0.3

    mux_max = min(VehicleParams.D, VehicleParams.mu_init + x[0] * (VehicleParams.D - VehicleParams.mu_init) / 5)
    mux = mux_max * np.sqrt(1 - min((x[3] / (muy_max * VehicleParams.g)) ** 2, 0.95))
    return mux


if __name__ == '__main__':
    slips = np.linspace(-1.5, 1.5, 100)

    vx = np.linspace(0, 10, 100)
    states = np.array([np.array([v, 0, 0, 0, 0, 0, 0, 0, 0, 0]) for v in vx])

    mus = [traction_ellipse(x) for x in states]

    # Plot the traction ellipse
    import matplotlib.pyplot as plt
    plt.plot(vx, mus, label="Traction ellipse")
    plt.xlabel("vx [m/s]")
    plt.ylabel("mux")
    plt.legend()
    plt.grid()
    plt.show()


   