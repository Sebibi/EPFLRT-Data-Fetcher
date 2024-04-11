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


def traction_ellipse(x: np.array) -> np.ndarray:
    # 1 = (ax / ax_max)^2 + (ay / ay_max)^2
    # 1 = (mux / mux_max)^2 + (ay * (Fz/g) / muy_max * Fz)^2
    # mux = mux_max * sqrt(1 - (ay / muy_max * g)^2)
    mux_max = VehicleParams.D
    muy_max = VehicleParams.D - 0.3
    mux = mux_max * np.sqrt(1 - min((x[3] / (muy_max * VehicleParams.g)) ** 2, 0.95))
    return mux


if __name__ == '__main__':
    slips = np.linspace(-1.5, 1.5, 100)

    forces = []
    for s in slips:
        x = np.array([0, 0, 0, 0, 0, s, 0, 0, 0])
        f = estimate_longitudinal_tire_force(x, 0)
        forces.append(f)
        print(x)
        print(f)

    # plot the forces
    import matplotlib.pyplot as plt
    plt.plot(slips, [f[0] for f in forces])
    plt.plot(slips, slips * 1000, 'ro')
    plt.show()