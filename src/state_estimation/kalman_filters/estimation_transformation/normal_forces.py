import numpy as np

from src.state_estimation.config.vehicle_params import VehicleParams

A_front = VehicleParams.A_front
c_lift = VehicleParams.c_lift
rho_air = VehicleParams.rho_air
m = VehicleParams.m_car
g = VehicleParams.g
lf = VehicleParams.lf
lr = VehicleParams.lr
l = VehicleParams.l
a = VehicleParams.a
b = VehicleParams.b
z_cg = VehicleParams.z_cg


def estimate_normal_force(state: np.ndarray, wheel_id: int) -> float:
    vx, ax, ay = state[0], state[2], state[3]
    F_lift = (A_front * c_lift * rho_air * (vx ** 2)) / 8.0
    x_mass_trans_cog = m * ax * z_cg / (2 * l)
    y_mass_trans_cog = m * ay * z_cg
    fz_cog = m * g / (2 * l)

    if wheel_id == 0:
        Fz = lr * fz_cog + y_mass_trans_cog / (2 * a) - x_mass_trans_cog
    elif wheel_id == 1:
        Fz = lr * fz_cog - y_mass_trans_cog / (2 * a) - x_mass_trans_cog
    elif wheel_id == 2:
        Fz = lf * fz_cog + y_mass_trans_cog / (2 * b) + x_mass_trans_cog
    elif wheel_id == 3:
        Fz = lf * fz_cog - y_mass_trans_cog / (2 * b) + x_mass_trans_cog
    else:
        raise ValueError("wheel_id must be in [0, 1, 2, 3]")
    return Fz + F_lift


def estimate_normal_forces(state: np.ndarray) -> np.ndarray:
    vx, ax, ay = state[0], state[2], state[3]
    F_lift = (A_front * c_lift * rho_air * (vx ** 2)) / 8.0

    x_mass_trans_cog = m * ax * z_cg / (2 * l)
    y_mass_trans_cog = m * ay * z_cg
    fz_cog = m * g / (2 * l)

    Fz_FL = lr * fz_cog + y_mass_trans_cog / (2 * a) - x_mass_trans_cog
    Fz_FR = lr * fz_cog - y_mass_trans_cog / (2 * a) - x_mass_trans_cog
    Fz_RL = lf * fz_cog + y_mass_trans_cog / (2 * b) + x_mass_trans_cog
    Fz_RR = lf * fz_cog - y_mass_trans_cog / (2 * b) + x_mass_trans_cog
    return np.array([Fz_FL, Fz_FR, Fz_RL, Fz_RR]) + F_lift
