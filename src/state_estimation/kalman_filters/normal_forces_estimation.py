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

def get_normal_forces(state: np.ndarray):
    vx, ax, ay = state[0], state[2], state[3]

    F_lift = (A_front * c_lift * rho_air * (vx**2)) / 8.0
    Fz_FL = m * g * lr / (2 * l) + m * ay * z_cg / (2 * a) - m * ax * z_cg / (2 * l)
    Fz_FR = m * g * lr / (2 * l) - m * ay * z_cg / (2 * a) - m * ax * z_cg / (2 * l)
    Fz_RL = m * g * lr / (2 * l) + m * ay * z_cg / (2 * b) - m * ax * z_cg / (2 * l)
    Fz_RR = m * g * lr / (2 * l) - m * ay * z_cg / (2 * b) - m * ax * z_cg / (2 * l)

    return np.array([Fz_FL, Fz_FR, Fz_RL, Fz_RR]) + F_lift



