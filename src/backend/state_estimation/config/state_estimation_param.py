import numpy as np
import streamlit as st
from src.backend.state_estimation.config.vehicle_params import VehicleParams


class SE_param:
    estimated_states_names = (["sensors_vXEst", "sensors_vYEst", "sensors_aXEst", "sensors_aYEst", "sensors_dpsi_est"] +
                              [f"sensors_s_{wheel}_est" for wheel in VehicleParams.wheel_names])

    dt = 0.01
    dim_x = 9

    # LKF
    ins_measurement_noise = np.diag([0.001, 0.001, 0.0001])
    vy_reset_noise = np.array([[0.1]])

    # EKF
    state_transition_noise = np.diag([0.00001, 0.0001, 0.0001, 0.0001, 0.0001] + [0.002 for _ in range(4)])

    # UKF
    alpha, beta, kappa = (0.001, 2., 0.)  # Sigma points parameter
    # alpha, beta, kappa = (0.1, 2., -1)  # Sigma points parameter

    wheel_speed_measurement_noise = np.diag([1.0 for _ in range(4)])
    longitudinal_force_measurement_noise = np.diag([1.0 for _ in range(4)])

    @classmethod
    def set_ins_measurement_noise(cls, ax: float, ay: float, yaw_rate: float):
        cls.ins_measurement_noise = np.diag([ax, ay, yaw_rate])

    @classmethod
    def set_vy_reset_noise(cls, vy_reset_noise: float):
        cls.vy_reset_noise = np.array([[vy_reset_noise]])

    @classmethod
    def set_state_transition_noise(cls, vx: float, vy: float, ax: float, ay: float, yaw_rate: float, slip: float):
        cls.state_transition_noise = np.diag([vx, vy, ax, ay, yaw_rate, slip, slip, slip, slip])

    @classmethod
    def set_sigma_points_param(cls, alpha: float, beta: float, kappa: float):
        cls.alpha = alpha
        cls.beta = beta
        cls.kappa = kappa

    @classmethod
    def set_wheel_speed_measurement_noise(cls, wheel_speed: float):
        cls.wheel_speed_measurement_noise = np.diag([wheel_speed for _ in range(4)])

    @classmethod
    def set_longitudinal_force_measurement_noise(cls, longitudinal_force: float):
        cls.longitudinal_force_measurement_noise = np.diag([longitudinal_force for _ in range(4)])


def tune_param_input(tab_name: str):
    # Tune the state estimation parameters
    cols_ref = [1, 3]

    st.markdown("## Tune the state estimation parameters")
    cols = st.columns(cols_ref)
    cols[0].markdown("### LKF")
    sub_cols = cols[1].columns(3)
    values = SE_param.ins_measurement_noise.diagonal()
    ax = sub_cols[0].number_input("ax", value=values[0], key=f"{tab_name} ax_lkf", format="%0.8f")
    ay = sub_cols[1].number_input("ay", value=values[1], key=f"{tab_name} ay_lkf", format="%0.8f")
    yaw_rate = sub_cols[2].number_input("yaw rate", value=values[2], key=f"{tab_name} yaw rate_lkf", format="%0.8f")
    SE_param.set_ins_measurement_noise(ax, ay, yaw_rate)
    st.divider()

    cols = st.columns(cols_ref)
    cols[0].markdown("### LKF vy reset")
    values = SE_param.vy_reset_noise.diagonal()
    vy_reset = cols[1].number_input("vy reset", value=values[0], key=f"{tab_name} vy reset", format="%0.8f")
    SE_param.set_vy_reset_noise(vy_reset_noise=vy_reset)
    st.divider()

    cols = st.columns(cols_ref)
    cols[0].markdown("### EKF")
    sub_cols = cols[1].columns(2)
    values = SE_param.state_transition_noise.diagonal()
    vx = sub_cols[0].number_input("vx", value=values[0], key=f"{tab_name} vx_ekf", format="%0.8f")
    vy = sub_cols[1].number_input("vy", value=values[1], key=f"{tab_name} vy_ekf", format="%0.8f")
    ax = sub_cols[0].number_input("ax", value=values[2], key=f"{tab_name} ax", format="%0.8f")
    ay = sub_cols[1].number_input("ay", value=values[3], key=f"{tab_name} ay", format="%0.8f")
    w = sub_cols[0].number_input("yaw rate", value=values[4], key=f"{tab_name} yaw rate_ekf", format="%0.8f")
    slip = sub_cols[1].number_input("slip ratio", value=values[5], key=f"{tab_name} slip", format="%0.8f")
    SE_param.set_state_transition_noise(vx, vy, ax, ay, w, slip)
    st.divider()

    cols = st.columns(cols_ref)
    cols[0].markdown("### UKF")
    sub_cols = cols[1].columns(3)
    values = [SE_param.alpha, SE_param.beta, SE_param.kappa]
    alpha = sub_cols[0].number_input("alpha", value=values[0], key=f"{tab_name} alpha", format="%0.8f")
    beta = sub_cols[1].number_input("beta", value=values[1], key=f"{tab_name} beta", format="%0.8f")
    kappa = sub_cols[2].number_input("kappa", value=values[2], key=f"{tab_name} kappa", format="%0.8f")
    SE_param.set_sigma_points_param(alpha, beta, kappa)

    sub_cols = cols[1].columns(2)
    values = [SE_param.wheel_speed_measurement_noise[0, 0], SE_param.longitudinal_force_measurement_noise[0, 0]]
    w = sub_cols[0].number_input("wheel speed", value=values[0], key=f"{tab_name} wheel speed", format="%0.8f")
    lf = sub_cols[1].number_input("longitudinal force", value=values[1], key=f"{tab_name} longitudinal force",
                                  format="%0.8f")
    SE_param.set_wheel_speed_measurement_noise(wheel_speed=w)
    SE_param.set_longitudinal_force_measurement_noise(longitudinal_force=lf)
    st.divider()

    cols = st.columns(cols_ref)
    cols[0].markdown("### Mu max")
    sub_cols = cols[1].columns(1)
    values = [VehicleParams.D]
    mu_max = sub_cols[0].number_input("mu max", value=values[0], key=f"{tab_name} mu max", format="%0.3f")
    VehicleParams.set_mu_max(mu_max)


