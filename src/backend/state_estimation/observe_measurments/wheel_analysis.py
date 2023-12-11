import pandas as pd
import streamlit as st

from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.frontend.plotting.plotting import plot_data


def plot_wheel_analysis(new_data: pd.DataFrame, tab_name: str, wheel_acc_cols: list, long_tire_force_cols: list,
                        long_tire_force_est_cols_est: list, normal_force_cols: list, wheel_speeds_cols_m_s: list,
                        wheel_speeds_cols_m_s_est: list, vl_cols: list, slip_cols_100: list, slip_cols_1000: list):
    wheel_id = st.selectbox("Select the wheel to plot", [0, 1, 2, 3], key=f"{tab_name} wheel id",
                            format_func=lambda wheel_id: VehicleParams.wheel_names[wheel_id])

    # Wheel speeds analysis
    accel_col = wheel_acc_cols[wheel_id]
    wheel_speed_col = wheel_speeds_cols_m_s[wheel_id]
    wheel_speed_est_col = wheel_speeds_cols_m_s_est[wheel_id]
    vl_col = vl_cols[wheel_id]
    long_tire_force_col = long_tire_force_cols[wheel_id]
    long_tire_force_est_col = long_tire_force_est_cols_est[wheel_id]
    normal_force_col = normal_force_cols[wheel_id]
    slip_col100 = slip_cols_100[wheel_id]
    slip_col1000 = slip_cols_1000[wheel_id]

    # Plot wheel acceleration and speed
    with st.expander("Wheel acceleration and speed"):
        plot_data(
            new_data, tab_name + "_wheel_acceleration_and_speed", title='Wheel acceleration',
            default_columns=[accel_col, wheel_speed_col],
        )

    # Plot wheel speeds
    with st.expander("Wheel speed"):
        plot_data(
            new_data, tab_name + "_wheel_speed", title='Wheel speed',
            default_columns=[wheel_speed_col, wheel_speed_est_col, vl_col],
        )

    # Plot longitudinal tire force
    with st.expander("Longitudinal tire force"):
        plot_data(
            new_data, tab_name + "_long_tire_force", title='Longitudinal tire force',
            default_columns=[long_tire_force_est_col, long_tire_force_col, slip_col100, slip_col1000],
        )

    # PLot raw wheel speeds
    with st.expander("All raw Wheel speeds"):
        plot_data(
            new_data, tab_name + "_raw_wheel_speeds", title='Measures Wheel Speeds',
            default_columns=wheel_speeds_cols_m_s,
        )
