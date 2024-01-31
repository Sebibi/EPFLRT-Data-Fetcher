import pandas as pd

from src.frontend.plotting.plotting import plot_data
import streamlit as st


def plot_new_features(new_data: pd.DataFrame, tab_name: str, wheel_acc_cols: list, long_tire_force_name: list,
                      long_tire_force_est_name_est: list, normal_force_name: list, wheel_speeds_cols_m_s: list,
                      wheel_speeds_cols_m_s_est: list, vl_cols: list, v_adiff: list):
    # Plot the data
    with st.expander("Steering deltas"):
        plot_data(
            new_data, tab_name + "_steering", title='Steering wheel and tires',
            default_columns=['delta_FL_deg', 'delta_FR_deg', 'sensors_steering_angle'],
        )
    with st.expander("Wheel accelerations"):
        plot_data(
            new_data, tab_name + "_acceleration", title='Accelerations observation',
            default_columns=['sensors_accX'] + wheel_acc_cols[2:],
        )

    with st.expander("Longitudinal tire forces"):
        plot_data(
            new_data, tab_name + "_long_tire_forces", title='Longitudinal Tire Force',
            default_columns=long_tire_force_name,
        )
    with st.expander("Estimated longitudinal tire forces"):
        plot_data(
            new_data, tab_name + "_long_tire_forces_est", title='Longitudinal Tire Force Estimation',
            default_columns=long_tire_force_est_name_est,
        )

    with st.expander("Normal forces"):
        plot_data(
            new_data, tab_name + "_normal_force", title='Normal Force',
            default_columns=normal_force_name + ['zero', 'sensors_steering_angle', 'sensors_gyroZ_deg'],
        )

    with st.expander("Estimated Wheel speeds"):
        plot_data(
            new_data, tab_name + "_wheel_speeds", title='Wheel speeds',
            default_columns=wheel_speeds_cols_m_s + wheel_speeds_cols_m_s_est,
        )
    with st.expander("Longitudinal velocities"):
        plot_data(
            new_data, tab_name + "_long_velocities", title='Longitudinal velocities',
            default_columns=[vl_cols[i] for i in [0, 2, 1, 3]] + ['zero', 'sensors_gyroZ'],
        )

    with st.expander("Velocities from acc difference"):
        plot_data(
            new_data, tab_name + "_v_adiff", title='Velocities from acceleration difference',
            default_columns=[v_adiff[0], 'sensors_vXEst', v_adiff[1], 'sensors_vYEst']
        )


