import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.kalman_filters.estimation_transformation import estimate_normal_forces
from src.frontend.plotting.plotting import plot_data
from src.frontend.tabs import Tab
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.measurments.measurement_transformation.wheel_speed import measure_wheel_speeds
from src.backend.state_estimation.measurments.measurement_transformation.steering_to_wheel_angle import \
    measure_delta_wheel_angle
from src.backend.state_estimation.kalman_filters.estimation_transformation.wheel_speed import estimate_wheel_speeds
from src.backend.state_estimation.kalman_filters.estimation_transformation.longitudonal_speed import \
    estimate_longitudinal_velocities
from src.backend.state_estimation.measurments.measurement_transformation.longitudonal_tire_force import \
    measure_tire_longitudinal_forces
from src.backend.state_estimation.kalman_filters.estimation_transformation.longitudinal_tire_force import \
    estimate_longitudinal_tire_forces
from src.backend.state_estimation.measurments.measurement_transformation.wheel_acceleration import \
    measure_wheel_acceleration


class Tab13(Tab):
    acc_cols = ['sensors_aXEst', 'sensors_aYEst']
    speed_cols = ['sensors_vXEst', 'sensors_vYEst']
    motor_torques_cols = [f'VSI_TrqFeedback_{wheel}' for wheel in VehicleParams.wheel_names]
    max_motor_torques_cols = [f'sensors_TC_Tmax_{wheel}' for wheel in VehicleParams.wheel_names]
    min_motor_torques_cols = [f'sensors_TC_Tmin_{wheel}' for wheel in VehicleParams.wheel_names]
    motor_speeds_cols = [f'VSI_Motor_Speed_{wheel}' for wheel in VehicleParams.wheel_names]
    slip_cols = [f'sensors_s_{wheel}_est' for wheel in VehicleParams.wheel_names]

    steering_col = 'sensors_steering_angle'
    knob_mode = 'sensors_Knob3_Mode'

    def __init__(self):
        super().__init__(name="tab13", description="Acceleration Analysis")
        if "data" not in self.memory:
            self.memory['data'] = pd.DataFrame()

        self.wheel_speeds_cols = [f'vWheel_{wheel}' for wheel in VehicleParams.wheel_names]
        self.wheel_speeds_est_cols = [f'vWheel_{wheel}_est' for wheel in VehicleParams.wheel_names]
        self.wheel_acceleration_cols = [f'accWheel_{wheel}' for wheel in VehicleParams.wheel_names]
        self.delta_wheel_angle_cols = [f'delta_wheel_{wheel}' for wheel in VehicleParams.wheel_names]
        self.vl_cols = [f'vL_{wheel}_est' for wheel in VehicleParams.wheel_names]
        self.normal_forces_cols = [f'Fz_{wheel}_est' for wheel in VehicleParams.wheel_names]
        self.longitudinal_forces_cols = [f'Fl_{wheel}' for wheel in VehicleParams.wheel_names]
        self.longitudinal_forces_est_cols = [f'Fl_{wheel}_est' for wheel in VehicleParams.wheel_names]
        self.brake_pressure_cols = ['sensors_brake_pressure_L' for _ in range(4)]

        self.slip_cols10 = [f'sensors_s_{wheel}_est_10' for wheel in VehicleParams.wheel_names]
        self.slip_cols100 = [f'sensors_s_{wheel}_est_100' for wheel in VehicleParams.wheel_names]
        self.slip_cols1000 = [f'sensors_s_{wheel}_est_1000' for wheel in VehicleParams.wheel_names]

    def build(self, session_creator) -> bool:

        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(st.session_state.sessions,
                                                              key=f"{self.name} session selector")
        if st.button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds()

            with st.spinner("Creating new features"):

                # Create steering wheel angle and steering rad
                data[self.steering_col + "_rad"] = data[self.steering_col].map(np.deg2rad)
                data[self.delta_wheel_angle_cols] = data[[self.steering_col]].apply(
                    lambda x: measure_delta_wheel_angle(x[0]), axis=1, result_type='expand')

                # Create slip10 slip100 and slip1000
                data[self.slip_cols10] = data[self.slip_cols].copy() * 10
                data[self.slip_cols100] = data[self.slip_cols].copy() * 100
                data[self.slip_cols1000] = data[self.slip_cols].copy() * 1000

                # Create wheel speeds and longitudinal velocity
                data[self.wheel_speeds_cols] = data[self.motor_speeds_cols].apply(
                    lambda x: measure_wheel_speeds(x) * VehicleParams.Rw, axis=1, result_type='expand')
                data[self.wheel_speeds_est_cols] = data[
                    SE_param.estimated_states_names + self.delta_wheel_angle_cols].apply(
                    lambda x: estimate_wheel_speeds(x[:9], x[9:]) * VehicleParams.Rw, axis=1, result_type='expand'
                )
                data[self.vl_cols] = data[SE_param.estimated_states_names + self.delta_wheel_angle_cols].apply(
                    lambda x: estimate_longitudinal_velocities(x[:9], x[9:]), axis=1, result_type='expand'
                )

                # Create wheel acceleration and Reset wheel acceleration
                for i in range(30):
                    measure_wheel_acceleration(wheel_speeds=np.array([0, 0, 0, 0], dtype=float))
                data[self.wheel_acceleration_cols] = np.array([
                    measure_wheel_acceleration(wheel_speeds=wheel_speeds)
                    for wheel_speeds in data[self.wheel_speeds_cols].values
                ])

                # Create Normal Forces
                data[self.normal_forces_cols] = data[SE_param.estimated_states_names].apply(estimate_normal_forces,
                                                                                            axis=1,
                                                                                            result_type='expand')

                # Create Longitudinal Forces
                data[self.longitudinal_forces_cols] = data[
                    self.motor_torques_cols + self.brake_pressure_cols + self.wheel_speeds_cols + self.wheel_acceleration_cols].apply(
                    lambda x: measure_tire_longitudinal_forces(x[:4], x[4:8], x[8:12], x[12:]), axis=1,
                    result_type='expand'
                )
                data[self.longitudinal_forces_est_cols] = data[SE_param.estimated_states_names].apply(
                    lambda x: estimate_longitudinal_tire_forces(x, use_traction_ellipse=False), axis=1,
                    result_type='expand'
                )

                self.memory['data'] = data

        if len(self.memory['data']) > 0:
            data = self.memory['data']

            with st.container(border=True):
                cols = st.columns(4)
                threshold = cols[1].number_input("Threshold", value=60, key=f"{self.name} APPS Threshold")
                left_shift = cols[2].number_input("Left Shift", value=20, key=f"{self.name} left shift")
                right_shift = cols[3].number_input("Right Shift", value=0, key=f"{self.name} right shift")
                if cols[0].toggle("Filter APPS > Threshold", key=f"{self.name} filter APPS"):
                    index = data['sensors_APPS_Travel'] > threshold
                    augmented_index = index | index.shift(right_shift) | index.shift(-left_shift)
                    data = data[augmented_index]

            with st.container(border=True):
                mode_int = data[self.knob_mode].iloc[0]
                ellapsed_time = data.index[-1] - data.index[0]
                arg_max_accx = data['sensors_aXEst'].rolling(10).mean().idxmax()
                max_accx = data['sensors_aXEst'].rolling(10).mean().max()
                arg_max_vx = data['sensors_vXEst'].rolling(10).mean().idxmax()
                max_vx = data['sensors_vXEst'].rolling(10).mean().max()

                cols = st.columns([2, 2, 3, 3])

                # Show metrics
                cols[0].metric("Mode", VehicleParams.ControlMode[mode_int])
                cols[1].metric("Time", f"{ellapsed_time:.2f} s")
                cols[2].metric("Max AccX", f"{max_accx:.2f} m/s²", f"At {arg_max_accx:.2f} s", delta_color="off")
                cols[3].metric("Max VX", f"{max_vx:.2f} m/s", f"At {arg_max_vx:.2f} s", delta_color="off")

            # PLot acceleration and speed
            with st.expander("Acceleration and Speed"):
                data['v_acc_integrated'] = data['sensors_aXEst'].cumsum() * 0.01
                plot_data(data=data, tab_name=self.name + "AS", title="Overview",
                          default_columns=self.acc_cols + self.speed_cols + ['v_acc_integrated'])

            # Plot wheel speeds
            with st.expander("Wheel Speeds"):
                plot_data(data=data, tab_name=self.name + "WS", title="Wheel Speeds",
                          default_columns=self.wheel_speeds_cols + self.speed_cols[:1] + ['v_acc_integrated'])

            # Plot the wheel slip
            with st.expander("Wheel Slip"):
                plot_data(data=data, tab_name=self.name + "Slip", title="Slip Ratios", default_columns=self.slip_cols)

            # Sanity check: plot the wheel speeds estimation
            with st.expander("Wheel Speeds Estimation"):
                fig, ax = plt.subplots(2, 2, figsize=(15, 10))
                for i, wheel in enumerate(VehicleParams.wheel_names):
                    cols = [self.wheel_speeds_cols[i], self.wheel_speeds_est_cols[i], self.vl_cols[i]]
                    data[cols].plot(ax=ax[i // 2, i % 2], title=f"Wheel {wheel} speed")
                plt.tight_layout()
                st.pyplot(fig)

            # Plot the normal forces
            with st.expander("Normal Forces"):
                plot_data(data=data, tab_name=self.name + "NF", title="Normal Forces",
                          default_columns=self.normal_forces_cols)

            # PLot wheel accelerations
            with st.expander("Wheel Accelerations"):
                plot_data(data=data, tab_name=self.name + "WA", title="Wheel Accelerations",
                          default_columns=self.wheel_acceleration_cols)

            # Plot the longitudinal forces
            with st.expander("Longitudinal Forces"):
                fig, ax = plt.subplots(2, 2, figsize=(15, 10))
                for i, wheel in enumerate(VehicleParams.wheel_names):
                    cols = [self.longitudinal_forces_cols[i], self.longitudinal_forces_est_cols[i],
                            self.slip_cols1000[i]]
                    data[cols].plot(ax=ax[i // 2, i % 2], title=f"Wheel {wheel} longitudinal force")
                plt.tight_layout()
                st.pyplot(fig)

            # Plot wheel torques
            with st.expander("Wheel MIN/MAX Torques"):
                add_slips = st.checkbox("Add slip ratios", key=f"{self.name} add slips")
                fig, ax = plt.subplots(2, 2, figsize=(15, 10))
                for i, wheel in enumerate(VehicleParams.wheel_names):
                    cols = [self.motor_torques_cols[i], self.max_motor_torques_cols[i], self.min_motor_torques_cols[i]]
                    if add_slips:
                        cols += [self.slip_cols1000[i]]
                    data[cols].plot(ax=ax[i // 2, i % 2], title=f"Wheel {wheel} torques")
                plt.tight_layout()
                st.pyplot(fig)

            with st.expander("Wheel torques"):
                plot_data(data=data, tab_name=self.name + "WT", title="Wheel Torques",
                          default_columns=self.motor_torques_cols)