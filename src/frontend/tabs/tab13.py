import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from stqdm import stqdm

from src.backend.state_estimation.config.state_estimation_param import SE_param
from src.backend.state_estimation.kalman_filters.estimation_transformation import estimate_normal_forces
from src.backend.state_estimation.kalman_filters.estimation_transformation.normal_forces import \
    estimate_aero_focre_one_tire
from src.backend.state_estimation.measurments.sensors import Sensors, get_sensors_from_data
from src.backend.state_estimation.state_estimator_app import StateEstimatorApp
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
    motor_torque_pos_lim = [f'MISC_Pos_Trq_Limit_{wheel}' for wheel in VehicleParams.wheel_names]
    motor_torque_neg_lim = [f'MISC_Neg_Trq_Limit_{wheel}' for wheel in VehicleParams.wheel_names]
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

        self.torque_pos_cols = [f'VSI_TrqPos_{wheel}' for wheel in VehicleParams.wheel_names]
        self.torque_neg_cols = [f'VSI_TrqNeg_{wheel}' for wheel in VehicleParams.wheel_names]

        self.sampling_time = 0.01

    def create_new_feature(self):
        data = self.memory['data'].copy()

        # Create steering wheel angle and steering rad
        data[self.steering_col + "_rad"] = data[self.steering_col].map(np.deg2rad)
        data[self.delta_wheel_angle_cols] = data[[self.steering_col]].apply(
            lambda x: measure_delta_wheel_angle(x[0]), axis=1, result_type='expand')

        # Create slip10 slip100 and slip1000
        data[self.slip_cols10] = data[self.slip_cols].copy() * 10
        data[self.slip_cols100] = data[self.slip_cols].copy() * 100
        data[self.slip_cols1000] = data[self.slip_cols].copy() * 1000

        # BPF 100
        max_bpf = 35
        data['sensors_BPF_100'] = data['sensors_BPF'] * 100 / max_bpf
        data['sensors_BPF_Torque'] = data['sensors_BPF'] * 597 / max_bpf

        # Motor Torque Cmd mean
        data['sensors_Torque_cmd_mean'] = data['sensors_Torque_cmd'].copy() / 4

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

        # Create Fsum and Fsum_est
        data['Fdrag'] = data[SE_param.estimated_states_names].apply(
            lambda x: estimate_aero_focre_one_tire(x), axis=1)

        data['Fsum'] = data[self.longitudinal_forces_cols].sum(axis=1) - data['Fdrag']
        data['Fsum_est'] = data[self.longitudinal_forces_est_cols].sum(axis=1) - data['Fdrag']
        data['Fsum_accx'] = data['sensors_accX'].map(lambda x: x * VehicleParams.m_car)
        data['Fsum_accxEst'] = data['sensors_aXEst'].apply(lambda x: x * VehicleParams.m_car)

        # Compute Torque command
        data[self.torque_pos_cols] = data[self.motor_torque_pos_lim].apply(lambda x: x / 0.773, axis=1)
        data[self.torque_neg_cols] = data[self.motor_torque_neg_lim].apply(lambda x: x / 0.773, axis=1)

        # Filter 0 values from RTK data and interpolate
        rtk_columns = [col for col in data.columns if 'RTK' in col]
        data[rtk_columns] = data[rtk_columns].replace(0, np.nan)
        data[rtk_columns] = data[rtk_columns].interpolate(method='linear', axis=0)

        # Compute the v norm from RTK data
        data['sensors_RTK_v_norm'] = np.sqrt(data['sensors_RTK_vx'].values ** 2 + data['sensors_RTK_vy'].values ** 2)

        data['sensors_pitch_rate_integ_deg'] = data['sensors_gyroY'].cumsum() * self.sampling_time * 180 / np.pi
        data['sensors_pitch_rate_deg'] = data['sensors_gyroY']* 180 / np.pi


        self.memory['data'] = data.copy()

    def compute_state_estimator(self):

        data = self.memory['data']
        samples = (data.index[0], data.index[-1])

        sensors_list: list[Sensors] = get_sensors_from_data(data.loc[samples[0]:samples[1]])
        estimator_app = StateEstimatorApp(independent_updates=False)
        estimations = [np.zeros(SE_param.dim_x) for _ in sensors_list]
        estimations_cov = [np.zeros(SE_param.dim_x) for _ in sensors_list]
        for i, sensors in stqdm(enumerate(sensors_list), total=len(sensors_list)):
            state, cov = estimator_app.run(sensors)
            estimations[i] = state
            estimations_cov[i] = cov

        # Update the data
        columns = SE_param.estimated_states_names
        data.loc[samples[0]: samples[1], columns] = np.array(estimations)
        self.memory['data'] = data.copy()

        # Update the data_cov
        index = data.loc[samples[0]: samples[1]].index
        data_cov = pd.DataFrame(estimations_cov, index=index, columns=columns)
        self.memory['data_cov'] = data_cov.copy()
        st.balloons()

        self.create_new_feature()

    def build(self, session_creator) -> bool:

        st.header(self.description)
        datetime_range = session_creator.r2d_session_selector(
            st.session_state.sessions,
            key=f"{self.name} session selector",
            session_info=True
        )

        cols = st.columns(6)
        if cols[0].button("Fetch this session", key=f"{self.name} fetch data button"):
            data = session_creator.fetch_data(datetime_range, verify_ssl=st.session_state.verify_ssl)
            data.index = (data.index - data.index[0]).total_seconds().round(2)
            self.memory['data'] = data
            self.create_new_feature()

        if len(self.memory['data']) > 0:
            cols[1].success("Data fetched")

        if cols[2].button("Compute State Estimation", key=f"{self.name} compute state estimation"):
            with cols[2].status("Computing state estimation..."):
                self.compute_state_estimator()

        if 'data_cov' in self.memory:
            cols[3].success("Computed")

        if cols[4].button("Create new features", key=f"{self.name} create new features"):
            with cols[4].status("Creating new features"):
                self.create_new_feature()

        if self.normal_forces_cols[0] in self.memory['data'].columns:
            cols[5].success("Created")

        st.divider()

        if len(self.memory['data']) > 0:
            data = self.memory['data']

            with st.expander("Filtering"):

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
                    cols = st.columns(3, gap='large')

                    apps_diff = data['sensors_APPS_Travel'].diff()
                    if cols[0].toggle("Filter APPS rising edge", key=f"{self.name} filter APPS rising edge"):
                        # Find and APPS rising edge
                        apps_rising_edge = apps_diff.gt(0)
                        apps_rising_edge = apps_rising_edge[apps_rising_edge].index
                        if len(apps_rising_edge) > 0:
                            rising_edge_time = cols[0].selectbox("Rising edge time",apps_rising_edge, key=f"{self.name} rising edge number")
                            data = data.loc[rising_edge_time:]
                        else:
                            st.warning("No rising edge found")

                    if cols[1].toggle("Filter APPS falling edge", key=f"{self.name} filter APPS falling edge"):
                        # Find and APPS falling edge
                        apps_falling_edge = apps_diff.lt(0)
                        apps_falling_edge = apps_falling_edge[apps_falling_edge].index
                        if len(apps_falling_edge) > 0:
                            falling_edge_time = cols[1].selectbox("Falling edge time", apps_falling_edge, key=f"{self.name} falling edge number")
                            data = data.loc[:falling_edge_time]
                        else:
                            st.warning("No falling edge found")

                    time_from_start = cols[2].number_input("Time from start [ms]", value=200,
                                                           key=f"{self.name} time from start")
                    if cols[2].toggle("Filter with time from start"):
                        data = data.iloc[:time_from_start]




            with st.container(border=True):
                mode_int = data[self.knob_mode].iloc[0]
                elapsed_time = data.index[-1] - data.index[0] + self.sampling_time
                arg_max_accx = data['sensors_accX'].rolling(10).mean().idxmax()
                max_accx = data['sensors_accX'].rolling(10).mean().max()
                arg_max_vx = data['sensors_vXEst'].rolling(10).mean().idxmax()
                max_vx = data['sensors_vXEst'].rolling(10).mean().max()
                mean_accx = data['sensors_accX'].mean()

                # Compute distance from velocity
                data['distance'] = data['sensors_vXEst'].cumsum() * self.sampling_time
                distance = data['distance'].iloc[-1]

                # Show metrics
                cols = st.columns([2, 2, 3, 3, 3, 3])
                cols[0].metric("Mode", VehicleParams.ControlMode[mode_int])
                cols[1].metric("Time", f"{elapsed_time:.2f} s")
                cols[2].metric("Mean AccX", f"{mean_accx:.2f} m/s²")
                cols[3].metric("Max AccX", f"{max_accx:.2f} m/s²", f"At {arg_max_accx:.2f} s", delta_color="off")
                cols[4].metric("Max VX", f"{max_vx:.2f} m/s", f"At {arg_max_vx:.2f} s", delta_color="off")
                cols[5].metric("Distance", f"{distance:.2f} m")

            st.divider()


            st.subheader("Session Overview")
            cols = st.columns(3)
            with cols[0]:
                driver_inputs_cols = ['sensors_APPS_Travel', 'sensors_BPF', 'sensors_steering_angle']
                plot_data(data=data, tab_name=self.name + "DI", title="Driver Inputs", default_columns=driver_inputs_cols, simple_plot=True)
            with cols[1]:
                car_outputs_cols = self.motor_torques_cols
                plot_data(data=data, tab_name=self.name + "CO", title="Car Outputs", default_columns=car_outputs_cols, simple_plot=True)
            with cols[2]:
                sensors_cols = ['sensors_accX', 'sensors_accY'] + self.wheel_speeds_cols + ['sensors_RTK_v_norm']
                plot_data(data=data, tab_name=self.name + "S", title="Sensors", default_columns=sensors_cols, simple_plot=True)
            st.divider()


            # PLot acceleration and speed
            with st.expander("Acceleration and Speed"):
                data['v_accX_integrated'] = data['sensors_accX'].cumsum() * self.sampling_time
                plot_data(data=data, tab_name=self.name + "AS", title="Overview",
                          default_columns=['sensors_accX', 'sensors_accY'] + self.acc_cols + self.speed_cols + ['v_accX_integrated'])

            # Plot wheel speeds
            with st.expander("Wheel Speeds"):
                if st.toggle("Show wheel speeds", key=f"{self.name} show wheel speeds"):
                    plot_data(data=data, tab_name=self.name + "WS", title="Wheel Speeds",
                          default_columns=self.wheel_speeds_cols + self.speed_cols[:1] + ['v_accX_integrated'])

            # Plot the wheel slip
            with st.expander("Wheel Slip"):
                if st.toggle("Show wheel slip", key=f"{self.name} show wheel slip"):
                    plot_data(data=data, tab_name=self.name + "Slip", title="Slip Ratios", default_columns=self.slip_cols)

            # Sanity check: plot the wheel speeds estimation
            with st.expander("Wheel Speeds Estimation subplots"):
                if st.toggle("Show wheel speeds estimation", key=f"{self.name} show wheel speeds estimation"):
                    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
                    for i, wheel in enumerate(VehicleParams.wheel_names):
                        cols = [self.wheel_speeds_cols[i], self.wheel_speeds_est_cols[i], self.vl_cols[i]]
                        data[cols].plot(ax=ax[i // 2, i % 2], title=f"Wheel {wheel} speed")
                    plt.tight_layout()
                    st.pyplot(fig)

            # Plot longitudinal force
            with st.expander("Wheel Speeds Estimation"):
                if st.toggle("Show longitudinal forces", key=f"{self.name} show wheel speed estimation"):
                    wheel = st.selectbox("Wheel", VehicleParams.wheel_names + ['all'], key=f"{self.name} wheel selection long force")
                    cols = self.wheel_speeds_cols + self.wheel_speeds_est_cols + self.vl_cols
                    if wheel != 'all':
                        cols = [col for col in cols if wheel in col]
                    plot_data(data=data, tab_name=self.name + "LF", title="Longitudinal Forces",
                              default_columns=cols)

            # Plot the normal forces
            with st.expander("Normal Forces"):
                if st.toggle("Show normal forces", key=f"{self.name} show normal forces"):
                    plot_data(data=data, tab_name=self.name + "NF", title="Normal Forces",
                              default_columns=self.normal_forces_cols)

            # PLot wheel accelerations
            with st.expander("Wheel Accelerations"):
                if st.toggle("Show wheel accelerations", key=f"{self.name} show wheel accelerations"):
                    plot_data(data=data, tab_name=self.name + "WA", title="Wheel Accelerations",
                              default_columns=self.wheel_acceleration_cols)

            # Plot the longitudinal forces
            with st.expander("Longitudinal Forces subplots"):
                if st.toggle("Show longitudinal forces", key=f"{self.name} show longitudinal forces"):
                    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
                    for i, wheel in enumerate(VehicleParams.wheel_names):
                        cols = [self.longitudinal_forces_cols[i], self.longitudinal_forces_est_cols[i],
                                self.slip_cols1000[i]]
                        data[cols].plot(ax=ax[i // 2, i % 2], title=f"Wheel {wheel} longitudinal force")
                    plt.tight_layout()
                    st.pyplot(fig)

            # Plot longitudinal force
            with st.expander("Longitudinal Forces"):
                if st.toggle("Show longitudinal forces", key=f"{self.name} show longitudinal force"):
                    wheel = st.selectbox("Wheel", VehicleParams.wheel_names + ['all'], key=f"{self.name} wheel selection long force")
                    cols = self.longitudinal_forces_cols + self.longitudinal_forces_est_cols + self.slip_cols1000
                    if wheel != 'all':
                        cols = [col for col in cols if wheel in col]
                    plot_data(data=data, tab_name=self.name + "LF", title="Longitudinal Forces",
                              default_columns=cols)

            # Plot wheel torques
            with st.expander("Wheel MIN/MAX Torques"):
                if st.toggle("Show wheel torques", key=f"{self.name} show wheel min/max torques"):
                    add_slips = st.checkbox("Add slip ratios", key=f"{self.name} add slips")
                    window_size = st.number_input("Moving average window size", value=1, key=f"{self.name} window size")
                    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

                    for i, wheel in enumerate(VehicleParams.wheel_names):
                        cols = [self.motor_torques_cols[i], self.max_motor_torques_cols[i], self.min_motor_torques_cols[i]]
                        if add_slips:
                            cols += [self.slip_cols1000[i]]
                        data[cols].rolling(window_size).mean().plot(ax=ax[i // 2, i % 2], title=f"Wheel {wheel} torques")
                    plt.tight_layout()
                    st.pyplot(fig)

            with st.expander("Wheel torques"):
                if st.toggle("Show wheel torques", key=f"{self.name} show wheel torques"):
                    plot_data(data=data, tab_name=self.name + "WT", title="Wheel Torques",
                              default_columns=self.motor_torques_cols)

            with st.expander("Fsum"):
                if st.toggle("Show Fsum", key=f"{self.name} show Fsum"):
                    plot_data(data=data, tab_name=self.name + "Fsum", title="Fsum",
                              default_columns=['Fsum', 'Fsum_est', 'Fsum_accx', 'Fsum_accxEst'])
