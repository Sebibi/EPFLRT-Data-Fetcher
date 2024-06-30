# %%
import numpy as np
from src.backend.state_estimation.config.vehicle_params import VehicleParams
from src.backend.state_estimation.measurments.measurement_transformation.steering_to_wheel_angle import \
    measure_delta_wheel_angle
import streamlit as st

# %%
def moment_coefficients(steering: float):
    delta = measure_delta_wheel_angle(steering)
    lf = VehicleParams.lf
    a2 = VehicleParams.a / 2
    b2 = VehicleParams.b / 2
    return np.array([
        lf * np.sin(delta[0]) + a2 * np.cos(delta[0]), lf * np.sin(delta[1]) - a2 * np.cos(delta[1]), b2, -b2,
    ])


def get_torque_ref(Tcmd, Tmax):

    if abs(Tcmd) > abs(sum(Tmax)):
        return Tmax
    else:
        return Tcmd * Tmax / sum(Tmax)


def get_moment_allocation(Mz, Tmax, Mz_coefficients):
    inv_Tmax = 1 / Tmax
    A_raw = np.array([
        Mz_coefficients,
        [inv_Tmax[0], 0, -inv_Tmax[2], 0],
        [0, inv_Tmax[1], 0, -inv_Tmax[3]],
        [inv_Tmax[0], inv_Tmax[1], 0, 0],
        [0, 0, inv_Tmax[2], inv_Tmax[3]],
        [inv_Tmax[0], 0, 0, inv_Tmax[3]],
        [0, inv_Tmax[1], inv_Tmax[2], 0],
        np.ones(4),
    ], dtype=np.float64)
    B_raw = np.linalg.pinv(A_raw)
    torques = B_raw @ np.array([Mz, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    return torques


def get_residual_torque_allocation(Mz, Tmax, Mz_coefficients):
    if Mz == 0:
        return np.zeros(4)

    inv_Tmax = 1 / Tmax
    if Mz_coefficients[0] == 0 and Mz_coefficients[2] == 0:
        A_raw = np.array([
            Mz_coefficients,
            [0, inv_Tmax[1], 0, -inv_Tmax[3]],
        ], dtype=np.float64)
    elif Mz_coefficients[1] == 0 and Mz_coefficients[3] == 0:
        A_raw = np.array([
            Mz_coefficients,
            [inv_Tmax[0], 0, -inv_Tmax[2], 0],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Not right or left residuals: {Mz_coefficients}")
    B_raw = np.linalg.pinv(A_raw)
    torques = B_raw @ np.array([Mz, 0], dtype=np.float64)
    return torques


def TA_explicit(Tmax, Tcmd, Mz_cmd, steering) -> dict:
    T_refs = get_torque_ref(Tcmd, Tmax)
    Mz_coefficients = moment_coefficients(steering)
    current_Mz = Mz_coefficients @ T_refs
    T_margins = Tmax - T_refs
    print("Torque Maximum", Tmax)
    print("Torque reference", T_refs.astype(int))
    print("Current Moment", int(current_Mz))
    print("Torque Margins", (T_margins).astype(int))

    Mz_error = Mz_cmd - current_Mz
    print("Moment error", int(Mz_error))
    Mz_allocation = get_moment_allocation(Mz_error, Tmax, Mz_coefficients)
    print("Moment allocation", Mz_allocation.astype(int))
    T_residuals = np.maximum(Mz_allocation - T_margins, 0)
    Mz_residuals = Mz_coefficients @ T_residuals
    print("Residuals", T_residuals.astype(int), int(Mz_residuals))
    Mz_residual_coefficients = (T_residuals == 0).astype(int) * Mz_coefficients
    print("Mz_residual_coefficients", Mz_residual_coefficients)
    Mz_residual_allocation = get_moment_allocation(Mz_residuals, Tmax, Mz_residual_coefficients)
    print("Mz_residual_allocation", Mz_residual_allocation.astype(int))
    Mz_residual_allocation *= (T_residuals == 0).astype(int)
    print("Mz_residual_allocation", Mz_residual_allocation.astype(int))
    T_final = T_refs + Mz_allocation + Mz_residual_allocation
    T_final = np.maximum(np.minimum(T_final, Tmax), -Tmax)

    
    print("Final Torque allocation", (T_final + 0.5).astype(int))
    print("Total T: ", Tcmd, "-->", int(sum(T_final) + 0.5))
    print("Total Mz:", Mz_cmd, "-->", int(T_final @ Mz_coefficients + 0.5))
    print("Grip Allocation", np.round(T_final / Tmax, 2))

    return {
        "Delta wheel angle": np.rad2deg(measure_delta_wheel_angle(steering)),
        "Mz_coefficients": Mz_coefficients,
        "Tmax": Tmax,
        "Trefs": T_refs,
        "current_Mz": current_Mz,
        "T_margins": T_margins,
        "Mz_error": Mz_error,
        "Mz_allocation": Mz_allocation,
        "T_residuals": T_residuals,
        "Mz_residuals": Mz_residuals,
        "Mz_residual_coefficients": Mz_residual_coefficients,
        "Mz_residual_allocation": Mz_residual_allocation,
        "T_final": T_final,
        "total_T": sum(T_final),
        "total_Mz": (T_final @ Mz_coefficients) / VehicleParams.Rw,
        "grip_allocation": T_final / Tmax,
    }


if __name__ == '__main__':

    wheels = ['FL', 'FR', 'RL', 'RR']

    st.set_page_config(layout="wide")
    st.title("Torque Allocation")
    st.markdown("This application allows you to allocate the torque to the wheels of the car")

    with st.sidebar:
        st.title("Parameters")
        use_slider = st.checkbox("Use Slider", value=False)

    cols = st.columns(3)
    if use_slider:
        Tcmd = cols[0].slider("Commanded Torque", -1100, 1100, 300)
        Mz_cmd = cols[1].slider("Commanded Moment", -2000, 2000, 0)
        steering = cols[2].slider("Steering Angle", -120, 120, 0)
    else:
        Tcmd = cols[0].number_input("Commanded Torque", -1100, 1100, 300)
        Mz_cmd = cols[1].number_input("Commanded Moment", -2000, 2000, 0)
        steering = cols[2].number_input("Steering Angle", -120, 120, 0)

    cols = st.columns(4)
    Tmax = np.array([0, 0, 0, 0])
    for i in range(4):
        if use_slider:
            Tmax[i] = cols[i].slider(f"{wheels[i]}", 0, 275, 200)
        else:
            Tmax[i] = cols[i].number_input(f"{wheels[i]}", 0, 275, 200)
    st.divider()


    # final_Tmax = -Tmax if Tcmd < 0 else Tmax
    if Tcmd >= 0:
        results_raw = TA_explicit(Tmax, Tcmd, Mz_cmd * VehicleParams.Rw, steering)
        results = {k: (v + 0.5).astype(int).tolist() if isinstance(v, np.ndarray) else int(v + 0.5) for k, v in
                   results_raw.items()}
    else:
        results_raw = TA_explicit(Tmax, -Tcmd, -Mz_cmd * VehicleParams.Rw, -steering)
        results_raw["T_final"] = -results_raw["T_final"]
        results_raw["grip_allocation"] = -results_raw["grip_allocation"]
        results_raw["total_T"] = -results_raw["total_T"]
        results_raw["total_Mz"] = -results_raw["total_Mz"]
        results = {k: (v + 0.5).astype(int).tolist() if isinstance(v, np.ndarray) else int(v + 0.5) for k, v in
                   results_raw.items()}

    results_raw_rounded = {k: np.round(v, 1) for k, v in results_raw.items()}

    cols = st.columns(3)
    delta = np.round(results_raw_rounded['total_T'] - Tcmd, 1)
    color = 'inverse' if Tcmd < 0 else 'inverse'if delta != 0 else 'off'
    st.warning(color)
    cols[0].metric(
        "Total Torque",
        results_raw_rounded["total_T"],
        f"Error: {-delta}",
        delta_color=color
    )
    delta = np.round(results_raw_rounded['total_Mz'] - Mz_cmd, 1)
    color = 'inverse' if Tcmd < 0 else 'inverse' if delta != 0 else 'off'
    cols[0].metric(
        "Total Moment",
        results_raw_rounded["total_Mz"],
        f"Error: {-delta}",
        delta_color=color
    )

    st.divider()
    for i in range(4):
        grip = abs(np.round(results_raw["grip_allocation"][i], 2))
        cols[i%2 + 1].metric(
            f"{wheels[i]}", results_raw_rounded["T_final"][i],
            f"Ratio: {int(grip * 100)}%",
            delta_color='off'
        )

    with st.expander("Show Details"):
        st.write(results_raw)