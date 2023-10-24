import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
import streamlit as st
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import MinMaxScaler

def date_to_influx(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def init_sessions_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = []

    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame()


def fetch_r2d_session(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.DataFrame]:
    start_date = date_to_influx(start_date)
    end_date = date_to_influx(end_date)

    query_r2d = f"""from(bucket:"ariane") 
    |> range(start: {start_date}, stop: {end_date})
    |> filter(fn: (r) => r["_measurement"] == "MISC")
    |> filter(fn: (r) => r["_field"] == "FSM")
    |> filter(fn: (r) => r["_value"] == "R2D")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> drop(columns: ["_start", "_stop"])
    |> yield(name: "max")
    """

    with st.spinner("Fetching r2d sessions from InfluxDB..."):
        with InfluxDBClient(url="https://epfl-rt-data-logging.epfl.ch:8443", token=token, org=org, verify_ssl=verify_sll) as client:
            df_r2d = client.query_api().query_data_frame(query=query_r2d, org=org)
            df_r2d.drop(columns=["result", "table"], inplace=True)
            df_r2d.set_index("_time", inplace=True)

    threshold = pd.Timedelta(seconds=10)
    separation_indexes = df_r2d.index[df_r2d.index.to_series().diff() > threshold].tolist()
    separation_indexes = [df_r2d.index[0]] + separation_indexes + [df_r2d.index[-1]]
    dfs = [df_r2d.loc[separation_indexes[i]:separation_indexes[i + 1]] for i in range(len(separation_indexes) - 1)]
    dfs = [df[:-1] for df in dfs]
    return dfs

def choose_r2d_session(dfs: List[pd.DataFrame]) -> str:
    date_format = "%Y-%m-%dT%H:%M:%SZ"
    dfs_options = [f"range(start: {df.index[0].strftime(date_format)}, stop: {df.index[-1].strftime(date_format)})" for
                   df in dfs]

    dfs_elapsed_time = [str((df.index[-1] - df.index[0]).floor('s'))[7:] for df in dfs]

    options_index = list(np.arange(len(dfs)))
    session_index = st.selectbox("Session", options=options_index, index=0, label_visibility="collapsed",
                                 format_func=lambda i: f"{i} - Duration ({dfs_elapsed_time[i]}) : " + dfs_options[i])
    return dfs_options[session_index]


def fetch_data(datetime_range: str) -> pd.DataFrame:
    query = f"""from(bucket:"ariane")
    |> {datetime_range}
    |> pivot(rowKey:["_time"], columnKey: ["_measurement", "_field"], valueColumn: "_value")
    |> drop(columns: ["_start", "_stop"])
    |> yield(name: "mean")
    """

    with st.spinner("Fetching data from InfluxDB..."):
        with InfluxDBClient(url="https://epfl-rt-data-logging.epfl.ch:8443", token=token, org=org, verify_ssl=verify_sll) as client:
            df = client.query_api().query_data_frame(query=query, org=org)
            df.drop(columns=["result", "table"], inplace=True)
            df.set_index("_time", inplace=True)
    return df


if __name__ == '__main__':

    # Initialize the application
    st.set_page_config(layout="wide")
    init_sessions_state()
    st.title("Data Fetcher")
    verify_sll = st.checkbox("Fetch with SSL", value=True)

    # API information
    st.subheader("Fetch data from InfluxDB")
    token = "cCtyspyt-jeehwf5Ayz5OmaOXnvkMj46z3C6UQlud4s8MiPZLFaFuM7z1Y_qqpmVyI5cvF4h9k-kl5dCiYmWFw=="
    org = "racingteam"
    st.markdown(f"**Organisation**: {org}")
    st.markdown(f"**Api token**: {token}")

    # Choose date range
    date_cols = st.columns(2)
    start_date = date_cols[0].date_input("Start date", value=pd.to_datetime("2023-10-05"))
    end_date = date_cols[1].date_input("End date", value=pd.to_datetime("2023-10-06"))

    # Fetch R2D sessions
    fetch = st.button("Fetch R2D sessions")
    if fetch:
        dfs = fetch_r2d_session(start_date, end_date)
        st.session_state.sessions = dfs
        st.success(f"Fetched {len(dfs)} sessions, select one in the dropdown below")

    # Choose R2D session and fetch data
    if len(st.session_state.sessions) > 0:
        st.header("Choose a session !")
        datetime_range = choose_r2d_session(st.session_state.sessions)
        if st.button("Fetch data"):
            data = fetch_data(datetime_range)
            data.index = (data.index - data.index[0]).total_seconds()
            st.session_state.data = data

    if len(st.session_state.data) > 0:
        # measurements = ["MISC", "AMS", "VSI", "sensors"]
        data = st.session_state.data
        # Select the Data
        st.subheader("Data to Download")
        # if st.button(label="Set start time to 0"):
        #    st.session_state.data.index = st.session_state.data.index - st.session_state.data.index[0]
        selected_columns = st.multiselect(label="Select the fields you want to download", options=data.columns, default=list(data.columns[:2]))
        samples_to_select = st.select_slider(label="Number of samples to select", options=data.index, value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}")
        output_data = data[selected_columns].loc[samples_to_select[0]:samples_to_select[1]]
        st.dataframe(output_data)

        # Download data
        file_name = st.text_input("File name", value="output_data.csv")
        st.download_button(
            label="Download data as CSV",
            data=output_data.to_csv().encode("utf-8"),
            file_name=file_name,
        )

        # Plot data
        st.subheader("Plot some data")

        columns_to_plot = st.multiselect(label="Select the labels to plot", options=data.columns, default=list(data.columns[:2]))
        samples_to_plot = st.select_slider(label="Number of samples to plot", options=data.index, value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}")
        plot_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]
        if st.checkbox("Smooth data"):
            smooth_cols = st.multiselect(label="Select the labels to smooth", options=plot_data.columns, default=list(plot_data.columns))
            plot_data[smooth_cols] = plot_data[smooth_cols].rolling(window=7).mean()

        if st.checkbox("Normalize data"):
            scaler = MinMaxScaler()
            data['sensors_vX'] = abs(data['sensors_vX'])
            scale_cols = st.multiselect(label="Select the labels to scale", options=plot_data.columns, default=list(plot_data.columns))
            plot_data[scale_cols] = pd.DataFrame(scaler.fit_transform(plot_data[scale_cols]), columns=scale_cols, index=plot_data.index)
        fig, ax = plt.subplots(figsize=(16, 9))
        # ax.axhline(y=25, color='r', linestyle='-', label='Horizontal Line at y=25')
        plot_data.plot(ax=ax)
        st.pyplot(fig, use_container_width=False)

