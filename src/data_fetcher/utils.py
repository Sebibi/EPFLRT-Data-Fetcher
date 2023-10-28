from typing import List

import numpy as np
import streamlit as st
import pandas as pd
from influxdb_client import InfluxDBClient


token = "cCtyspyt-jeehwf5Ayz5OmaOXnvkMj46z3C6UQlud4s8MiPZLFaFuM7z1Y_qqpmVyI5cvF4h9k-kl5dCiYmWFw=="
org = "racingteam"


def date_to_influx(date: pd.Timestamp) -> str:
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_r2d_session(start_date: pd.Timestamp, end_date: pd.Timestamp, verify_ssl: bool) -> List[pd.DataFrame]:
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
        with InfluxDBClient(url="https://epfl-rt-data-logging.epfl.ch:8443", token=token, org=org,
                            verify_ssl=verify_ssl) as client:
            df_r2d = client.query_api().query_data_frame(query=query_r2d, org=org)
            if len(df_r2d) == 0:
                return []
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


def fetch_data(datetime_range: str, verify_ssl: bool) -> pd.DataFrame:
    query = f"""from(bucket:"ariane")
    |> {datetime_range}
    |> pivot(rowKey:["_time"], columnKey: ["_measurement", "_field"], valueColumn: "_value")
    |> drop(columns: ["_start", "_stop"])
    |> yield(name: "mean")
    """

    with st.spinner("Fetching data from InfluxDB..."):
        with InfluxDBClient(url="https://epfl-rt-data-logging.epfl.ch:8443", token=token, org=org,
                            verify_ssl=verify_ssl) as client:
            df = client.query_api().query_data_frame(query=query, org=org)
            if len(df) == 0:
                return pd.DataFrame()
            df.drop(columns=["result", "table"], inplace=True)
            df.set_index("_time", inplace=True)
    return df
