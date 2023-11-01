from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from src.api_call.base import Fetcher
from src.utils import date_to_influx


class SessionCreator:

    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def fetch_r2d_session(self, start_date: pd.Timestamp, end_date: pd.Timestamp, verify_ssl: bool) -> List[
        pd.DataFrame]:
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
            df_r2d = self.fetcher.fetch_data(query=query_r2d, verify_sll=verify_ssl)
        if len(df_r2d) == 0:
            return []
        else:
            st.info(df_r2d)
            threshold = pd.Timedelta(seconds=10)
            separation_indexes = df_r2d.index[df_r2d.index.to_series().diff() > threshold].tolist()
            separation_indexes = [df_r2d.index[0]] + separation_indexes + [df_r2d.index[-1]]
            dfs = [df_r2d.loc[separation_indexes[i]:separation_indexes[i + 1]] for i in range(len(separation_indexes) - 1)]
            dfs = [df[:-1] for df in dfs]
            return dfs

    def fetch_data(self, datetime_range: str, verify_ssl: bool) -> pd.DataFrame:
        query = f"""from(bucket:"ariane")
        |> {datetime_range}
        |> pivot(rowKey:["_time"], columnKey: ["_measurement", "_field"], valueColumn: "_value")
        |> drop(columns: ["_start", "_stop"])
        |> yield(name: "mean")
        """

        with st.spinner("Fetching session data from InfluxDB..."):
            return self.fetcher.fetch_data(query, verify_sll=verify_ssl)

    def r2d_session_selector(self, dfs: List[pd.DataFrame]) -> str:
        dfs_options = [f"range(start: {date_to_influx(df.index[0])}, stop: {date_to_influx(df.index[-1])})"
                       for
                       df in dfs]

        dfs_elapsed_time = [str((df.index[-1] - df.index[0]).floor('s'))[7:] for df in dfs]

        options_index = list(np.arange(len(dfs)))
        session_index = st.selectbox(
            "Session", options=options_index, index=0,
            label_visibility="collapsed",
            format_func=lambda i: f"{i} - Duration ({dfs_elapsed_time[i]}) : " + dfs_options[i]
        )
        return dfs_options[session_index]

    def r2d_multi_session_selector(self, dfs: List[pd.DataFrame]) -> List[str]:
        dfs_options = [f"range(start: {date_to_influx(df.index[0])}, stop: {date_to_influx(df.index[-1])})"
                       for
                       df in dfs]

        dfs_elapsed_time = [str((df.index[-1] - df.index[0]).floor('s'))[7:] for df in dfs]

        options_index = list(np.arange(len(dfs)))
        session_indexes = st.multiselect(
            label="Sessions", options=options_index,
            label_visibility="collapsed",
            default=options_index,
            format_func=lambda i: f"{i} - Duration ({dfs_elapsed_time[i]}) : " + dfs_options[i]
        )
        return [dfs_options[i] for i in session_indexes]
