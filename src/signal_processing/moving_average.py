import pandas as pd
import streamlit as st


def moving_avg_input(data: pd.DataFrame, key: str, label: str = None) -> pd.DataFrame:
    """
    Perform a moving average on the data.
    :param data: data to perform the moving average on
    :param key: widget key
    :param label: widget label
    :return: data with moving average applied
    """
    if label is None:
        label = "Moving average window size [10ms]"
    window_size = int(st.number_input(
        label=label,
        value=1, min_value=1, max_value=1000, key=key))
    return data.rolling(window=window_size).mean()
