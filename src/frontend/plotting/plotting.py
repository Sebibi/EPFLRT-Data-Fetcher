from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def plot_data(
        data: pd.DataFrame, tab_name: str, title: str = "Sensors",
        default_columns: list = None, fig_ax: Tuple[plt.Figure, plt.Axes] = None) -> Tuple[List[str], List[str]]:

    columns_to_plot = st.multiselect(
        label="Select the labels to plot",
        options=data.columns,
        default=["sensors_aXEst", "sensors_vXEst"] if default_columns is None else default_columns,
        key=f"{tab_name} columns to plot",
    )
    samples_to_plot = st.select_slider(
        label="Number of samples to plot", options=data.index,
        value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}",
        key=f"{tab_name} samples to plot",
    )
    plot_data = data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]]

    cols = st.columns(2)
    fig, ax = plt.subplots(figsize=(10, 5)) if fig_ax is None else fig_ax
    window_size = cols[1].number_input(f"Moving average to be applied to data", value=1, step=1, min_value=1, key=f"{tab_name} window size")
    rolled_plot_data = plot_data.rolling(window=window_size).mean()
    legend = cols[0].checkbox("Show legend", value=True, key=f"{tab_name} show legend")
    rolled_plot_data.plot(ax=ax, subplots=cols[0].checkbox("Subplots", value=False, key=f"{tab_name} subplots"), legend=legend)
    plt.tight_layout()
    if legend:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    st.pyplot(fig)
    return columns_to_plot, samples_to_plot


def plot_data_comparaison(
        data: pd.DataFrame, tab_name: str, title: str = "Sensors",
        default_columns: list = None, fig_ax: Tuple[plt.Figure, plt.Axes] = None,
        comparison_names: List[str] = ["positive", "negative"],
        extra_columns: List[str] = [],
) -> Tuple[List[str], List[str]]:

    columns_to_plot = default_columns
    assert len(columns_to_plot) == 2

    samples_to_plot = st.select_slider(
        label="Number of samples to plot", options=data.index,
        value=[data.index[0], data.index[-1]], format_func=lambda x: f"{x:.2f}",
        key=f"{tab_name} samples to plot",
    )
    plot_data = data[columns_to_plot + extra_columns].loc[samples_to_plot[0]:samples_to_plot[1]]


    cols = st.columns(2)
    fig, ax = plt.subplots(figsize=(10, 5)) if fig_ax is None else fig_ax
    window_size = cols[1].number_input(f"Moving average to be applied to data", value=1, step=1, min_value=1, key=f"{tab_name} window size")
    rolled_plot_data = plot_data.rolling(window=window_size).mean()

    legend = cols[0].checkbox("Show legend", value=True, key=f"{tab_name} show legend")
    extra = cols[1].checkbox("Show extra columns", value=True, key=f"{tab_name} show extra columns")

    time = rolled_plot_data.index
    series1 = rolled_plot_data[columns_to_plot[0]]
    series2 = rolled_plot_data[columns_to_plot[1]]

    ax.plot(time, series1, label=columns_to_plot[0])
    ax.plot(time, series2, label=columns_to_plot[1], color='black')
    ax.fill_between(time, series1, series2, where=(np.abs(series1) > np.abs(series2)), facecolor='green', alpha=0.3, interpolate=True, label=comparison_names[0])
    ax.fill_between(time, series1, series2, where=(np.abs(series1) < np.abs(series2)), facecolor='red', alpha=0.3, interpolate=True, label=comparison_names[1])

    if extra:
        for c in extra_columns:
            ax.plot(time, rolled_plot_data[c], label=c)

    # Compute performance metrics and display them
    mse = ((series1 - series2) ** 2).mean()
    mae = (np.abs(series1 - series2)).mean()
    rmse = np.sqrt(mse)

    cols[0].markdown(f"**MSE**: {mse:.2f}, **MAE**: {mae:.2f},  **RMSE**: {rmse:.2f}")

    if legend:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    st.pyplot(fig)
    return columns_to_plot, samples_to_plot