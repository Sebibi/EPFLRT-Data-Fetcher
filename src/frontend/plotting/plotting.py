from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go


def plot_data(
        data: pd.DataFrame, tab_name: str, title: str = "Sensors",
        default_columns: list = None, fig_ax: Tuple[plt.Figure, plt.Axes] = None,
        simple_plot: bool = False
) -> Tuple[List[str], List[str]]:

    if simple_plot:
        fig, ax = plt.subplots(figsize=(10, 5)) if fig_ax is None else fig_ax
        data[default_columns].plot(ax=ax)
        plt.tight_layout()
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        st.pyplot(fig)
        return default_columns, [data.index[0], data.index[-1]]



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
    use_plotly = cols[0].checkbox("Use Plotly", value=False, key=f"{tab_name} use plotly")
    legend = cols[0].checkbox("Show legend", value=True, key=f"{tab_name} show legend") if not use_plotly else False
    subplots = cols[0].checkbox("Subplots", value=False, key=f"{tab_name} subplots") if not use_plotly else False

    if use_plotly:
        if fig_ax is not None:
            st.warning("Plotly backend does not support custom axis, using default one")
        pd.options.plotting.backend = "plotly"
        plotly_fig = go.Figure()
        res = rolled_plot_data.plot()
        st.plotly_chart(res, use_container_width=True)
    else:
        pd.options.plotting.backend = "matplotlib"
        rolled_plot_data.plot(ax=ax, subplots=subplots, legend=legend)
        plt.tight_layout()
        if legend:
            ax.legend()
        ax.set_title(title)
        ax.set_xlabel('Time [s]')
        st.pyplot(fig)
    # Reset backend to default
    pd.options.plotting.backend = "matplotlib"
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


def plot_multiple_data(datas: list[pd.DataFrame], tab_name: str, title: str = "Sensors",
                       default_columns: list = None, fig_ax: Tuple[plt.Figure, plt.Axes] = None) -> Tuple[List[str], List[str]]:

    # Update column_names
    for i, data in enumerate(datas[1:]):
        datas[i+1].columns = [f"{col}_{i+1}" for col in datas[i+1].columns]
    full_data = pd.concat(datas, axis=1)

    columns_to_plot = full_data.columns
    samples_to_plot = st.select_slider(
        label="Number of samples to plot", options=full_data.index,
        value=[full_data.index[0], full_data.index[-1]], format_func=lambda x: f"{x:.2f}",
        key=f"{tab_name} samples to plot",
    )

    cols = st.columns(2)
    sub_cols = cols[0].columns(2)
    w = sub_cols[0].number_input("Figure Width", value=10, step=1, min_value=1, key=f"{tab_name} figure width")
    h= sub_cols[1].number_input("Figure Height ", value=5, step=1, min_value=1, key=f"{tab_name} figure height")
    fig, ax = plt.subplots(figsize=(w, h)) if fig_ax is None else fig_ax
    window_size = cols[1].number_input(f"Moving average to be applied to data", value=1, step=1, min_value=1, key=f"{tab_name} window size")
    rolled_plot_data = full_data[columns_to_plot].loc[samples_to_plot[0]:samples_to_plot[1]].rolling(window=window_size).mean()
    legend = sub_cols[1].checkbox("Show legend", value=True, key=f"{tab_name} show legend")
    plot_subplots = sub_cols[0].checkbox("Subplots", value=False, key=f"{tab_name} subplots")
    if plot_subplots:
        subplot_columns = [[full_col for full_col in columns_to_plot if col in full_col]for col in datas[0].columns]
    else:
        subplot_columns = False
    rolled_plot_data.plot(ax=ax, subplots=subplot_columns, legend=legend)

    plt.tight_layout()
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time [s]')
    st.pyplot(fig)
    return columns_to_plot, samples_to_plot