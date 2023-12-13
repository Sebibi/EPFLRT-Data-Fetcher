import pandas as pd

from src.frontend.plotting.plotting import plot_data


def plot_model_analysis(data: pd.DataFrame, tab_name: str, acc_fsum_cols: list, acc_fsum_est_cols: list):
    plot_data(
        data=data,
        tab_name=tab_name + "_model_analysis",
        default_columns=["sensors_accX", "sensors_accY"] + acc_fsum_cols + acc_fsum_est_cols,
        title="Model analysis",
    )
