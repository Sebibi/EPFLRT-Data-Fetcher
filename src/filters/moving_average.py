import numpy as np
import pandas as pd

def moving_average(array: list, window_size: int) -> list:
    """Calculates the moving average of a given array.

    Args:
        array (list): The array to calculate the moving average of.
        window_size (int): The size of the window to use for the moving average.

    Returns:
        list: The moving average of the given array.
    """
    moving_average = []
    for i in range(len(array)):
        if i < window_size:
            moving_average.append(sum(array[:i+1]) / (i+1))
        else:
            moving_average.append(sum(array[i-window_size+1:i+1]) / window_size)
    return moving_average