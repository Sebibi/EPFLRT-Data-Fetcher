from collections import deque

import numpy as np

delta = 20


def measure_wheel_acceleration(wheel_speeds: np.ndarray, hist=deque([np.zeros(4)], maxlen=delta)) -> np.ndarray:
    old_wheel_speeds = hist[0]
    hist.append(wheel_speeds)
    return (wheel_speeds - old_wheel_speeds) / (delta * 0.01)


if __name__ == '__main__':
    ws = np.array([1, 2, 3, 4])
    for i in range(100):
        print(measure_wheel_acceleration(ws))
