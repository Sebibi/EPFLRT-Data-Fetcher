import numpy as np
import matplotlib.pyplot as plt


class VehicleParams:

    # Sampling time
    dt = 0.01

    # Aero
    A_front = 1.61
    c_drag = 0.97
    c_lift = 2.33
    rho_air = 1.2

    # Motor
    w_m_max = 681
    tau_m_max = 21
    gear_ratio = 13.188
    tau_w_max = 129

    # Steering
    kin = 0.233
    kout = 0.162

    # Rigid parameters
    g = 0.81
    lzz = 180
    a = 1.24
    b = 1.24
    l = 1.57
    lf = 0.785
    lr = 0.785
    m_car = 300
    z_cg = 0.295

    # Wheel parameters
    Iw = 0.3
    Rw = 0.202
    kb = 0.18
    kd = 0.17
    ks = 15
    wheel_names = ['FL', 'FR', 'RL', 'RR']

    # Tire parameters
    B = 11.15
    C = 1.98
    D = 1.67
    E = 0.97
    BCD = B * C * D

    @classmethod
    def magic_formula(cls, slip_ratio: float | np.ndarray) -> float | np.ndarray:
        """
        :param slip_ratio:
        :return: mu
        """
        B, C, D, E = cls.B, cls.C, cls.D, cls.E
        return D * np.sin(C * np.arctan(B * slip_ratio - E * (B * slip_ratio - np.arctan(B * slip_ratio))))

    @classmethod
    def linear_inverse_magic_formula(cls, mu: float | np.ndarray) -> float | np.ndarray:
        """
        :param mu:
        :return: slip_ratio
        """
        return mu / cls.BCD


if __name__ == '__main__':
    slip_range = np.linspace(-0.3, 0.3, 100)
    mus = [VehicleParams().magic_formula(s) for s in slip_range]
    fmus = list(filter(lambda x: abs(x) < 1.5, mus))
    slip_range_inv = [VehicleParams().linear_inverse_magic_formula(mu) for mu in fmus]

    plt.plot(slip_range, mus, label="magic formula")
    plt.plot(slip_range_inv, fmus, label="linear")
    plt.axvline(x=0.05, color='r')
    plt.axvline(x=-0.05, color='r')

    plt.show()
