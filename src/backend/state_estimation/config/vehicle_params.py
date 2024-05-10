from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


class VehicleParams:

    # Sampling time
    dt = 0.01

    # Control Modes
    ControlMode = defaultdict(lambda: 'Unkown', {1:'Rien', 2:'TV', 3:'TCO', 4:'TCC', 5:'TCO + TV', 6:'TCC + TV', 7:'TCOL'})

    # Aero
    A_front = 1.61
    c_drag = 0.8 # 1.55 #
    c_lift = 0.8 # 4.6 # 0.8
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
    g = 9.785
    lzz = 180
    a = 1.24
    b = 1.24
    l = 1.57
    lf = 0.785
    lr = 0.785
    m_car = 300
    z_cg = 0.295

    # Wheel parameters
    Iw = 0.3 # 0.3
    Rw = 0.202
    kb = 0 # 0.18
    kd = 0.17
    ks = 5
    wheel_names = ['FL', 'FR', 'RL', 'RR']

    # Tire parameters
    B = 11.15
    C = 1.98
    D = 1.5
    E = 0.97
    BCD = B * C * D
    mu_init = 1.1
    # old_mu_max = 1.67

    # Cornering stiffness
    kf = 1.0
    kr = 1.0


    @classmethod
    def magic_formula(cls, slip_ratio: float | np.ndarray, mux: float = None) -> float | np.ndarray:
        """
        :param mux: friction coefficient from traction ellipse
        :param slip_ratio:
        :return: mu
        """
        B, C, D, E = cls.B, cls.C, cls.D, cls.E
        if mux is not None:
            D = mux
        mu = D * np.sin(C * np.arctan(B * slip_ratio - E * (B * slip_ratio - np.arctan(B * slip_ratio))))
        return mu

    @classmethod
    def linear_inverse_magic_formula(cls, mu: float | np.ndarray) -> float | np.ndarray:
        """
        :param mu:
        :return: slip_ratio
        """
        return mu / cls.BCD

    @classmethod
    def set_mu_max(cls, mu_max: float):
        cls.D = mu_max


if __name__ == '__main__':
    slip_range = np.linspace(-0.5, 0.5, 100)
    mus = [VehicleParams().magic_formula(s) for s in slip_range]
    fmus = list(filter(lambda x: abs(x) < 1.5, mus))
    slip_range_inv = [VehicleParams().linear_inverse_magic_formula(mu) for mu in fmus]

    plt.plot(slip_range, mus, label="magic formula")
    # plt.plot(slip_range_inv, fmus, label="linear")
    # plt.axvline(x=0.05, color='r')
    # plt.axvline(x=-0.05, color='r')

    # Plot the maximum and minimum friction coefficient
    plt.axhline(y=np.max(mus), color='g')
    plt.axhline(y=np.min(mus), color='g')

    # Plot the slip ratio corresponding to 1.0 friction coefficient
    plt.axvline(x=VehicleParams().linear_inverse_magic_formula(1.2), color='black')

    plt.text(0.05, np.max(mus), f"mu_max={np.max(mus):.2f}")
    plt.text(0.05, np.min(mus), f"mu_min={np.min(mus):.2f}")

    # Plot the optimal slip ratio
    plt.axvline(x=slip_range[np.argmax(mus)], color='r')
    plt.axvline(x=slip_range[np.argmin(mus)], color='r')
    plt.text(slip_range[np.argmax(mus)], 0.5, f"optimal slip ratio={slip_range[np.argmax(mus)]:.5f}")

    # Set x axis resolution to 0.1
    plt.xticks(np.arange(-0.5, 0.5, 0.1))
    
    plt.grid()
    plt.xlabel("Slip ratio")
    plt.ylabel("Friction coefficient")
    plt.legend()
    plt.show()
