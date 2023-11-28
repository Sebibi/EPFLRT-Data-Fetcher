import numpy as np


class VehicleParams:

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
    z_cg = 0.288

    # Wheel parameters
    lw = 0.3
    Rw = 0.2
    kb = 0.18
    kd = 0.17
    ks = 15

    # Tire parameters
    B = 11.15
    C = 1.98
    D = 1.67
    E = 0.97


    def magic_formula(self, Fz, kappa, alpha) -> float:
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return D * np.sin(C * np.arctan(B * kappa - E * (B * kappa - np.arctan(B * kappa)))) + alpha


    def inverse_magic_formula(self, Fz, Fx, alpha) -> float:
        B = self.B
        C = self.C
        D = self.D
        E = self.E
        return np.tan((1 / B) * (np.arctan(Fx / (D * Fz)) + E * np.arctan(Fx / (D * Fz))))