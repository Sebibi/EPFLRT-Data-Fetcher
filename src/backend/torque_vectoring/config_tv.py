from src.backend.state_estimation.config.vehicle_params import VehicleParams

class TVParams:

    kp = 100
    Ki = 0
    K_understeer = 0

    wheel_base = 1.570

    @classmethod
    def set_K_understeer(cls, K_understeer: float):
        cls.K_understeer = K_understeer