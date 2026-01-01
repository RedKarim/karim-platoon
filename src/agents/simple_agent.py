from ..core.vehicle import VehicleControl
import math
import numpy as np

class SimpleAgent:
    def __init__(self, vehicle, behavior='normal'):
        self.vehicle = vehicle
        self.behavior = behavior
        self.target_speed = 50 / 3.6
        self.k_p = 1.0
        self.k_d = 0.5
        self.k_i = 0.0
        self.prev_error = 0
        self.integral_error = 0
        self.max_throttle = 0.75
        self.max_brake = 0.3
        self.desired_gap = 10.0
        
    def _update_information(self, ego_vehicle_speed):
        self.target_speed = ego_vehicle_speed / 3.6 # input in km/h? check usage

    def run_step(self, leading_vehicle, distance, distance_to_packleader, leading_vehicle_speed, leader_is_ego, debug=False):
        # Simple ACC
        target_gap = self.desired_gap
        error = distance - target_gap
        
        # PD Control on gap
        p_term = self.k_p * error
        d_term = self.k_d * (leading_vehicle_speed - self.get_speed())
        
        control_signal = p_term + d_term
        
        control = VehicleControl()
        if control_signal > 0:
            control.throttle = np.clip(control_signal, 0, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-control_signal, 0, self.max_brake)
            
        control.steer = 0.0 # Straight line following for now
        return control

    def set_global_plan(self, plan):
        pass
        
    def get_speed(self):
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2)
