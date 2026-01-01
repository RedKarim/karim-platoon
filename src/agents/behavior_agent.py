# Copyright (c) 2025, MAHDYAR KARIMI
# Ported from EcoLead BehaviorAgent to work without CARLA

"""
BehaviorAgent implements the exact logic from EcoLead's BehaviorAgent.
This is a complete port with CARLA dependencies removed.
"""

from ..core.vehicle import VehicleControl
from ..core.traffic_light import TrafficLightState  
import math
import numpy as np
from collections import deque

#
# Behavior Types (from EcoLead)
class Cautious:
    """Class for Cautious agent."""
    max_speed = 40
    speed_lim_dist = 6
    speed_decrease = 12
    safety_time = 3
    min_proximity_threshold = 12
    braking_distance = 6
    tailgate_counter = 0

class Normal:
    """Class for Normal agent."""
    max_speed = 50
    speed_lim_dist = 3
    speed_decrease = 10
    safety_time = 3
    min_proximity_threshold = 10
    braking_distance = 5
    tailgate_counter = 0

class Aggressive:
    """Class for Aggressive agent."""
    max_speed = 70
    speed_lim_dist = 1
    speed_decrease = 8
    safety_time = 3
    min_proximity_threshold = 8
    braking_distance = 4
    tailgate_counter = -1

def get_speed(vehicle):
    """Get vehicle speed in km/h."""
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2) * 3.6  # m/s to km/h

class BehaviorAgent:
    """
    BehaviorAgent - Complete port from EcoLead.
    Implements CACC with adaptive PID gains, exactly matching EcoLead's logic.
    """
    
    def __init__(self, vehicle, behavior='normal'):
        self._vehicle = vehicle
        self._max_brake = 0.3
        
        # Vehicle information
        self._speed = 0
        self._speed_limit = 50/3.6
        self._min_speed = 5
        self.previous_target_speed = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.start_time = None
        self.filtered_derivative = 0.0
        
        # Waypoints for lateral control
        self.waypoints = None
        self.current_waypoint_index = 0
        self.waypoint_reached_threshold = 5.0
        
        # PID lateral controller
        self.lat_K_P = 1.0
        self.lat_K_I = 0.0
        self.lat_K_D = 0.0
        self.lat_dt = 0.1
        self.lat_error_buffer = deque(maxlen=10)
        
        # Steering smoothing
        self.past_steering = 0.0
        self.max_steer_change = 0.05  # Reduced from 0.1 for smoother, less noisy steering
        self.max_steer = 0.8
        
        # Behavior parameters
        if behavior == 'cautious':
            self._behavior = Cautious()
        elif behavior == 'normal':
            self._behavior = Normal()
        elif behavior == 'aggressive':
            self._behavior = Aggressive()
        else:
            self._behavior = Normal()
        
        # Traffic light manager reference
        self._traffic_light_manager = None
    
    def _update_information(self, ego_vehicle_speed=None):
        """Update agent information."""
        self._speed = get_speed(self._vehicle)
        if ego_vehicle_speed is not None:
            self._speed_limit = ego_vehicle_speed
        
    def set_global_plan(self, plan):
        """Set waypoints for path following."""
        if plan and len(plan) > 0:
            self.waypoints = [wp for wp, _ in plan]
        else:
            self.waypoints = None
    
    def _update_waypoint_progress(self, current_location):
        """Update waypoint progress."""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return
        
        while self.current_waypoint_index < len(self.waypoints):
            target_wp = self.waypoints[self.current_waypoint_index]
            target_location = target_wp.transform.location
            distance = current_location.distance(target_location)
            
            if distance < self.waypoint_reached_threshold:
                self.current_waypoint_index += 1
            else:
                break
    
    def _calculate_steering(self, vehicle_transform):
        """PID lateral controller."""
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return 0.0
        
        target_waypoint = self.waypoints[self.current_waypoint_index]
        ego_loc = vehicle_transform.location
        yaw_rad = math.radians(vehicle_transform.rotation.yaw)
        v_vec = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0])
        
        w_loc = target_waypoint.transform.location
        w_vec = np.array([w_loc.x - ego_loc.x, w_loc.y - ego_loc.y, 0.0])
        
        wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
        if wv_linalg == 0:
            angle_error = 0.0
        else:
            angle_error = math.acos(np.clip(np.dot(w_vec, v_vec) / wv_linalg, -1.0, 1.0))
        
        cross = np.cross(v_vec, w_vec)
        if cross[2] < 0:
            angle_error *= -1.0
        
        self.lat_error_buffer.append(angle_error)
        if len(self.lat_error_buffer) >= 2:
            de = (self.lat_error_buffer[-1] - self.lat_error_buffer[-2]) / self.lat_dt
            ie = sum(self.lat_error_buffer) * self.lat_dt
        else:
            de = 0.0
            ie = 0.0
        
        steering = np.clip((self.lat_K_P * angle_error) + (self.lat_K_D * de) + (self.lat_K_I * ie), -1.0, 1.0)
        
        # Steering rate limiting
        if steering > self.past_steering + self.max_steer_change:
            steering = self.past_steering + self.max_steer_change
        elif steering < self.past_steering - self.max_steer_change:
            steering = self.past_steering - self.max_steer_change
        
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        self.past_steering = steering
        return steering
    
    def car_following_manager_ramp(self, vehicle, distance, debug=True):
        """Ramp-up phase CACC (from EcoLead)."""
        T = 1.0
        d0 = 0.0
        MAX_SPEED = self._behavior.max_speed
        MIN_SPEED = self._min_speed
        SPEED_LIMIT = self._speed_limit
        DECEL_FACTOR = 0.2
        EMERGENCY_TTC = 3.0
        
        if not hasattr(self, "previous_target_speed"):
            self.previous_target_speed = 0.0
        
        ego_speed_kmh = self._speed
        leader_speed_kmh = get_speed(vehicle)
        ego_speed_ms = ego_speed_kmh / 3.6
        leader_speed_ms = leader_speed_kmh / 3.6
        delta_v = ego_speed_ms - leader_speed_ms
        epsilon = 1e-3
        
        if delta_v > 0:
            ttc = distance / (delta_v + epsilon)
        else:
            ttc = float('inf')
        
        desired_distance = d0 + (leader_speed_ms * T)
        lower_band = 0.9 * desired_distance
        upper_band = 1.1 * desired_distance
        
        if ttc < EMERGENCY_TTC:
            target_speed = max(MIN_SPEED, leader_speed_kmh * (1 - DECEL_FACTOR))
        else:
            if distance < lower_band:
                decel_speed = ego_speed_kmh * 0.95
                target_speed = min(decel_speed, leader_speed_kmh)
            elif distance > upper_band:
                gap_error = distance - desired_distance
                speed_increase_kmh = ((gap_error / T) * 3.6) / 3
                raw_target_speed = ego_speed_kmh + speed_increase_kmh
                overshoot_margin = (SPEED_LIMIT - leader_speed_kmh)
                allowed_speed = leader_speed_kmh + overshoot_margin
                target_speed = min(raw_target_speed, allowed_speed)
                
                LOW_SPEED_THRESHOLD = 7.3
                MIN_BUMP = 0.2
                if SPEED_LIMIT < LOW_SPEED_THRESHOLD:
                    min_target_for_low_speed = ego_speed_kmh + MIN_BUMP
                    target_speed = min(target_speed, min_target_for_low_speed)
            else:
                target_speed = leader_speed_kmh
       
        target_speed = max(MIN_SPEED, target_speed)
        target_speed = min(target_speed, MAX_SPEED, SPEED_LIMIT)
        
        alpha = 0.2
        old_speed = self.previous_target_speed
        smoothed_speed = alpha * old_speed + (1 - alpha) * target_speed
        self.previous_target_speed = smoothed_speed
        
        # Generate control
        control = VehicleControl()
        speed_error = (smoothed_speed - ego_speed_kmh) / 3.6
        if speed_error > 0:
            control.throttle = np.clip(speed_error * 2.0, 0, 0.75)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-speed_error * 2.0, 0, 0.3)
        
        return control
    
    def car_following_manager(self, vehicle, current_gap, debug=True):
        """Main CACC implementation (from EcoLead)."""
        current_time = 0.1 * (hasattr(self, '_tick_count') and self._tick_count or 0)
        if not hasattr(self, '_tick_count'):
            self._tick_count = 0
        self._tick_count += 1
        
        if self.start_time is None:
            self.start_time = current_time
        
        dt = 0.1
        v_ego = self._speed / 3.6
        v_prev = get_speed(vehicle) / 3.6
        v_leader = self._speed_limit / 3.6
        
        sim_time_elapsed = current_time - self.start_time
        if sim_time_elapsed < 10:
            if v_ego < 13:
                return self.car_following_manager_ramp(vehicle, current_gap)
        
        min_gap, max_gap = 6, 15.0
        s_0 = 1
        T_gap = 1
        if v_ego < 5:
            s_desired = max(min_gap, s_0 + v_ego * T_gap * 0.6)
        else:
            s_desired = max(min_gap, min(s_0 + v_ego * T_gap, max_gap))
        
        gap_error = current_gap - s_desired
        
        K1_min, K1_max = 0.2, 0.6
        K2_min, K2_max = 0.3, 0.7
        abs_gap_error = abs(current_gap - s_desired)
        K1 = K1_min + (K1_max - K1_min) * (1 - np.exp(-abs_gap_error / max_gap))
        K2 = K2_min + (K2_max - K2_min) * (1 - np.exp(-abs_gap_error / max_gap))
        
        v_target = v_prev + K1 * gap_error + K2 * (v_leader - v_prev)
        
        if v_ego < 5:
            Kp, Ki, Kd = 0.35, 0.02, 0.5
        elif 5 <= v_ego < 8:
            Kp, Ki, Kd = 0.30, 0.04, 0.65
        elif 8 <= v_ego < 12:
            Kp, Ki, Kd = 0.25, 0.05, 0.50
        else:
            Kp, Ki, Kd = 0.20, 0.05, 0.30
        
        v_target = max(0, min(v_target, (self._speed_limit) / 3.6 + 2))
        speed_error = v_target - v_ego
        
        raw_derivative = (speed_error - self.prev_error) / dt
        alpha = 0.1
        self.filtered_derivative = alpha * raw_derivative + (1 - alpha) * getattr(self, 'filtered_derivative', raw_derivative)
        self.prev_error = speed_error
        
        v_ego += Kp * speed_error + Ki * self.integral + Kd * self.filtered_derivative
        v_ego = max(0, min(v_ego, self._speed_limit / 3.6))
        self.previous_target_speed = v_ego
        
        # Generate control
        control = VehicleControl()
        control_error = v_ego - (self._speed / 3.6)
        if control_error > 0:
            control.throttle = np.clip(control_error * 2.0, 0, 0.75)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = np.clip(-control_error * 2.0, 0, 0.3)
        
        return control
    
    def emergency_stop(self):
        """Emergency stop."""
        control = VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control
    
    def run_step(self, vehicle, distance, distance_to_packleader, leader_velocity, leader, debug=False):
        """Main control step (from EcoLead)."""
        ego_vehcile_speed = get_speed(self._vehicle)
        
        if leader:
            # This is leader - simple cruise
            control = VehicleControl()
            control.throttle = 0.5
            control.brake = 0.0
        else:
            # Check traffic lights
            if hasattr(self, '_traffic_light_manager') and self._traffic_light_manager:
                vehicle_location = self._vehicle.get_location()
                for tl_id, tl_data in self._traffic_light_manager.traffic_lights.items():
                    if tl_data['current_state'] in [TrafficLightState.Red, TrafficLightState.Yellow]:
                        tl_loc = tl_data['actor'].transform.location
                        dist_to_light = vehicle_location.distance(tl_loc)
                        if 0 <= dist_to_light <= 2.0:
                            control = self.emergency_stop()
                            # Apply steering
                            vehicle_transform = self._vehicle.get_transform()
                            current_location = self._vehicle.get_location()
                            self._update_waypoint_progress(current_location)
                            steering = self._calculate_steering(vehicle_transform)
                            control.steer = steering
                            return control
            
            # Check braking distance
            col_distance = distance  # Simplified - no collision detection
            if distance > 0 and col_distance >= -1:
                if col_distance < self._behavior.braking_distance and col_distance > 0:
                    control = self.emergency_stop()
                else:
                    control = self.car_following_manager(vehicle, distance, debug=True)
            else:
                # Normal behavior
                control = VehicleControl()
                control.throttle = 0.5
                control.brake = 0.0
        
        # Apply steering
        vehicle_transform = self._vehicle.get_transform()
        current_location = self._vehicle.get_location()
        self._update_waypoint_progress(current_location)
        steering = self._calculate_steering(vehicle_transform)
        control.steer = steering
        
        return control
