import math
import numpy as np
from ..logic.idm import IDM
from ..core.vehicle import VehicleControl
from .. import utils
from ..core.traffic_light import TrafficLightState

class PurePursuitController:
    """Pure pursuit lateral controller for path following."""
    
    def __init__(self, lookahead_gain=1.0, min_lookahead=2.0, max_lookahead=10.0):
        self.lookahead_gain = lookahead_gain
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
    
    def calculate_steering(self, vehicle_transform, waypoints, current_index, speed):
        """Calculate steering angle using pure pursuit algorithm."""
        # Calculate lookahead distance based on speed
        lookahead_distance = np.clip(
            self.lookahead_gain * speed, 
            self.min_lookahead, 
            self.max_lookahead
        )
        
        # Find lookahead point
        lookahead_point = self._find_lookahead_point(
            vehicle_transform.location, 
            waypoints, 
            current_index, 
            lookahead_distance
        )
        
        if lookahead_point is None:
            return 0.0
        
        # Calculate steering angle
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Vector to lookahead point in global frame
        dx = lookahead_point.x - vehicle_x
        dy = lookahead_point.y - vehicle_y
        
        # Transform to vehicle frame
        local_x = dx * math.cos(-vehicle_yaw) - dy * math.sin(-vehicle_yaw)
        local_y = dx * math.sin(-vehicle_yaw) + dy * math.cos(-vehicle_yaw)
        
        # Pure pursuit formula: steer = atan(2 * L * sin(alpha) / ld)
        # where L is wheelbase, alpha is angle to lookahead point, ld is lookahead distance
        # Simplified: steer proportional to lateral error
        wheelbase = 2.8  # Match vehicle.py
        if abs(local_x) > 0.01:  # Avoid division by zero
            curvature = 2.0 * local_y / (lookahead_distance ** 2)
            steer_angle = math.atan(curvature * wheelbase)
            # Normalize to [-1, 1] range (CARLA style)
            max_steer = math.radians(45)  # Match vehicle.py max_steer_angle
            steering = np.clip(steer_angle / max_steer, -1.0, 1.0)
        else:
            steering = 0.0
        
        return steering
    
    def _find_lookahead_point(self, current_location, waypoints, current_index, lookahead_distance):
        """Find the lookahead point on the path."""
        if current_index >= len(waypoints):
            return None
        
        # Search forward from current waypoint for lookahead point
        accumulated_distance = 0.0
        prev_location = current_location
        
        for i in range(current_index, len(waypoints)):
            wp_location = waypoints[i].location
            segment_distance = prev_location.distance(wp_location)
            accumulated_distance += segment_distance
            
            if accumulated_distance >= lookahead_distance:
                return wp_location
            
            prev_location = wp_location
        
        # If we reach the end, return the last waypoint
        return waypoints[-1].location if waypoints else None

class IDMAgent:
    def __init__(self, vehicle, waypoints, config={}):
        self.vehicle = vehicle
        # waypoints expected to be Transform objects or dicts. 
        # If passed from main, likely transforms. 
        self.waypoints = waypoints 
        self.prev_acceleration = 0.0
        self.current_waypoint_index = 0
        
        # Parameters (matching EcoLead)
        self.DESIRED_VELOCITY = 17
        self.ACCELERATION_LIMITS = (-5.0, 4.0)
        self.TRAFFIC_LIGHT_MIN_DISTANCE = 0.5  # Fixed: was 5.0, now matches EcoLead
        self.EMERGENCY_DECELERATION = -4.0
        self.WAYPOINT_REACHED_THRESHOLD = 5.0  # Match EcoLead
        
        # Initialize pure pursuit controller
        self.lateral_controller = PurePursuitController(
            lookahead_gain=1.0,
            min_lookahead=2.0,
            max_lookahead=10.0
        )
        
        # Steering smoothing (match EcoLead's rate limiting)
        self.past_steering = 0.0
        self.max_steer_change = 0.03  # Reduced from 0.05 for smoother, less noisy steering
        self.max_steer = 0.8  # Max steering angle
        
    def get_vehicle_state(self):
        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        speed = math.sqrt(v.x**2 + v.y**2)
        return {
            'x': t.location.x,
            'y': t.location.y,
            'yaw': math.radians(t.rotation.yaw),
            'speed': speed,
            'acceleration': self.prev_acceleration
        }

    def _update_waypoint_progress(self, current_location):
        """Update waypoint progress based on current location."""
        if self.current_waypoint_index >= len(self.waypoints):
            return
        
        while self.current_waypoint_index < len(self.waypoints):
            target_location = self.waypoints[self.current_waypoint_index].location
            distance = current_location.distance(target_location)
            
            if distance < self.WAYPOINT_REACHED_THRESHOLD:
                self.current_waypoint_index += 1
            else:
                break

    def _check_traffic_light_braking(self, traffic_light_manager):
        current_location = self.vehicle.get_location()
        # Iterate over traffic lights
        for tl_id, tl_data in traffic_light_manager.traffic_lights.items():
            state = tl_data['current_state']
            if state in [TrafficLightState.Red, TrafficLightState.Yellow]:
                # Recompute direct distance
                tl_loc = tl_data['actor'].transform.location
                direct_dist = current_location.distance(tl_loc)
                
                # Fixed: Match EcoLead's condition (0 <= dist <= 0.5)
                if 0 <= direct_dist <= self.TRAFFIC_LIGHT_MIN_DISTANCE:
                     return True
        return False

    def on_tick(self, traffic_light_manager):
        v = self.vehicle.get_velocity()
        current_speed = math.sqrt(v.x**2 + v.y**2)
        current_location = self.vehicle.get_location()
        current_transform = self.vehicle.get_transform()

        # Update waypoint progress
        self._update_waypoint_progress(current_location)

        # Check for red light brake
        if self._check_traffic_light_braking(traffic_light_manager):
            acceleration = self.EMERGENCY_DECELERATION
            target_speed = 0.0
        else:
            # Free flow or car following
            # Find nearest red light for IDM target
            dist_to_light = None
            for tl_id, tl_data in traffic_light_manager.traffic_lights.items():
                if tl_data['current_state'] in [TrafficLightState.Red, TrafficLightState.Yellow]:
                     d = tl_data.get('distance', 1000)
                     if 0 < d < 200:
                         if dist_to_light is None or d < dist_to_light:
                             dist_to_light = d
            
            if dist_to_light:
                # IDM with virtual stop hurdle
                acceleration = IDM(0, current_speed, dist_to_light, 0) # Xp=dist, Vp=0
                target_speed = self.DESIRED_VELOCITY
            else:
                # Free flow
                acceleration = IDM(0, current_speed, 1000, self.DESIRED_VELOCITY)
                target_speed = self.DESIRED_VELOCITY

        acceleration = np.clip(acceleration, *self.ACCELERATION_LIMITS)
        self.prev_acceleration = acceleration
        
        # Apply Longitudinal Control
        control = VehicleControl()
        if acceleration > 0.1:
            # Positive acceleration - use throttle (match EcoLead mapping)
            control.throttle = np.clip(acceleration / 3.0, 0.0, 0.8)
            control.brake = 0.0
        elif acceleration < -0.1:
            # Negative acceleration - use brake (match EcoLead mapping)
            control.throttle = 0.0
            control.brake = np.clip(-acceleration / 4.0, 0.0, 0.6)
        else:
            # Small accelerations - coast
            control.throttle = 0.0
            control.brake = 0.0
        
        # Apply Lateral Control using Pure Pursuit
        steering = self.lateral_controller.calculate_steering(
            current_transform,
            self.waypoints,
            self.current_waypoint_index,
            current_speed
        )
        
        # Apply steering rate limiting (match EcoLead)
        if steering > self.past_steering + self.max_steer_change:
            steering = self.past_steering + self.max_steer_change
        elif steering < self.past_steering - self.max_steer_change:
            steering = self.past_steering - self.max_steer_change
        
        # Apply steering limits
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        self.past_steering = steering
        
        control.steer = steering
        
        self.vehicle.apply_control(control)
        
        # Check route completion (match EcoLead logic)
        route_end = False
        remaining_waypoints = len(self.waypoints) - self.current_waypoint_index
        if remaining_waypoints <= 5:  # ROUTE_COMPLETION_THRESHOLD
            route_end = True
        
        return acceleration, route_end, target_speed
