import math
import numpy as np
from ..logic.idm import IDM
from ..core.vehicle import VehicleControl
from .. import utils
from ..core.traffic_light import TrafficLightState

class IDMAgent:
    def __init__(self, vehicle, waypoints, config={}):
        self.vehicle = vehicle
        # waypoints expected to be Transform objects or dicts. 
        # If passed from main, likely transforms. 
        self.waypoints = waypoints 
        self.prev_acceleration = 0.0
        self.current_waypoint_index = 0
        
        # Parameters
        self.DESIRED_VELOCITY = 17
        self.ACCELERATION_LIMITS = (-5.0, 4.0)
        self.TRAFFIC_LIGHT_MIN_DISTANCE = 5.0
        self.EMERGENCY_DECELERATION = -4.0
        
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

    def _check_traffic_light_braking(self, traffic_light_manager):
        current_location = self.vehicle.get_location()
        # Iterate over traffic lights (mocked logic)
        for tl_id, tl_data in traffic_light_manager.traffic_lights.items():
            state = tl_data['current_state']
            if state in [TrafficLightState.Red, TrafficLightState.Yellow]:
                # Assuming traffic light manager updates 'distance' in its update loop
                dist = tl_data.get('distance', float('inf'))
                # Or recompute direct distance
                tl_loc = tl_data['actor'].transform.location
                direct_dist = current_location.distance(tl_loc)
                
                if direct_dist < self.TRAFFIC_LIGHT_MIN_DISTANCE and direct_dist > -2:
                     return True
        return False

    def on_tick(self, traffic_light_manager):
        v = self.vehicle.get_velocity()
        current_speed = math.sqrt(v.x**2 + v.y**2)
        current_location = self.vehicle.get_location()

        # Check for red light brake
        if self._check_traffic_light_braking(traffic_light_manager):
            acceleration = self.EMERGENCY_DECELERATION
            target_speed = 0.0
        else:
            # Free flow or car following
            # Find nearest red light for IDM target?
            # Simplified: Infinite distance if no car ahead (TrafficManager logic handling followers needed)
            # This agent seems to focus on ego interaction with lights in this snippet
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
        
        # Apply Control
        control = VehicleControl()
        if acceleration > 0:
            control.throttle = np.clip(acceleration/3.0, 0, 1)
        else:
            control.brake = np.clip(-acceleration/4.0, 0, 1)
        
        # Lateral Control (Simple Mock)
        # In a full port we'd implement pure pursuit here
        control.steer = 0.0 
        
        self.vehicle.apply_control(control)
        
        # Check route completion (Mock)
        route_end = False
        
        return acceleration, route_end, target_speed
