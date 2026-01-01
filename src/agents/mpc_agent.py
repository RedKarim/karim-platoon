import numpy as np
import math
from ..logic.mpc_controller import MPCController
from ..logic.ego_model import EgoModel
from ..core.vehicle import VehicleControl
from .. import utils
# import carla # NO CARLA

def get_waypoints(waypoints_list, N, vehicle_x, vehicle_y, vehicle_psi, current_wp_idx):
    # Logic remains same, processing pure logic
    waypoints = []
    num_waypoints = len(waypoints_list)
    min_distance = float('inf')
    closest_idx = current_wp_idx

    for idx in range(current_wp_idx, num_waypoints):
        wp = waypoints_list[idx]
        dx = wp['x'] - vehicle_x
        dy = wp['y'] - vehicle_y
        distance = np.hypot(dx, dy)
        heading_to_wp = np.arctan2(dy, dx)
        angle = vehicle_psi - heading_to_wp
        angle = np.arctan2(np.sin(angle), np.cos(angle))

        if abs(angle) < np.pi / 2:
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        else:
            continue

    if min_distance == float('inf'):
        closest_idx = current_wp_idx
        # Fallback dummy waypoints relative to vehicle if end reached
        return [(10*i, 0) for i in range(1, N+1)], current_wp_idx

    end_idx = min(closest_idx + N, num_waypoints)
    waypoints_subset = waypoints_list[closest_idx:end_idx]

    waypoints = []
    for wp in waypoints_subset:
        x_global = wp['x']
        y_global = wp['y']
        shift_x = x_global - vehicle_x
        shift_y = y_global - vehicle_y
        x_vehicle = shift_x * np.cos(-vehicle_psi) - shift_y * np.sin(-vehicle_psi)
        y_vehicle = shift_x * np.sin(-vehicle_psi) + shift_y * np.cos(-vehicle_psi)
        if x_vehicle >= 0:
            waypoints.append((x_vehicle, y_vehicle))

    if len(waypoints) < N:
        last_wp = waypoints[-1] if waypoints else (0, 0)
        while len(waypoints) < N:
            waypoints.append(last_wp)

    return waypoints[:N], closest_idx

def map_acceleration_to_throttle_brake(a_desired, a_max, a_min, should_brake=False):
    if should_brake:
        throttle = 0.0
        brake = 1.0
    else:
        if a_desired > 0:
            throttle = np.clip(a_desired / a_max, 0.0, 1)
            brake = 0.0
        elif a_desired < 0:
            throttle = 0.0
            brake = np.clip(-a_desired / abs(a_min), 0.0,1)
        else:
            throttle = 0.0
            brake = 0.0
    return throttle, brake

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

class MPCAgent(object):
    def __init__(self, ego_vehicle, config):
        mpc_config = config.get('MPCController', {})
        weights = mpc_config.get('weights', {})
        fuel_coeffs = mpc_config.get('fuel_consumption', {})

        self.N = mpc_config.get('N', 10)
        self.dt = mpc_config.get('dt', 0.1)
        self.ref_v = mpc_config.get('ref_v',0)
        self.a_max = mpc_config.get('a_max', 3.0)
        steer_max_deg = mpc_config.get('steer_max_deg', 25)
        self.steer_max = np.deg2rad(steer_max_deg)
        self.v_max = mpc_config.get('v_max', 14.0)
        self.previous_velocity = 0
        
        # In mock, these dict keys might be missing if config is minimal
        self.weight_cte = weights.get('cte', 1.0)
        
        self.prev_delta = 0.0
        self.prev_a = 0  
        self.vehicle = ego_vehicle
        
        # Steering smoothing (to reduce noise in kinematic model)
        self.past_steering = 0.0
        self.max_steer_change = 0.05  # Max steering change per timestep
        
        self.ego_model = EgoModel(self.dt)
        self.mpc_controller = MPCController(self.ego_model,mpc_config,weights,fuel_coeffs)
        # Assuming config/route.csv relative to project root
        self.waypoints = utils.read_waypoints_from_csv("config/route.csv")
        self.current_wp_idx = 0

    def on_tick(self,ref_v,stop_location=None):
        """
        MPC control tick - simplified for straight road.
        Steering is always 0, MPC optimizes acceleration only.
        """
        self.ref_v = ref_v
        
        if hasattr(self.vehicle.velocity, 'magnitude'): # Vector3D from primitives
            current_velocity = self.vehicle.velocity.magnitude
        else:
            # Fallback if it's already a float or struct with x,y
            vel = self.vehicle.get_velocity()
            current_velocity = math.sqrt(vel.x**2 + vel.y**2)
            
        vehicle_location = self.vehicle.get_location()
        vehicle_x = vehicle_location.x
        vehicle_y = vehicle_location.y

        # Acceleration calc
        if self.dt > 0:
            calculated_acceleration = (current_velocity - self.previous_velocity) / self.dt
        else:
            calculated_acceleration = 0
        self.previous_velocity = current_velocity

        vehicle_transform = self.vehicle.get_transform()
        vehicle_rotation = vehicle_transform.rotation
        vehicle_psi = math.radians(vehicle_rotation.yaw)
       
        waypoints_coords, self.current_wp_idx = get_waypoints(
            self.waypoints, self.N, vehicle_x, vehicle_y, vehicle_psi, self.current_wp_idx
        )

        if stop_location:
            control = VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
            self.vehicle.apply_control(control)
            return 0, False, self.ref_v

        x_vals = [coord[0] for coord in waypoints_coords]
        y_vals = [coord[1] for coord in waypoints_coords]
        
        if not x_vals: 
             # No waypoints
             return 0, False, self.ref_v

        coeffs = np.polyfit(x_vals, y_vals, 3)
        coeffs = coeffs[::-1] 

        # For straight road, CTE and EPSI should be ~0
        cte = 0.0  # No cross-track error on straight road
        epsi = 0.0  # No heading error on straight road

        pred_x = current_velocity * self.dt
        pred_y = 0
        pred_psi = 0.0  # No steering on straight road
        pred_v = current_velocity + calculated_acceleration * self.dt
        pred_cte = 0.0  # Straight road
        pred_epsi = 0.0  # Straight road

        current_state = np.array([pred_x, pred_y, pred_psi, pred_v, pred_cte, pred_epsi])
        
        result = self.mpc_controller.solve(current_state, coeffs, 0.0, self.prev_a, self.ref_v)
        if result is None:
            # Solver failed
            return 0, False, self.ref_v
            
        optimal_delta, optimal_a, mpc_x, mpc_y = result

        self.prev_delta = 0.0  # Always 0 for straight road
        self.prev_a = optimal_a[0]

        throttle_cmd, brake_cmd = map_acceleration_to_throttle_brake(optimal_a[0], 
                                                                     self.mpc_controller.a_max, 
                                                                     -self.mpc_controller.a_max, 
                                                                     should_brake=False)
        
        # Straight road - no steering needed
        control = VehicleControl()
        control.steer = 0.0  # Always 0 for straight road
        control.throttle = np.clip(throttle_cmd, 0.0, 0.6)
        control.brake = np.clip(brake_cmd, 0.0, 0.28)
        
        self.vehicle.apply_control(control)
        return optimal_a[0], False, self.ref_v
