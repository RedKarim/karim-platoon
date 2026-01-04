import numpy as np
import math
import time
import json
from ..logic.mpc_controller import MPCController
from ..logic.ego_model import EgoModel
from ..core.vehicle import VehicleControl
from .. import utils
# import carla # NO CARLA

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

        # Remote Mode / Latency Mitigation
        self.remote_mode = False
        self.mqtt_client = None
        self.mitigate_latency = False
        self.control_buffer = {} # Stores latest control message

        # Metrics for logging
        self.current_latency = 0.0
        self.packet_loss_count = 0 
        self.last_control_timestamp = 0
        self.control_recvd_count = 0
        self.control_sent_count = 0

    def set_remote_mode(self, enabled, mqtt_client, mitigate_latency=False):
        self.remote_mode = enabled
        self.mqtt_client = mqtt_client
        self.mitigate_latency = mitigate_latency
        
        if self.remote_mode and self.mqtt_client:
            topic = f"platoon/vehicle/{self.vehicle.id}/control"
            self.mqtt_client.subscribe(topic, self._on_control_message)

    def _on_control_message(self, topic, payload):
        try:
            if isinstance(payload, str):
                data = json.loads(payload)
            else:
                data = payload
            self.control_buffer = data
            self.control_recvd_count += 1
        except Exception as e:
            print(f"Error parsing control msg: {e}")

    def on_tick(self,ref_v,stop_location=None):
        """
        MPC control tick.
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

        # ---------------------------------------------------------
        # REMOTE CONTROL LOGIC
        # ---------------------------------------------------------
        if self.remote_mode and self.mqtt_client:
            timestamp_now = time.time()
            
            # 1. Publish State
            state_msg = {
                'id': self.vehicle.id,
                'timestamp': timestamp_now,
                'x': vehicle_x,
                'y': vehicle_y,
                'v': current_velocity,
                'psi': vehicle_psi,
                'ref_v': self.ref_v
            }
            self.mqtt_client.publish(f"platoon/vehicle/{self.vehicle.id}/state", state_msg)
            self.control_sent_count += 1
            
            # 2. Receive Control (Non-blocking check of buffer)
            # In a real sync simulation, we might wait. Here we use whatever is latest.
            if not self.control_buffer:
                # No control yet, stay idle or keep prev?
                return 0, False, self.ref_v
                
            control_data = self.control_buffer
            
            # Calculate metrics
            ts_sent = control_data.get('timestamp_sent_from_vehicle', 0)
            ts_computed = control_data.get('timestamp_computed', 0)
            # RTT = Now - Sent
            # Note: time.time() is wall clock. Simulation might run slower/faster?
            # User request specifically asks to "measure round trip latency".
            rtt = timestamp_now - ts_sent
            self.current_latency = rtt
            
            optimal_a_array = control_data.get('optimal_a', [])
            
            # 3. Apply Control with Mitigation?
            target_accel = 0.0
            
            if self.mitigate_latency and optimal_a_array:
                # Index shifting
                # If latency is 0.2s and dt is 0.1s, we should use index 2
                k = int(max(0, rtt / self.dt))
                if k < len(optimal_a_array):
                    target_accel = optimal_a_array[k]
                else:
                    target_accel = optimal_a_array[-1] # Extrapolate/Hold
            elif optimal_a_array:
                # Standard: use first element (k=0)
                target_accel = optimal_a_array[0]
                
            # Convert to Throttle/Brake
            throttle_cmd, brake_cmd = utils.map_acceleration_to_throttle_brake(
                target_accel, self.a_max, -self.a_max, should_brake=False
            )
            
            control = VehicleControl()
            control.steer = 0.0
            control.throttle = np.clip(throttle_cmd, 0.0, 0.6)
            control.brake = np.clip(brake_cmd, 0.0, 0.28)
            self.vehicle.apply_control(control)
            
            return target_accel, False, self.ref_v
            
        # ---------------------------------------------------------
        # LOCAL CONTROL LOGIC
        # ---------------------------------------------------------
        waypoints_coords, self.current_wp_idx = utils.get_waypoints(
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
        # For simplicity in local mode we assume ideal track
        
        pred_x = current_velocity * self.dt
        pred_y = 0
        pred_psi = 0.0 
        pred_v = current_velocity + calculated_acceleration * self.dt
        pred_cte = 0.0  
        pred_epsi = 0.0  

        current_state = np.array([pred_x, pred_y, pred_psi, pred_v, pred_cte, pred_epsi])
        
        result = self.mpc_controller.solve(current_state, coeffs, 0.0, self.prev_a, self.ref_v)
        if result is None:
            # Solver failed
            return 0, False, self.ref_v
            
        optimal_delta, optimal_a, mpc_x, mpc_y = result

        self.prev_delta = 0.0  
        self.prev_a = optimal_a[0]

        throttle_cmd, brake_cmd = utils.map_acceleration_to_throttle_brake(optimal_a[0], 
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
