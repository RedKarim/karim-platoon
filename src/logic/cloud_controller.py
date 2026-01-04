import threading
import time
import json
import numpy as np
import math
import os
from .mqtt_interface import MQTTClientWrapper
from .mpc_controller import MPCController
from .ego_model import EgoModel
from .. import utils
from ..agents.behavior_agent import BehaviorAgent
from ..core.primitives import Transform, Location

# Mock classes for BehaviorAgent compatibility
class MockVelocity:
    def __init__(self, v_ms):
        self.x = v_ms
        self.y = 0

class MockVehicle:
    def __init__(self, id, v_ms=0):
        self.id = id
        self.velocity = MockVelocity(v_ms)
        self.transform = Transform() # Default at 0,0
    
    def get_velocity(self):
        return self.velocity

    def get_location(self):
        return self.transform.location

class CloudController:
    def __init__(self, config, route_path="config/route.csv"):
        self.config = config
        self.running = False
        self.controllers = {} # vehicle_id -> MPCController OR BehaviorAgent
        self.states = {} # vehicle_id -> latest_state
        self.current_wp_indices = {} # vehicle_id -> index
        
        # Load Route
        if not os.path.exists(route_path):
             route_path = os.path.join(os.getcwd(), route_path)
             
        self.waypoints = utils.read_waypoints_from_csv(route_path)
        
        # Initialize MQTT
        self.mqtt_client = MQTTClientWrapper("cloud_controller")
        
        # Configuration for MPC
        self.mpc_config = config.get('MPCController', {})
        self.weights = self.mpc_config.get('weights', {})
        self.fuel_coeffs = self.mpc_config.get('fuel_consumption', {})
        self.dt = self.mpc_config.get('dt', 0.1)
        self.N = self.mpc_config.get('N', 10)
        
    def start(self):
        self.running = True
        self.mqtt_client.connect()
        self.mqtt_client.subscribe("platoon/vehicle/+/state", self._on_vehicle_state)
        print("Cloud Controller Started and Listening...")
        
        # Start processing loop
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.mqtt_client.disconnect()
        
    def _on_vehicle_state(self, topic, payload):
        try:
            if isinstance(payload, str):
                data = json.loads(payload)
            else:
                data = payload
                
            v_id = data.get('id')
            if v_id is not None:
                self.states[v_id] = data
                
        except Exception as e:
            print(f"Error parsing vehicle state: {e}")

    def _get_controller(self, v_id, agent_type='mpc'):
        if v_id not in self.controllers:
            if agent_type == 'behavior_agent':
                # Create Behavior Agent
                mock_vehicle = MockVehicle(v_id)
                agent = BehaviorAgent(mock_vehicle, behavior='normal')
                self.controllers[v_id] = agent
                print(f"Initialized Cloud Behavior Agent for Vehicle {v_id}")
            else:
                # Create MPC Controller
                ego_model = EgoModel(self.dt)
                controller = MPCController(ego_model, self.mpc_config, self.weights, self.fuel_coeffs)
                self.controllers[v_id] = controller
                self.current_wp_indices[v_id] = 0
                print(f"Initialized Cloud MPC for Vehicle {v_id}")
                
        return self.controllers[v_id]

    def _process_loop(self):
        while self.running:
            current_time = time.time()
            vehicle_ids = list(self.states.keys())
            
            for v_id in vehicle_ids:
                state_data = self.states[v_id]
                agent_type = state_data.get('type', 'mpc')
                sim_timestamp = state_data.get('timestamp', 0)
                
                controller = self._get_controller(v_id, agent_type)
                
                # Extract Common State
                v = state_data['v'] # m/s (MPCAgent sends msg magnitude, BehaviorAgent sends value)
                
                control_msg = {
                    'id': v_id,
                    'timestamp_sent_from_vehicle': sim_timestamp,
                    'timestamp_computed': time.time(),
                }
                
                if isinstance(controller, BehaviorAgent):
                    # Process Behavior Agent Logic
                    params = state_data.get('parameters', {})
                    gap = params.get('gap', 100)
                    leader_v = params.get('leader_v', 10)
                    leader_id = params.get('leader_id', -1)
                    speed_limit = params.get('speed_limit', 13.8)
                    
                    # Update Mock Vehicle
                    controller._vehicle.velocity.x = v 
                    controller._speed = v * 3.6 # km/h internal
                    controller._speed_limit = speed_limit
                    
                    # Create Mock Leader
                    leader_mock = MockVehicle(leader_id, leader_v)
                    
                    # Run logic
                    # car_following_manager takes (vehicle, gap)
                    # It returns VehicleControl
                    control = controller.car_following_manager(leader_mock, gap)
                    
                    control_msg['throttle'] = float(control.throttle)
                    control_msg['brake'] = float(control.brake)
                    control_msg['optimal_a'] = [] # No horizon for Behavior Agent
                    
                else:
                    # Process MPC Logic
                    x = state_data['x']
                    y = state_data['y']
                    psi = state_data['psi']
                    ref_v = state_data.get('ref_v', 10.0)
                    
                    # 1. Find Local Waypoints
                    current_idx = self.current_wp_indices.get(v_id, 0)
                    local_waypoints, new_idx = utils.get_waypoints(
                        self.waypoints, self.N, x, y, psi, current_idx
                    )
                    self.current_wp_indices[v_id] = new_idx
                    
                    if not local_waypoints:
                        continue
                        
                    # 2. Fit Polynomial
                    x_vals = [p[0] for p in local_waypoints]
                    y_vals = [p[1] for p in local_waypoints]
                    
                    if not x_vals:
                        continue
    
                    coeffs = np.polyfit(x_vals, y_vals, 3)
                    coeffs = coeffs[::-1]
                    
                    # 3. Setup State (Prediction)
                    if not hasattr(controller, 'prev_v'):
                        controller.prev_v = v
                    
                    accel = (v - controller.prev_v) / self.dt
                    controller.prev_v = v
                    
                    pred_x = v * self.dt
                    pred_y = 0
                    pred_psi = 0
                    pred_v = v + accel * self.dt
                    pred_cte = 0
                    pred_epsi = 0
                    
                    current_state = np.array([pred_x, pred_y, pred_psi, pred_v, pred_cte, pred_epsi])
    
                    # 4. Solve
                    prev_a = getattr(controller, 'last_optimal_a', 0.0)
                    result = controller.solve(current_state, coeffs, 0.0, prev_a, ref_v)
                    
                    if result:
                        optimal_delta, optimal_a, mpc_x, mpc_y = result
                        
                        # Store for next step
                        controller.last_optimal_a = optimal_a[0]
                        
                        # Pack Control Msg
                        control_msg['optimal_a'] = optimal_a.tolist()
                        control_msg['optimal_delta'] = float(optimal_delta)
                        
                        th, br = utils.map_acceleration_to_throttle_brake(
                            optimal_a[0], controller.a_max, -controller.a_max
                        )
                        control_msg['throttle'] = float(th)
                        control_msg['brake'] = float(br)
                
                # Publish
                self.mqtt_client.publish(f"platoon/vehicle/{v_id}/control", control_msg)
            
            time.sleep(0.01)
