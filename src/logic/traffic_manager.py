import csv
import random
import math
from datetime import datetime
from .platoon_manager import PlatoonManager
from ..agents.behavior_agent import BehaviorAgent

class VehicleTrafficManager:
    def __init__(self, client, world, waypoints, scenario, behaviour, ego_vehicle, traffic_light_manager, num_behind, num_vehicles=20, spacing=7.5, front_vehicle_autopilot=False):
        self.client = client
        self.world = world
        self.num_vehicles = num_vehicles
        self.waypoints = waypoints
        self.traffic_vehicles = []
        self.front_vehicle = None
        self.behind_vehicles = []
        # self.traffic_manager = self.client.get_trafficmanager(8000) # Removed
        self.vehicle_bps = ['vehicle.mini'] # simplified
        self.csv_files = {}
        self.map = None # world.get_map() # Mock map not fully implemented yet
        self.behaviour = behaviour
        self.spacing = spacing
        self.scenario = scenario
        self.autopilot = front_vehicle_autopilot
        self.ego_vehicle = ego_vehicle
        self.num_behind = num_behind
        self.front_vehicle_speed = 0.0
        self.front_vehicle_transform = None
        self.platoon_managers = []  
        self.traffic_light_manager = traffic_light_manager
        # self.setup_traffic_manager()
        self.start_time = None

    def setup_traffic_manager(self):
        pass

    def configure_vehicle(self, vehicle, autopilot=True):
        pass

    def get_front_vehicle_status(self, ego_vehicle):
        if self.front_vehicle:
            front_vehicle_location = self.front_vehicle.get_location()
            front_vehicle_velocity = self.front_vehicle.get_velocity()
            speed = math.sqrt(front_vehicle_velocity.x**2 + front_vehicle_velocity.y**2 + front_vehicle_velocity.z**2)
            distance = ego_vehicle.get_location().distance(front_vehicle_location)
            return {
                'location': front_vehicle_location,
                'speed': speed,
                'distance': distance
            }
        return None
    
    def spawn_scenario(self):
        if self.scenario == 'packleader':
            self.spawn_pack()
            vehicles_list = [self.ego_vehicle] + [v['vehicle'] for v in self.behind_vehicles]
            platoon_manager = PlatoonManager(vehicles=vehicles_list, leader_id=self.ego_vehicle.id, traffic_manager=self, behaviour_agents=self.behind_vehicles,platoon_id=1)
            self.platoon_managers.append(platoon_manager)
            self.traffic_light_manager.set_platoon_manager(platoon_manager)
        elif self.scenario == 'idm_packleader':
            self.spawn_pack()
            vehicles_list = [self.ego_vehicle] + [v['vehicle'] for v in self.behind_vehicles]
            platoon_manager = PlatoonManager(vehicles=vehicles_list, leader_id=self.ego_vehicle.id, traffic_manager=self, behaviour_agents=self.behind_vehicles, platoon_id=1)
            self.platoon_managers.append(platoon_manager)
            self.traffic_light_manager.set_platoon_manager(platoon_manager)

    def spawn_pack(self):
        previous_vehicle = self.ego_vehicle
        # Calculate approximate step size from first few waypoints
        step_dist = 0.3 # Default guess
        if len(self.waypoints) > 1:
            loc0 = self.waypoints[0].location
            loc1 = self.waypoints[1].location
            step_dist = loc0.distance(loc1)
            if step_dist == 0: step_dist = 0.3
            
        indices_per_spacing = int(self.spacing / step_dist) if step_dist > 0 else 25
        
        for i in range(self.num_behind):
            index = len(self.waypoints) - (i+1) * indices_per_spacing
            if index < 0: index += len(self.waypoints) # Wrap
            index = index % len(self.waypoints)
            
            spawn_transform = self.waypoints[index]
            # spawn_transform.location.z += 2 
            
            # Spawn logic using our mock World
            # We assume World.spawn_actor takes (bp, transform)
            vehicle = self.world.spawn_actor('vehicle.mini', spawn_transform)
            
            if vehicle:
                print("Creating Behavior Agents with global plan for behind vehicles.")
                agent = BehaviorAgent(vehicle, behavior=self.behaviour)
                
                # CRITICAL: Set waypoints for steering! Convert waypoints to (waypoint, RoadOption) tuples
                # RoadOption is not used in our implementation, so we use None
                waypoint_plan = [(wp, None) for wp in self.waypoints]
                agent.set_global_plan(waypoint_plan)
                print(f"Set global plan with {len(waypoint_plan)} waypoints for follower agent")
                
                role_name = f'behind_{i+1}'
                
                print(f"Spawned behind vehicle {i+1} at index {index}")
                
                self.behind_vehicles.append({'id':vehicle.id,'vehicle': vehicle, 'agent': agent, 'role_name': role_name, 'following':previous_vehicle})
                previous_vehicle = vehicle
                # self.csv_files[f'velocity_{vehicle.id}'] = ...

    def update_pack(self, ego_agent, current_tick):
        optimal_a = 0
        ref_v = 0
        ref_v_mpc = 0 
        
        for platoon_manager in self.platoon_managers:
            mpc_agent = False
            self.traffic_light_manager.set_platoon_manager(platoon_manager)
            distances = [100]

            if platoon_manager.platoon_id > 1:
                # Split platoon logic
                ego_vehicle = platoon_manager.behind_vehicles[0]['vehicle']
                ego_velocity = ego_vehicle.get_velocity()
                ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
                
                last_vehicle = platoon_manager.behind_vehicles[-1]['vehicle']
                self.traffic_light_manager.update_traffic_lights(last_vehicle.get_location(), current_tick)
                ref_v, platoon_status, tl_id, eta_to_light = self.traffic_light_manager.calculate_reference_velocity() # Dummy needed
            else:
                mpc_agent = True
                last_vehicle = platoon_manager.behind_vehicles[-1]['vehicle'] if platoon_manager.behind_vehicles else self.ego_vehicle
                self.traffic_light_manager.update_traffic_lights(last_vehicle.get_location(), current_tick)
                
                ego_velocity = self.ego_vehicle.get_velocity()
                ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
                
                # Mock return for logic
                ref_v, platoon_status, tl_id, eta_to_light = self.traffic_light_manager.calculate_reference_velocity(ego_speed) # Passed arg
                
            for i, vehicle_data in enumerate(platoon_manager.behind_vehicles):
                vehicle = vehicle_data['vehicle']
                agent = vehicle_data['agent']
                following_vehicle = vehicle_data['following']
                id = vehicle_data['id']
                
                # Distance calc
                dist = following_vehicle.get_location().distance(vehicle.get_location())
                if platoon_manager.leader_id == id:
                     # It's leader
                     pass
                else:
                    distances.append(dist)
                
                following_speed = math.sqrt(following_vehicle.get_velocity().x**2 + following_vehicle.get_velocity().y**2)
                
                # Pass traffic light manager to agent for traffic light awareness
                agent._traffic_light_manager = self.traffic_light_manager
                
                agent._update_information(ref_v * 3.6)
                control = agent.run_step(following_vehicle, dist, 0, following_speed, False)
                vehicle.apply_control(control)

            if mpc_agent:
                # Call Ego Agent
                # Assuming ego_agent is MPCAgent
                optimal_a, route_end, ref_v_mpc = ego_agent.on_tick(ref_v, False)

            platoon_manager.update_platoon(platoon_status, tl_id, ref_v, eta_to_light, distances=distances)
            
            if platoon_status["mode"]=="SPLIT":
                self.platoon_managers.append(platoon_status["sub_platoon"])

        return optimal_a, ref_v, ref_v_mpc

    def update_idm_pack(self, ego_agent, current_tick):
        """Update platoon with IDM ego agent."""
        optimal_a = 0
        ref_v = 0
        ref_v_mpc = 0
        
        for platoon_manager in self.platoon_managers:
            self.traffic_light_manager.set_platoon_manager(platoon_manager)
            distances = [100]

            # Platoon leader (ego vehicle controlled by IDM agent)
            last_vehicle = platoon_manager.behind_vehicles[-1]['vehicle'] if platoon_manager.behind_vehicles else self.ego_vehicle
            self.traffic_light_manager.update_traffic_lights(last_vehicle.get_location(), current_tick)
            
            ego_velocity = self.ego_vehicle.get_velocity()
            ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
            
            ref_v, platoon_status, tl_id, eta_to_light = self.traffic_light_manager.calculate_reference_velocity(ego_speed)
            
            # Update follower vehicles using BehaviorAgent
            for i, vehicle_data in enumerate(platoon_manager.behind_vehicles):
                vehicle = vehicle_data['vehicle']
                agent = vehicle_data['agent']
                following_vehicle = vehicle_data['following']
                id = vehicle_data['id']
                
                # Distance calc
                dist = following_vehicle.get_location().distance(vehicle.get_location())
                if platoon_manager.leader_id != id:
                    distances.append(dist)
                
                following_speed = math.sqrt(following_vehicle.get_velocity().x**2 + following_vehicle.get_velocity().y**2)
                
                # Pass traffic light manager to agent for traffic light awareness
                agent._traffic_light_manager = self.traffic_light_manager
                
                agent._update_information(ref_v * 3.6)
                control = agent.run_step(following_vehicle, dist, 0, following_speed, False)
                vehicle.apply_control(control)

            # Call IDM Ego Agent with traffic_light_manager
            optimal_a, route_end, ref_v_mpc = ego_agent.on_tick(self.traffic_light_manager)

            platoon_manager.update_platoon(platoon_status, tl_id, ref_v, eta_to_light, distances=distances)
            
            if platoon_status["mode"]=="SPLIT":
                self.platoon_managers.append(platoon_status["sub_platoon"])

        return optimal_a, ref_v, ref_v_mpc 
    
    def cleanup(self):
        for pm in self.platoon_managers:
            pm.cleanup()
