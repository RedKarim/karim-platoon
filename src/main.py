import argparse
import time
import yaml
import os
import sys

# Add project root to path for imports to work
# Add project root to path for imports to work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.core.world import Client, Map, Waypoint
from src.core.primitives import Transform, Location, Rotation
from src.core.traffic_light import TrafficLight, TrafficLightState

from src.logic.traffic_light_manager import TrafficLightManager
from src.logic.traffic_manager import VehicleTrafficManager
from src.agents.mpc_agent import MPCAgent
from src.agents.idm_agent import IDMAgent
from src import utils

def main():
    parser = argparse.ArgumentParser(description="Karim-Platoon Simulator")
    parser.add_argument('--scenario', default='packleader', help='Scenario to run: packleader, following, idm_packleader')
    parser.add_argument('--config', default='config/config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), '..', args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load Waypoints
    route_path = os.path.join(os.path.dirname(__file__), '..', 'config/route.csv')
    waypoint_transforms = utils.read_csv_waypoints(route_path)
    # Convert to pure Locations for managers
    waypoint_locations = [t.location for t in waypoint_transforms]
    
    # Init Simulation
    client = Client()
    world = client.get_world()
    
    # Verify Map
    map_obj = Map(waypoints=[Waypoint(t) for t in waypoint_transforms])
    world.set_map(map_obj)
    
    # Spawn Traffic Lights (Mock)
    # In reality we'd parse this from somewhere or config. 
    # For now, spawn one traffic light at a known waypoint index
    tl_loc = waypoint_locations[min(len(waypoint_locations)-10, 100)] # 100th waypoint
    traffic = world.spawn_actor('traffic_light', Transform(tl_loc))
    # Configure it
    traffic_lights_config = {
        traffic.id: {
            'initial_state': TrafficLightState.Green,
            'green_time': 100.0, # Long green for testing
            'red_time': 5.0
        }
    }

    # Traffic Light Manager
    tl_manager = TrafficLightManager(client, traffic_lights_config, waypoint_locations)
    
    # Spawn Ego Vehicle
    ego_spawn = waypoint_transforms[0]
    ego_vehicle = world.spawn_actor('vehicle.mustang', ego_spawn)
    print(f"Spawned Ego Vehicle with ID: {ego_vehicle.id}")
    
    # Agent Init
    if args.scenario in ['packleader', 'following']:
        agent = MPCAgent(ego_vehicle, config)
        print("Initialized MPC Agent")
    else:
        agent = IDMAgent(ego_vehicle, waypoint_transforms, config)
        print("Initialized IDM Agent")

    # Traffic Manager
    # behaviour arg is usually a string path or enum
    traffic_manager = VehicleTrafficManager(
        client, world, waypoint_transforms, 
        scenario=args.scenario,
        behaviour='normal',
        ego_vehicle=ego_vehicle,
        traffic_light_manager=tl_manager,
        num_behind=3 # 3 followers
    )
    
    # Spawn Scenario
    traffic_manager.spawn_scenario()
    
    # Loop
    try:
        while True:
            # Tick logic
            world.tick()
            current_tick = world.frame
            
            # Update Traffic Lights
            ego_loc = ego_vehicle.get_location()
            tl_manager.update_traffic_lights(ego_loc, current_tick)
            
            # Agent Control
            if args.scenario == 'idm_packleader':
                traffic_manager.update_idm_pack(agent, current_tick)
            else: # MPC scenario
                traffic_manager.update_pack(agent, current_tick)
                
            # Log
            vel = ego_vehicle.get_velocity()
            speed = (vel.x**2 + vel.y**2)**0.5
            print(f"Tick: {current_tick} | Frame Time: {world.time:.2f} | Ego Speed: {speed:.2f} m/s")
            
            time.sleep(0.1)
            
            if current_tick > 200: # Short run for demo
                break
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        traffic_manager.cleanup()
        tl_manager.stop()

if __name__ == '__main__':
    main()
