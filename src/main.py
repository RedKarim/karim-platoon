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
from src.core.visualizer import Visualizer
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
    # Traffic Light Setup
    # EcoLead uses IDs 13, 11, 20. We will place them at approximate equidistant locations on the loop.
    # Total waypoints ~5000 (roughly based on indices in previous outputs).
    # TL 13: Index 1200
    # TL 11: Index 2500
    # TL 20: Index 3800
    traffic_light_config = {
        13: {'initial_state': TrafficLightState.Red, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 1200},
        11: {'initial_state': TrafficLightState.Green, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 2500},
        20: {'initial_state': TrafficLightState.Red, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 3800}
    }
    
    traffic_lights_dict = {}
    waypoints = [Waypoint(t) for t in waypoint_transforms] # Re-create waypoints list for clarity
    for tl_id, config in traffic_light_config.items():
        if config['location_index'] < len(waypoints):
            loc = waypoints[config['location_index']].transform.location
            tl = world.spawn_actor(f"traffic_light.{tl_id}", Transform(loc, Rotation(0,0,0)))
            tl.id = tl_id # Force ID
            traffic_lights_dict[tl_id] = tl
        else:
            print(f"Warning: Waypoint index {config['location_index']} out of range for TL {tl_id}")

    # Initialize Traffic Light Manager with our custom map of lights
    # We need to pass the config that allows looking up these lights
    tl_manager = TrafficLightManager(client, traffic_light_config, waypoints)
    # The manager expects to find lights in world.get_actors(), so we spawned them above.
    
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
    
    # Visualization
    visualizer = Visualizer(waypoint_locations)
    
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
            
            # Visualization Update
            # Collect all traffic vehicles from all platoon managers
            all_traffic = []
            for pm in traffic_manager.platoon_managers:
                all_traffic.extend(pm.behind_vehicles)
            
            visualizer.update(
                ego_vehicle, 
                all_traffic, 
                tl_manager.get_traffic_lights(),
                {'tick': current_tick, 'time': world.time, 'speed': speed}
            )

            # time.sleep(0.1) # Controlled by plt.pause in update
            
            if current_tick > 1000: # Longer run
                break
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        traffic_manager.cleanup()
        tl_manager.stop()
        visualizer.close()

if __name__ == '__main__':
    main()
