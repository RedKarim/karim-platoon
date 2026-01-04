import time
import yaml
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import threading

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
from src.logic.mqtt_interface import MQTTClientWrapper
from src.logic.cloud_controller import CloudController

def main():
    # Check command line arguments
    # Usage: python3 main.py [MPC|IDM] [MQTT|MQTTMITIGATE]
    
    mode = 'MPC'
    use_mqtt = False
    mitigate_latency = False
    
    if len(sys.argv) >= 2:
        mode = sys.argv[1].upper()
        
    if len(sys.argv) >= 3:
        option = sys.argv[2].upper()
        if option == 'MQTT':
            use_mqtt = True
        elif option == 'MQTTMITIGATE':
            use_mqtt = True
            mitigate_latency = True
            
    if mode not in ['MPC', 'IDM']:
        print("Error: Mode must be either 'MPC' or 'IDM'")
        sys.exit(1)
        
    print(f"Running simulation in {mode} mode")
    if use_mqtt:
        print(f"MQTT Enabled. Latency Mitigation: {mitigate_latency}")
        
    # Setup MQTT logic
    cloud_controller = None
    mqtt_clients = []
    
    def mqtt_client_factory(v_id):
        client = MQTTClientWrapper(f"vehicle_{v_id}")
        mqtt_clients.append(client)
        return client

    if use_mqtt:
        # Load config for CloudController
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config/config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        cloud_controller = CloudController(config)
        cloud_controller.start()
        # Give it a moment to connect
        time.sleep(1)

    # Load Config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config/config.yaml')
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
    
    # Traffic Light Setup - 6 signals along 1500m straight road
    traffic_light_config = {
        1: {'initial_state': TrafficLightState.Red, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 666},    # 200m
        2: {'initial_state': TrafficLightState.Green, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 1333}, # 400m
        3: {'initial_state': TrafficLightState.Red, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 2000},   # 600m
        4: {'initial_state': TrafficLightState.Green, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 3000}, # 900m
        5: {'initial_state': TrafficLightState.Red, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 3666},   # 1100m
        6: {'initial_state': TrafficLightState.Green, 'green_time': 20.0, 'red_time': 20.0, 'location_index': 4333}, # 1300m
    }
    
    traffic_lights_dict = {}
    waypoints = [Waypoint(t) for t in waypoint_transforms]
    for tl_id, tl_config in traffic_light_config.items():
        if tl_config['location_index'] < len(waypoints):
            loc = waypoints[tl_config['location_index']].transform.location
            tl = world.spawn_actor(f"traffic_light.{tl_id}", Transform(loc, Rotation(0,0,0)))
            tl.id = tl_id
            traffic_lights_dict[tl_id] = tl

    # Initialize Traffic Light Manager
    tl_manager = TrafficLightManager(client, traffic_light_config, waypoints)
    
    # Spawn Ego Vehicle
    # Calculate required distance for followers: 7 followers * 7.5m spacing + buffer
    num_followers = 7
    spacing = 7.5
    required_distance = num_followers * spacing
    
    ego_spawn_index = 0
    for i, wp in enumerate(waypoint_transforms):
        if wp.location.x >= required_distance:
            ego_spawn_index = i
            break
            
    if ego_spawn_index >= len(waypoint_transforms):
        ego_spawn_index = len(waypoint_transforms) - 1
        print("Warning: Route too short for full platoon spacing from 0")

    ego_spawn = waypoint_transforms[ego_spawn_index]
    ego_vehicle = world.spawn_actor('vehicle.mustang', ego_spawn)
    print(f"Spawned Ego Vehicle with ID: {ego_vehicle.id} at Index: {ego_spawn_index} (x={ego_spawn.location.x:.2f}m)")
    
    # Set scenario and agent based on mode (matching EcoLead)
    if mode == 'MPC':
        scenario = 'packleader'  # MPC mode: full platoon scenario
        ego_vehicle_controller = 'mpc'
        behavior = 'normal'
        agent = MPCAgent(ego_vehicle, config)
        print("Initialized MPC Agent")
        
        if use_mqtt:
            mqtt_client = mqtt_client_factory(ego_vehicle.id)
            mqtt_client.connect()
            agent.set_remote_mode(True, mqtt_client, mitigate_latency=mitigate_latency)
            mqtt_clients.append(mqtt_client) # Track for cleanup
            
    else:  # IDM mode
        scenario = 'idm_packleader'  # IDM mode: ego vehicle with followers
        ego_vehicle_controller = 'idm'
        behavior = 'normal'
        agent = IDMAgent(ego_vehicle, waypoint_transforms, config)
        print("Initialized IDM Agent")
        # NOTE: IDMAgent remote mode not implemented in this scope as request focused on MPC mainly,
        # but logic supports it if we updated IDMAgent. For now assume IDM is local if mode=IDM.

    # Traffic Manager (matching EcoLead configuration)
    traffic_manager = VehicleTrafficManager(
        client, world, waypoint_transforms, 
        scenario=scenario,
        behaviour=behavior,
        ego_vehicle=ego_vehicle,
        traffic_light_manager=tl_manager,
        num_behind=7,  # 7 followers (match EcoLead)
        ego_spawn_index=ego_spawn_index,
        mqtt_client_factory=mqtt_client_factory if use_mqtt else None # Pass factory
    )
    
    # Visualization
    visualizer = Visualizer(waypoint_locations)
    
    # Spawn Scenario
    traffic_manager.spawn_scenario()
    
    # Initialize data collection for trajectory plot AND CSV logging
    trajectory_data = [] # List of dicts
    
    # Loop
    try:
        while True:
            # Tick logic
            world.tick()
            current_tick = world.frame
            
            # Update Traffic Lights
            ego_loc = ego_vehicle.get_location()
            tl_manager.update_traffic_lights(ego_loc, current_tick)
            
            # Agent Control (matching EcoLead's mode-based logic)
            if mode == 'IDM':
                # IDM mode with platoon
                traffic_manager.update_idm_pack(agent, current_tick)
            else:
                # MPC mode with platoon
                traffic_manager.update_pack(agent, current_tick)
                
            # Log Data
            # "save the data of all the cars in a csv file lke how the Ecolead project does"
            # We will gather data for ALL cars (Ego + Followers) into one list/DataFrame.
            
            # Gather Ego Stats
            vel = ego_vehicle.get_velocity()
            speed = (vel.x**2 + vel.y**2)**0.5  # m/s
            ego_pos = ego_vehicle.get_location()
            
            ego_latency = getattr(agent, 'current_latency', 0.0)
            
            # Collect trajectory data
            # Store ego vehicle data
            ego_entry = {
                'Timestamp': world.time,
                'ID': ego_vehicle.id,
                'Role': 'Leader',
                'Position_X': ego_pos.x,
                'Position_Y': ego_pos.y,
                'Velocity': speed,
                'Latency': ego_latency,
            }
            trajectory_data.append(ego_entry)
            
            # Collect follower vehicle data
            for pm in traffic_manager.platoon_managers:
                # Iterate through vehicles in order to check gaps
                # behind_vehicles is ordered list? Yes.
                if not pm.behind_vehicles:
                    continue

                prev_veh = pm.behind_vehicles[0]['following'] # This is leader for 1st follower
                
                for i, vehicle_info in enumerate(pm.behind_vehicles):
                    v_id = vehicle_info['id']
                    v = vehicle_info['vehicle']
                    v_agent = vehicle_info['agent']
                    
                    v_vel = v.get_velocity()
                    v_speed = math.sqrt(v_vel.x**2 + v_vel.y**2)
                    v_pos = v.get_location()
                    
                    # Gap check
                    if prev_veh:
                        prev_pos = prev_veh.get_location()
                        gap = prev_pos.distance(v_pos)
                        if gap < 2.0:
                            print(f"!!! CRITICAL WARNING: Gap between {prev_veh.id} and {v_id} is {gap:.2f}m (< 2.0m) !!!")
                    
                    prev_veh = v # Update for next
                    
                    v_latency = getattr(v_agent, 'current_latency', 0.0)
                    
                    entry = {
                        'Timestamp': world.time,
                        'ID': v_id,
                        'Role': 'Follower',
                        'Position_X': v_pos.x,
                        'Position_Y': v_pos.y,
                        'Velocity': v_speed,
                        'Latency': v_latency
                    }
                    trajectory_data.append(entry)
            
            print(f"Tick: {current_tick} | Time: {world.time:.2f} | Payload: {len(trajectory_data)} rows")
            
            # Visualization Update
            all_traffic = []
            for pm in traffic_manager.platoon_managers:
                all_traffic.extend(pm.behind_vehicles)
            
            leader_ids = traffic_manager.get_leader_ids()
            visualizer.update(
                ego_vehicle, 
                all_traffic, 
                tl_manager.get_traffic_lights(),
                {'tick': current_tick, 'time': world.time, 'speed': speed},
                leader_ids=leader_ids
            )

            # time.sleep(0.1) # Controlled by plt.pause in update
            
            # Check termination condition: Last vehicle must pass 1500m
            all_x_positions = [ego_pos.x]
            for pm in traffic_manager.platoon_managers:
                 for v_data in pm.behind_vehicles:
                     all_x_positions.append(v_data['vehicle'].get_location().x)
            
            min_x = min(all_x_positions)
            
            # Timeout safety (e.g. 600s / 6000 ticks) to prevent infinite loop if stuck
            if min_x > 1500.0 or current_tick > 6000:
                print(f"Simulation ended. All vehicles passed 1500m (Min X: {min_x:.2f}) or Timeout.")
                break
                
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
    finally:
        print("\nStopping Simulation...")
        if cloud_controller:
            cloud_controller.stop()
        
        for c in mqtt_clients:
            c.disconnect()
            
        traffic_manager.cleanup()
        tl_manager.stop()
        visualizer.close()

        print("\nSaving Data...")
        if trajectory_data:
            df = pd.DataFrame(trajectory_data)
            
            # Save Raw Data
            csv_filename = f"platoon_data_{mode}_{'MQTT' if use_mqtt else 'LOCAL'}.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
            
            # Save Latency Data specifically if requested
            if use_mqtt:
                latency_filename = f"latency_data_{'MITIGATION' if mitigate_latency else 'NORMAL'}.csv"
                df_latency = df[['Timestamp', 'ID', 'Latency']]
                df_latency.to_csv(latency_filename, index=False)
                print(f"Latency data saved to {latency_filename}")
                
            # Plotting Trajectory (Existing Logic adapted)
            print("Generating trajectory plot...")
            plt.figure(figsize=(14, 10))
            
            # Plot Ego
            ego_df = df[df['ID'] == ego_vehicle.id]
            if not ego_df.empty:
                plt.plot(ego_df['Timestamp'], ego_df['Position_X'], label='Leader (Ego)', linewidth=2.5)
                
            # Plot Followers
            followers = df[df['Role'] == 'Follower']['ID'].unique()
            for fid in followers:
                f_df = df[df['ID'] == fid]
                plt.plot(f_df['Timestamp'], f_df['Position_X'], label=f'Follower {fid}', alpha=0.7)
                
            plot_title = f'Trajectory - {mode}'
            plot_filename = f'trajectory_plot_{mode}.png'
            
            if use_mqtt:
                if mitigate_latency:
                    plot_title += " (Mitigated)"
                    plot_filename = f'trajectory_plot_{mode}_latency_mitigation.png'
                else:
                    plot_title += " (Latency)"
                    plot_filename = f'trajectory_plot_{mode}_latency.png'
            
            # Plot Traffic Lights
            # Overlay traffic light phases
            # Assuming fixed cycle: Initial -> (Red/Green) -> Toggle ...
            colors = {TrafficLightState.Red: 'red', TrafficLightState.Green: 'green', TrafficLightState.Yellow: 'yellow'}
            
            for tl_id, config in traffic_light_config.items():
                if config['location_index'] < len(waypoint_transforms):
                    tl_pos = waypoint_transforms[config['location_index']].location.x
                    
                    # specific to this config structure
                    cycle_time = config['green_time'] + config['red_time']
                    current_t = 0
                    state = config['initial_state']
                    
                    # Decide color intervals
                    # To allow for arbitrary simulation length, we iterate up to world.time
                    max_time = world.time
                    t = 0
                    while t < max_time:
                        duration = config['green_time'] if state == TrafficLightState.Green else config['red_time']
                        color = colors.get(state, 'gray')
                        
                        # Plot horizontal line segment for this phase
                        if color == 'red':
                            plt.hlines(y=tl_pos, xmin=t, xmax=min(t+duration, max_time), colors=color, linestyles='solid', linewidth=2, alpha=0.5)
                        
                        # Update for next segment
                        t += duration
                        state = TrafficLightState.Red if state == TrafficLightState.Green else TrafficLightState.Green
            
            plt.xlabel('Time (s)')
            plt.ylabel('Position X (m)')
            plt.title(plot_title)
            plt.xlim(0, 150) # Force Time 0 to 150s as requested
            plt.ylim(bottom=0) # Force Position 0
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
            
            # Plot Latency if MQTT
            if use_mqtt:
                plt.figure(figsize=(10, 6))
                if not ego_df.empty:
                    plt.plot(ego_df['Timestamp'], ego_df['Latency']*1000, label='Leader Latency') # ms
                plt.xlabel('Time (s)')
                plt.ylabel('Latency (ms)')
                plt.title('Control Loop Latency')
                plt.grid(True)
                plt.savefig('latency_plot.png')
                print("Latency plot saved.")

if __name__ == '__main__':
    main()
