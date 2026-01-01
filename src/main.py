import time
import yaml
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math

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
    # Check command line arguments (matching EcoLead)
    if len(sys.argv) != 2:
        print("Usage: python3 main.py [MPC|IDM]")
        print("  MPC: Run with MPC controller and platoon scenario")
        print("  IDM: Run with IDM controller and platoon scenario")
        sys.exit(1)
    
    mode = sys.argv[1].upper()
    if mode not in ['MPC', 'IDM']:
        print("Error: Mode must be either 'MPC' or 'IDM'")
        sys.exit(1)
    
    print(f"Running simulation in {mode} mode")

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
        1: {'initial_state': TrafficLightState.Red, 'green_time': 25.0, 'red_time': 20.0, 'location_index': 666},    # 200m
        2: {'initial_state': TrafficLightState.Green, 'green_time': 25.0, 'red_time': 20.0, 'location_index': 1333}, # 400m
        3: {'initial_state': TrafficLightState.Red, 'green_time': 25.0, 'red_time': 20.0, 'location_index': 2000},   # 600m
        4: {'initial_state': TrafficLightState.Green, 'green_time': 25.0, 'red_time': 20.0, 'location_index': 3000}, # 900m
        5: {'initial_state': TrafficLightState.Red, 'green_time': 25.0, 'red_time': 20.0, 'location_index': 3666},   # 1100m
        6: {'initial_state': TrafficLightState.Green, 'green_time': 25.0, 'red_time': 20.0, 'location_index': 4333}, # 1300m
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
    ego_spawn_index = 400
    if len(waypoint_transforms) > 400:
        ego_spawn = waypoint_transforms[ego_spawn_index]
    else:
        ego_spawn = waypoint_transforms[0]
    ego_vehicle = world.spawn_actor('vehicle.mustang', ego_spawn)
    print(f"Spawned Ego Vehicle with ID: {ego_vehicle.id}")
    
    # Set scenario and agent based on mode (matching EcoLead)
    if mode == 'MPC':
        scenario = 'packleader'  # MPC mode: full platoon scenario
        ego_vehicle_controller = 'mpc'
        behavior = 'normal'
        agent = MPCAgent(ego_vehicle, config)
        print("Initialized MPC Agent")
    else:  # IDM mode
        scenario = 'idm_packleader'  # IDM mode: ego vehicle with followers
        ego_vehicle_controller = 'idm'
        behavior = 'normal'
        agent = IDMAgent(ego_vehicle, waypoint_transforms, config)
        print("Initialized IDM Agent")

    # Traffic Manager (matching EcoLead configuration)
    traffic_manager = VehicleTrafficManager(
        client, world, waypoint_transforms, 
        scenario=scenario,
        behaviour=behavior,
        ego_vehicle=ego_vehicle,
        traffic_light_manager=tl_manager,
        num_behind=7  # 7 followers (match EcoLead)
    )
    
    # Visualization
    visualizer = Visualizer(waypoint_locations)
    
    # Spawn Scenario
    traffic_manager.spawn_scenario()
    
    # Initialize data collection for trajectory plot
    trajectory_data = []
    vehicle_data = {}  # Track data for each follower vehicle
    
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
                
            # Log
            vel = ego_vehicle.get_velocity()
            speed = (vel.x**2 + vel.y**2)**0.5  # m/s
            ego_pos = ego_vehicle.get_location()
            
            # Collect trajectory data
            tl_states = {}
            for tl_id, tl_data in tl_manager.get_traffic_lights().items():
                state = tl_data['current_state']
                tl_states[tl_id] = 'Red' if state in [TrafficLightState.Red, TrafficLightState.Yellow] else 'Green'
            
            # Store ego vehicle data
            ego_data = {
                'Timestamp': world.time,
                'Velocity': speed,
                'Position_X': ego_pos.x,
                'Position_Y': ego_pos.y,
                'Light State 1': tl_states.get(1, 'Unknown'),
                'Light State 2': tl_states.get(2, 'Unknown'),
                'Light State 3': tl_states.get(3, 'Unknown'),
                'Light State 4': tl_states.get(4, 'Unknown'),
                'Light State 5': tl_states.get(5, 'Unknown'),
                'Light State 6': tl_states.get(6, 'Unknown')
            }
            trajectory_data.append(ego_data)
            
            # Collect follower vehicle data
            for pm in traffic_manager.platoon_managers:
                for vehicle_info in pm.behind_vehicles:
                    v_id = vehicle_info['id']
                    v = vehicle_info['vehicle']
                    v_vel = v.get_velocity()
                    v_speed = math.sqrt(v_vel.x**2 + v_vel.y**2)
                    v_pos = v.get_location()
                    
                    if v_id not in vehicle_data:
                        vehicle_data[v_id] = []
                    
                    vehicle_data[v_id].append({
                        'Time': world.time,
                        'Velocity': v_speed,
                        'Position_X': v_pos.x,
                        'Position_Y': v_pos.y
                    })
            
            print(f"Tick: {current_tick} | Frame Time: {world.time:.2f} | Ego Speed: {speed:.2f} m/s")
            
            # Visualization Update
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
        print("\nGenerating trajectory plot...")
        
        # Process data and create plot
        if trajectory_data:
            # Create DataFrame for ego vehicle
            df_ego = pd.DataFrame(trajectory_data)
            
            # Calculate cumulative distance for ego vehicle
            df_ego['Time_diff'] = df_ego['Timestamp'].diff().fillna(0)
            df_ego['Distance_step'] = df_ego['Velocity'] * df_ego['Time_diff']
            df_ego['Distance_cumulative'] = df_ego['Distance_step'].cumsum()
            
            # Calculate cumulative distance for follower vehicles
            vehicle_dfs = {}
            for v_id, v_data in vehicle_data.items():
                df_v = pd.DataFrame(v_data)
                df_v['Time_diff'] = df_v['Time'].diff().fillna(0)
                df_v['Distance_step'] = df_v['Velocity'] * df_v['Time_diff']
                df_v['Distance_cumulative'] = df_v['Distance_step'].cumsum()
                vehicle_dfs[v_id] = df_v
            
            # Define traffic light positions on straight road
            traffic_lights = {
                1: 200,    # Traffic light 1 at 200m
                2: 400,    # Traffic light 2 at 400m
                3: 600,    # Traffic light 3 at 600m
                4: 900,    # Traffic light 4 at 900m
                5: 1100,   # Traffic light 5 at 1100m
                6: 1300,   # Traffic light 6 at 1300m
            }
            
            # Create plot
            fig = plt.figure(figsize=(14, 10))
            
            # Plot traffic light phases
            time = df_ego['Timestamp'] - df_ego['Timestamp'].iloc[0]
            tl_states = {
                1: df_ego['Light State 1'],
                2: df_ego['Light State 2'],
                3: df_ego['Light State 3'],
                4: df_ego['Light State 4'],
                5: df_ego['Light State 5'],
                6: df_ego['Light State 6'],
            }
            
            for tl_id, position in traffic_lights.items():
                states = tl_states[tl_id]
                for i in range(len(states) - 1):
                    start_time = time.iloc[i]
                    end_time = time.iloc[i + 1]
                    color = 'red' if states.iloc[i] == 'Red' else 'green'
                    plt.plot([start_time, end_time], [position, position], 
                            color=color, linewidth=3, alpha=0.6)
            
            # Plot ego vehicle trajectory
            plt.plot(time, df_ego['Distance_cumulative'], 
                    color='blue', label=f'{mode} Vehicle (Ego)', linewidth=2.5)
            
            # Plot follower vehicles
            other_vehicles_start_point = [-12 * (i + 1) for i in range(len(vehicle_dfs))]
            counter = 0
            for (v_id, df_v), start_y in zip(sorted(vehicle_dfs.items()), other_vehicles_start_point):
                counter += 1
                adjusted_y = df_v['Distance_cumulative'] + start_y
                v_time = df_v['Time'] - df_v['Time'].iloc[0]
                plt.plot(v_time, adjusted_y, label=f'Vehicle {counter}', linewidth=2)
            
            # Formatting
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Distance (m)', fontsize=12)
            plt.xlim(0, max(135, time.iloc[-1] if len(time) > 0 else 135))
            plt.legend(fontsize=10, loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.title(f'{mode} Mode - Space-Time Diagram', fontsize=14)
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'trajectory_{mode.lower()}_mode.png'
            plt.savefig(plot_filename, dpi=150)
            print(f"Trajectory plot saved to: {plot_filename}")
            plt.close()
        
        traffic_manager.cleanup()
        tl_manager.stop()
        visualizer.close()

if __name__ == '__main__':
    main()
