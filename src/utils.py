import csv
import math
import numpy as np
from .core.primitives import Location, Rotation, Transform

def calculate_fuel_consumption(v, u):
    M_h = 1500.0
    C_D = 0.3
    A_v = 2.2
    mu = 0.01
    g = 9.81
    rho_a = 1.225
    
    b0, b1, b2, b3 = 0.1569, 0.02450, -0.0007415, 0.00005975
    c0, c1, c2 = 0.07224, 0.09681, 0.001075
    idle_fuel_rate = 0.1 / 3600
    
    a_drag = (0.5 * C_D * rho_a * A_v * v**2) / M_h
    a_roll = mu * g
    a_net = u - a_drag - a_roll

    f_cruise = b0 + b1*v + b2*(v**2) + b3*(v**3)
    f_accel = a_net * (c0 + c1*v + c2*(v**2))
    
    zeta = 1 if (np.isclose(v, 0, atol=1e-3) or u < 0) else 0
    fuel_consumption = (1 - zeta) * (f_cruise + f_accel) + zeta * idle_fuel_rate
    return fuel_consumption/1000, a_net

def read_csv_waypoints(csv_file):
    transforms = []
    # print(f"Reading waypoints from {csv_file}")
    try:
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                # Assuming simple flat world for now, preserving full transform
                pitch = float(row['pitch'])
                yaw = float(row['yaw'])
                roll = float(row['roll'])
                t = Transform(Location(x,y,z), Rotation(pitch, yaw, roll))
                transforms.append(t)
    except FileNotFoundError:
        print(f"File {csv_file} not found. Returning empty waypoints.")
    return transforms

def read_waypoints_from_csv(file_path):
    # Returns list of dicts as expected by mpc_agent
    waypoints = []
    try:
        with open(file_path, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                waypoints.append({
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'z': float(row['z']),
                    'pitch': float(row['pitch']),
                    'yaw': float(row['yaw']),
                    'roll': float(row['roll'])
                })
    except FileNotFoundError:
        pass
    return waypoints

def get_waypoints(waypoints_list, N, vehicle_x, vehicle_y, vehicle_psi, current_wp_idx):
    """
    Extracts local waypoints transformed to vehicle coordinate frame.
    """
    waypoints = []
    num_waypoints = len(waypoints_list)
    min_distance = float('inf')
    closest_idx = current_wp_idx

    # Find closest waypoint ahead
    # Optimization: Search within a window if needed, but linear search from current idx is okay for small arrays
    for idx in range(current_wp_idx, num_waypoints):
        wp = waypoints_list[idx]
        dx = wp['x'] - vehicle_x
        dy = wp['y'] - vehicle_y
        distance = np.hypot(dx, dy)
        heading_to_wp = np.arctan2(dy, dx)
        angle = vehicle_psi - heading_to_wp
        angle = np.arctan2(np.sin(angle), np.cos(angle)) # Normalize -pi to pi

        if abs(angle) < np.pi / 2:
            if distance < min_distance:
                min_distance = distance
                closest_idx = idx
        else:
            # If angle > 90 deg, likely previous point (unless U-turn), but since we iterate forward...
            # Actually, if we passed it, angle > 90.
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
            brake = np.clip(-a_desired / abs(a_min), 0.0, 1)
        else:
            throttle = 0.0
            brake = 0.0
    return throttle, brake

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))
