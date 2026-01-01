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
