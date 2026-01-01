#!/usr/bin/env python3
"""
Generate a straight road route.csv for the karim-platoon simulation.
Creates a 1500m straight road with waypoints every 0.3m.
"""

import csv

def generate_straight_route(output_file="config/route.csv", length=1500, spacing=0.3):
    """
    Generate a straight road route CSV file.
    
    Args:
        output_file: Path to output CSV file
        length: Total road length in meters
        spacing: Distance between waypoints in meters
    """
    num_waypoints = int(length / spacing) + 1
    
    print(f"Generating straight road route:")
    print(f"  Length: {length}m")
    print(f"  Spacing: {spacing}m")
    print(f"  Total waypoints: {num_waypoints}")
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z', 'pitch', 'yaw', 'roll'])
        
        for i in range(num_waypoints):
            x = i * spacing
            y = 0.0  # Straight road along X axis
            z = 0.0  # Flat road
            pitch = 0.0
            yaw = 0.0  # Heading east (positive X direction)
            roll = 0.0
            
            writer.writerow([x, y, z, pitch, yaw, roll])
    
    print(f"  Output: {output_file}")
    print(f"  Done!")
    
    # Print traffic light suggested positions
    print(f"\nSuggested traffic light positions (evenly spaced):")
    tl_positions = [200, 400, 600, 900, 1100, 1300]
    for i, pos in enumerate(tl_positions, 1):
        index = int(pos / spacing)
        print(f"  TL {i}: {pos}m -> waypoint index {index}")

if __name__ == "__main__":
    generate_straight_route()
