import math
from .primitives import Vector3D, Location, Rotation, Transform

class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle = throttle
        self.steer = steer
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse

class BoundingBox:
    def __init__(self, extent):
        self.extent = extent # Vector3D (half-size: x=length/2, y=width/2, z=height/2)
        self.location = Vector3D(0,0,0) # Local offset
        self.rotation = Rotation(0,0,0)

class Vehicle:
    def __init__(self, vehicle_id, transform, blueprint_id="vehicle.tesla.model3"):
        self.id = vehicle_id
        self.transform = transform
        self.velocity = Vector3D(0,0,0) # m/s (world frame)
        self.acceleration = Vector3D(0,0,0)
        self.control = VehicleControl()
        self.is_alive = True
        
        # Physics Params (Simple Bicycle Model)
        # Physics Params (Simple Bicycle Model)
        self.wheelbase = 2.8 # meters
        self.max_steer_angle = math.radians(45) # rad
        
        # Dimensions (Half-extents)
        # Default Model 3 ish: 4.7m long, 1.85m wide, 1.4m high
        self.bounding_box = BoundingBox(Vector3D(2.35, 0.925, 0.7))

    def apply_control(self, control):
        self.control = control

    def get_location(self):
        return self.transform.location

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity

    def get_acceleration(self):
        return self.acceleration
    
    def get_physics_control(self):
        # Mock returns dict or object? Main.py just prints it.
        return "PhysicsControl(Mock)"

    def destroy(self):
        self.is_alive = False

    def tick(self, dt):
        """Update vehicle state for straight-line motion (no steering/lateral dynamics)."""
        # Current state
        x = self.transform.location.x
        y = self.transform.location.y
        yaw = math.radians(self.transform.rotation.yaw)
        v = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        
        # Realistic vehicle parameters (calibrated to CARLA Tesla Model 3)
        max_accel = 3.0  # m/s^2 (realistic for electric vehicle)
        max_brake_decel = 8.0  # m/s^2 (emergency braking)
        comfortable_brake = 4.0  # m/s^2 (normal braking)
        
        # Aerodynamic parameters
        drag_coefficient = 0.24  # Tesla Model 3 Cd
        frontal_area = 2.2  # m^2
        air_density = 1.225  # kg/m^3
        vehicle_mass = 1847  # kg (Tesla Model 3)
        
        # Non-linear throttle/brake response
        if self.control.throttle > 0:
            # Diminishing returns at high speeds (drag and power limits)
            speed_factor = max(0.3, 1.0 - (v / 30.0))
            acc_input = self.control.throttle * max_accel * speed_factor
        elif self.control.brake > 0:
            # Progressive braking force
            if self.control.brake > 0.5:
                acc_input = -self.control.brake * max_brake_decel
            else:
                acc_input = -self.control.brake * comfortable_brake
        else:
            # Coasting - apply rolling resistance
            rolling_resistance = -0.01 * 9.81
            acc_input = rolling_resistance
        
        # Apply aerodynamic drag (quadratic with velocity)
        if v > 0.1:
            drag_force = 0.5 * drag_coefficient * air_density * frontal_area * (v ** 2)
            drag_accel = -drag_force / vehicle_mass
            acc_input += drag_accel
        
        # Straight-line motion only - no lateral dynamics
        # Vehicle maintains constant heading (yaw unchanged)
        
        # Integrate state using forward Euler
        new_x = x + v * math.cos(yaw) * dt
        new_y = y + v * math.sin(yaw) * dt
        new_v = max(0, v + acc_input * dt)  # Prevent negative velocity
        
        # Update State - yaw stays constant (straight road)
        self.transform.location.x = new_x
        self.transform.location.y = new_y
        # self.transform.rotation.yaw stays unchanged (straight motion)
        
        new_yaw = yaw  # Keep original heading
        self.velocity.x = new_v * math.cos(new_yaw)
        self.velocity.y = new_v * math.sin(new_yaw)
        
        self.acceleration.x = acc_input * math.cos(new_yaw)
        self.acceleration.y = acc_input * math.sin(new_yaw)
