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
        """Update vehicle state based on control and dt."""
        # Simple Kinematic Bicycle Model
        
        # Input: throttle [0,1], brake [0,1], steer [-1, 1]
        acc_input = self.control.throttle * 5.0 - self.control.brake * 10.0 # simple scaling
        steer_angle = self.control.steer * self.max_steer_angle
        
        # Current state
        x = self.transform.location.x
        y = self.transform.location.y
        yaw = math.radians(self.transform.rotation.yaw)
        v = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        
        # Heading change
        # beta = arctan(lr / (lf + lr) * tan(steer)) -> approximated for CG at center
        # yaw_rate = v / L * tan(steer_angle)
        yaw_rate = (v / self.wheelbase) * math.tan(steer_angle)
        
        # Update
        new_x = x + v * math.cos(yaw) * dt
        new_y = y + v * math.sin(yaw) * dt
        new_yaw = yaw + yaw_rate * dt
        new_v = v + acc_input * dt
        if new_v < 0: new_v = 0 # No reverse for simplicity unless requested
        
        # Update State
        self.transform.location.x = new_x
        self.transform.location.y = new_y
        self.transform.rotation.yaw = math.degrees(new_yaw)
        
        self.velocity.x = new_v * math.cos(new_yaw)
        self.velocity.y = new_v * math.sin(new_yaw)
        
        self.acceleration.x = acc_input * math.cos(new_yaw) # Simplified (tangential only)
        self.acceleration.y = acc_input * math.sin(new_yaw)
