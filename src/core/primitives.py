import math

class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Vector3D(x={self.x}, y={self.y}, z={self.z})"
        
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        
    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    @property
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

class Location(Vector3D):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
        
    def __repr__(self):
        return f"Location(x={self.x}, y={self.y}, z={self.z})"

class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch = float(pitch)
        self.yaw = float(yaw)
        self.roll = float(roll)

    def __repr__(self):
        return f"Rotation(pitch={self.pitch}, yaw={self.yaw}, roll={self.roll})"
        
    def get_forward_vector(self):
        # Simplified conversion: mostly interested in Yaw for 2D/2.5D
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        return Vector3D(
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad)
        )

class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location if location else Location()
        self.rotation = rotation if rotation else Rotation()

    def __repr__(self):
        return f"Transform(location={self.location}, rotation={self.rotation})"
        
    def get_forward_vector(self):
        return self.rotation.get_forward_vector()

class Waypoint:
    def __init__(self, transform):
        self.transform = transform
