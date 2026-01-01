from .primitives import Transform, Waypoint, Location
# from .vehicle import Vehicle # Circular import if world spawns vehicle? No, World imports Vehicle.
import time

class WorldSettings:
    def __init__(self):
        self.synchronous_mode = False
        self.fixed_delta_seconds = 0.1
        self.substepping = False
        self.max_substep_delta_time = 0.01
        self.max_substeps = 10

class Snapshot:
    def __init__(self, frame, elapsed_seconds):
        self.frame = frame
        self.timestamp = Timestamp(elapsed_seconds)

class Timestamp:
    def __init__(self, elapsed_seconds):
        self.elapsed_seconds = elapsed_seconds

class Map:
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def get_waypoint(self, location):
        # Return closest waypoint
        if not self.waypoints:
            return Waypoint(Transform(location))
        
        closest = min(self.waypoints, key=lambda wp: location.distance(wp.transform.location))
        return closest

class World:
    def __init__(self):
        self.actors = []
        self.settings = WorldSettings()
        self.frame = 0
        self.time = 0.0
        self.map = None
        
    def get_settings(self):
        return self.settings

    def apply_settings(self, settings):
        self.settings = settings

    def tick(self):
        self.frame += 1
        dt = self.settings.fixed_delta_seconds if self.settings.fixed_delta_seconds else 0.1
        self.time += dt
        
        for actor in self.actors:
            if hasattr(actor, 'tick'):
                actor.tick(dt)
        return self.frame

    def get_snapshot(self):
        return Snapshot(self.frame, self.time)

    def spawn_actor(self, blueprint, transform):
        # Blueprint could be just a string or object
        from .vehicle import Vehicle
        from .traffic_light import TrafficLight
        
        role = blueprint.get_attribute('role_name') if hasattr(blueprint, 'get_attribute') else 'vehicle'
        
        actor = None
        if "vehicle" in str(blueprint): # checking if it's a vehicle bp
             # Generate ID
            vid = len(self.actors) + 1
            actor = Vehicle(vid, transform, str(blueprint))
        elif "traffic_light" in str(blueprint):
            tid = len(self.actors) + 100
            actor = TrafficLight(tid, transform)
            
        if actor:
            self.actors.append(actor)
            return actor
        return None

    def get_actors(self):
        return ActorList(self.actors)

    def get_map(self):
        return self.map

    def set_map(self, map_obj):
        self.map = map_obj

    def get_blueprint_library(self):
        return BlueprintLibrary()

class Client:
    def __init__(self, host="localhost", port=2000):
        self.world = World()
        
    def get_world(self):
        return self.world

    def get_trafficmanager(self, port=8000):
        # Return a dummy traffic manager object or handle internally
        return None

class ActorList(list):
    def filter(self, pattern):
        # Simple filter by type name
        filtered = ActorList()
        for actor in self:
            if "vehicle" in pattern and hasattr(actor, 'get_velocity'):
                filtered.append(actor)
            elif "traffic_light" in pattern and hasattr(actor, 'get_state'):
                filtered.append(actor)
        return filtered

class BlueprintLibrary:
    def filter(self, pattern):
        # Return a mock blueprint
        return [MockBlueprint(pattern)]

class MockBlueprint:
    def __init__(self, id):
        self.id = id
        self.attributes = {}
    
    def set_attribute(self, key, value):
        self.attributes[key] = value
        
    def get_attribute(self, key):
        return self.attributes.get(key)
    
    def __str__(self):
        return self.id
