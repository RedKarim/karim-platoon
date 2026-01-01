from enum import Enum
from .primitives import Location, Transform

class TrafficLightState(Enum):
    Red = 0
    Yellow = 1
    Green = 2
    Off = 3
    Unknown = 4

class TrafficLight:
    def __init__(self, tl_id, transform):
        self.id = tl_id
        self.transform = transform
        self.state = TrafficLightState.Red
        self.frozen = False
        self.stop_waypoints = [] # List of waypoints

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def freeze(self, freeze):
        self.frozen = freeze
        
    def get_transform(self):
        return self.transform

    def set_stop_waypoints(self, waypoints):
        self.stop_waypoints = waypoints
        
    def get_stop_waypoints(self):
        return self.stop_waypoints
