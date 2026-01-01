from dataclasses import dataclass
from typing import List

@dataclass
class PCM:
    vehicle_id: int
    desired_acceleration: float
    desired_spacing: float
    platoon_id: int
    position_in_platoon: int
    target_speed: float
    distance_to_front:float

@dataclass
class PAM:
    leader_id: int
    platoon_id: int
    platoon_size: int
    platoon_speed: float
    vehicle_ids: List[int]
    platoon_position: tuple  # x, y coordinates
    eta_to_light: int # estimated time to next traffic light
    platoon_length: float
    status: str  # e.g. 'Split', 'Merge', or 'stable'
    leaving_vehicles: List[int]
    split_decision_cntr: 0
    corridor_id: int
