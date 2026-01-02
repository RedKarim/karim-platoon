from .platoon_manager import PlatoonManager
import math
from ..core.traffic_light import TrafficLightState
# Mock carla module for compatibility with ported code
class MockCarla:
    TrafficLightState = TrafficLightState
carla = MockCarla()

class TrafficLightManager:
    def __init__(self, client, traffic_lights_config, waypoint_locations):
        self.client = client
        self.world = client.get_world()
        self.traffic_lights_config = traffic_lights_config
        self.traffic_lights = {}
        # Handle Waypoint objects from my core
        self.waypoints = [wp.transform.location if hasattr(wp, 'transform') else wp for wp in waypoint_locations]
        self.fixed_delta_seconds = 0.1 # Default, can get from settings if needed
        self.v_min = 1
        self.v_max = 50/3.6
        self.ref_v = self.v_max
        self.init_delay = 5 
        self.start_time = None
        self.platoon_manager = None
        self.split_counter = 1
        self.corridor_id = 1
        self.processed_ids = set()
        self.corridor_change = False
        self.pams = []
        
        # Deactivate other traffic lights (Mock World handles get_actors)
        self.deactivate_other_traffic_lights()
        # Initialize
        self.initialize_traffic_lights()

    def deactivate_other_traffic_lights(self):
        # In our mock world, we might filter by type string or object check
        all_traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        for tl in all_traffic_lights:
            if tl.id not in self.traffic_lights_config:
                tl.set_state(TrafficLightState.Off)
                # tl.freeze(True) # Assuming freeze exists
                print(f"Deactivated Traffic Light ID {tl.id} set to Off.")

    def initialize_traffic_lights(self):
        all_traffic_lights = self.world.get_actors().filter('traffic.traffic_light')
        current_tick = self.world.get_snapshot().frame
        
        for tl in all_traffic_lights:
            if tl.id in self.traffic_lights_config:
                config = self.traffic_lights_config[tl.id]
                initial_state = config.get('initial_state', TrafficLightState.Red)
                green_duration = config.get('green_time', 10.0)
                red_duration = config.get('red_time', 10.0)
                
                green_ticks = int(green_duration / self.fixed_delta_seconds)
                red_ticks = int(red_duration / self.fixed_delta_seconds)
                
                tl.set_state(initial_state)
                # tl.freeze(True)
                
                self.traffic_lights[tl.id] = {
                    'actor': tl,
                    'current_state': initial_state,
                    'green_ticks': green_ticks,
                    'red_ticks': red_ticks,
                    'green_time': green_duration,
                    'red_time': red_duration,
                    'last_change_tick': current_tick,
                    'remaining_time': green_duration if initial_state == TrafficLightState.Green else red_duration,
                }
                print(f"Traffic Light {tl.id} initialized with state {initial_state} and {self.traffic_lights[tl.id]['remaining_time']:.2f} seconds remaining.")
                
        # Sort traffic lights by ID for straight road
        self.traffic_lights = {key: self.traffic_lights[key] for key in sorted(self.traffic_lights.keys())}
        # Update corridor_id to first light
        if self.traffic_lights:
            self.corridor_id = list(self.traffic_lights.keys())[0]

    def calculate_route_distance(self, start_location, end_location):
        """
        Calculate distance along straight road (simply X coordinate difference).
        """
        distance = end_location.x - start_location.x
        return max(0, distance)  # Only positive distances (ahead)

    def update_traffic_lights(self, vehicle_location, current_tick):
        self.current_tick = current_tick
        
        if not hasattr(self, 'processed_ids'):
            self.processed_ids = set()

        for tl_id, tl_data in list(self.traffic_lights.items()):
            # Get stop waypoints
            stop_waypoints = tl_data['actor'].get_stop_waypoints()
            if stop_waypoints:
                 stop_location = stop_waypoints[0].transform.location
            else:
                 stop_location = tl_data['actor'].get_transform().location

            distance_to_light = self.calculate_route_distance(vehicle_location, stop_location)
            tl_data['distance'] = distance_to_light
            
            elapsed_ticks = self.current_tick - tl_data['last_change_tick']
            max_ticks = tl_data['green_ticks'] if tl_data['current_state'] == TrafficLightState.Green else tl_data['red_ticks']
            remaining_ticks = max_ticks - elapsed_ticks
            tl_data['remaining_time'] = max(0, remaining_ticks * self.fixed_delta_seconds)

            # Process passed lights
            if distance_to_light < 2 and tl_id not in self.processed_ids:
                 # Logic to handle passed light if needed (e.g. remove from active list or mark processed)
                 # self.processed_ids.add(tl_id)
                 pass

            if elapsed_ticks >= max_ticks:
                if tl_data['current_state'] == TrafficLightState.Green:
                    new_state = TrafficLightState.Red
                    next_duration = tl_data['red_ticks']
                else:
                    new_state = TrafficLightState.Green
                    next_duration = tl_data['green_ticks']
                
                tl = tl_data['actor']
                tl.set_state(new_state)
                tl_data['current_state'] = new_state
                tl_data['last_change_tick'] = self.current_tick
                tl_data['remaining_time'] = next_duration * self.fixed_delta_seconds

    def stop(self):
        for tl_data in self.traffic_lights.values():
            # tl_data['actor'].freeze(False)
            pass
        print("Traffic Light Manager stopped.")

    def get_traffic_lights(self):
        return self.traffic_lights
    
    def is_red_light_ahead(self, vehicle_location, threshold_distance=15.0):
        for tl_id, tl_data in self.traffic_lights.items():
            state = tl_data['current_state']
            if state in [TrafficLightState.Red, TrafficLightState.Yellow]:
                tl_location = tl_data['actor'].get_location()
                distance = tl_location.x - vehicle_location.x
                if 0 < distance <= threshold_distance:
                    return True, tl_id
        return False, None
    
    def get_closest_traffic_light(self, vehicle_location):
        closest_id = None
        min_distance = float('inf')
        for tl_id, tl_data in self.traffic_lights.items():
            tl_location = tl_data['actor'].get_location()
            distance = tl_location.x - vehicle_location.x
            if 0 < distance < min_distance:
                min_distance = distance
                closest_id = tl_id
        
        if closest_id is not None:
             # ETA based on default or platoon speed
             if hasattr(self, 'platoon_manager') and self.platoon_manager:
                 speed = self.platoon_manager.pam.platoon_speed
                 if speed > 0.1:
                    return closest_id, min_distance / speed
             return closest_id, min_distance / 14.0
        return None, None
    
    def get_traffic_light_info(self, tl_id):
        if tl_id in self.traffic_lights:
            tl_data = self.traffic_lights[tl_id]
            return {
                'current_state': tl_data['current_state'],
                'remaining_time': tl_data['remaining_time'],
                'green_time': tl_data['green_time'],
                'red_time': tl_data['red_time'],
                'green_start_time': tl_data['remaining_time'] if tl_data['current_state'] == TrafficLightState.Red else 0,
                'green_end_time': tl_data['remaining_time'] if tl_data['current_state'] == TrafficLightState.Green else tl_data['green_time'],
            }
        return None

    def set_platoon_manager(self, platoon_manager: PlatoonManager):
        # print("Setting Platoon Manager...")
        self.platoon_manager = platoon_manager
        self.refresh_pams(pam=platoon_manager.pam)

    def refresh_pams(self, pam):
        if pam and pam.platoon_id not in [p.platoon_id for p in self.pams]:
            self.pams.append(pam)
        else:
            for existing_pam in self.pams:
                if existing_pam.platoon_id == pam.platoon_id:
                    existing_pam.platoon_speed = pam.platoon_speed
                    existing_pam.leader_id = pam.leader_id
                    existing_pam.leaving_vehicles = pam.leaving_vehicles
                    existing_pam.status = pam.status
                    existing_pam.split_decision_cntr = pam.split_decision_cntr
                    existing_pam.eta_to_light = pam.eta_to_light
                    existing_pam.platoon_length = pam.platoon_length
                    existing_pam.vehicle_ids = pam.vehicle_ids
                    existing_pam.platoon_position = pam.platoon_position
                    existing_pam.platoon_size = pam.platoon_size
                    existing_pam.corridor_id = pam.corridor_id

    # -------------------------------------------------------------
    # Logic Ported from EcoLead
    # -------------------------------------------------------------
    def calculate_reference_velocity(self, current_speed=0):
        current_time = self.world.get_snapshot().timestamp.elapsed_seconds
        if self.start_time is None:
            self.start_time = current_time

        sim_time_elapsed = current_time - self.start_time
        if sim_time_elapsed < self.init_delay:
            return self.ref_v, {
                "mode": "WAITING",  
                "velocity": self.ref_v,
                "front_group": [self.platoon_manager.pam.vehicle_ids],
                "rear_group": [],
                "sub_platoon": None
            }, None, 1000
        
        pam = self.platoon_manager.pam
        all_pcms = self.platoon_manager.pcms
        related_pcms = [pcm for pcm in all_pcms if pcm.vehicle_id in pam.vehicle_ids]
        if not related_pcms: 
            return self.ref_v, {"mode":"NONE","velocity":self.ref_v,"front_group":[],"rear_group":[],"sub_platoon":None}, None, 100
        
        leader_pcm = related_pcms[0]
        # Valid traffic lights ahead
        valid_tls = {k: v for k, v in self.traffic_lights.items() if 'distance' in v and v['distance'] > -50}
        
        if not valid_tls:
             return self.ref_v, {"mode":"NONE","velocity":self.ref_v,"front_group":[],"rear_group":[],"sub_platoon":None}, None, 100
             
        self.corridor_id, tl_data = min(
            valid_tls.items(), key=lambda item: item[1]['distance'] if item[1]['distance'] > 0 else 99999)
        distance_to_light = tl_data['distance']
        
        if distance_to_light - pam.platoon_length >= 300:
            return pam.platoon_speed, {
                "mode": "WAITING FOR SPAT",
                "velocity": pam.platoon_speed,
                "front_group": [pam.vehicle_ids],
                "rear_group": [],
                "sub_platoon": None
            }, self.corridor_id, 100
        
        target_speed = leader_pcm.target_speed if leader_pcm.target_speed > 0.1 else 13.0
        if distance_to_light - pam.platoon_length < 0 or (distance_to_light-pam.platoon_length)<target_speed:
            self.corridor_change = True
        else: self.corridor_change = False

        if distance_to_light < pam.platoon_speed:
            return pam.platoon_speed, {
                "mode": "PASSING",
                "velocity": pam.platoon_speed,
                "front_group": [pam.vehicle_ids],
                "rear_group": [],
                "sub_platoon": None
            }, self.corridor_id, max(0.1,tl_data["distance"]/(pam.platoon_speed+0.001))

        feasible_windows = self._compute_feasible_green_windows()
        if not feasible_windows:
            return self.ref_v, {
                "mode": "NONE",
                "velocity": None,
                "front_group": [],
                "rear_group": [],
                "sub_platoon": None
            }, self.corridor_id, None
        
        # Determine window
        platoon_id = self.platoon_manager.platoon_id
        if platoon_id > 1 and len(feasible_windows) > 1:
            # Check other platoons - Simplified for now, use first window usually
            # EcoLead logic for cooperative merging omitted for brevity unless critical
            pass
            
        (vf_start, vf_end, green_start, green_end) = feasible_windows[0]
        feasible_range = (vf_start, vf_end)

        bool_sat, v_saturation = self._check_saturation_flow(pam.platoon_length, green_start, green_end, vf_start, distance_to_light)
        
        if feasible_range and bool_sat and (pam.platoon_speed > v_saturation or self.corridor_change):
            entire_check = self._check_entire_platoon_feasibility(
                feasible_range, distance_to_light, green_start, green_end, related_pcms, pam, v_saturation
            )
            if entire_check["status"] == "ENTIRE":
                self.ref_v = entire_check["velocity"]
                return self.ref_v, {
                    "mode": "ENTIRE",
                    "velocity": self.ref_v,
                    "front_group": related_pcms,
                    "rear_group": [],
                    "sub_platoon": None
                }, self.corridor_id, tl_data["distance"]/self.ref_v if self.ref_v > 0.1 else 100

            elif entire_check["status"] == "SPLIT":
                result = self.split_for_first_green_window(
                    pam, related_pcms, distance_to_light, green_start, green_end, feasible_range, v_saturation
                )
                eta = max(0.1, tl_data["distance"]/(result["velocity"]+0.001)) if result.get("velocity") else 100
                return self._finalize_split_decision(result, self.corridor_id, eta)
            else:
                if len(feasible_windows) > 1:
                    (vf_start, vf_end, green_start, green_end) = feasible_windows[1]
                    feasible_range = (vf_start, vf_end)
                    result = self.split_for_first_green_window(
                        pam, related_pcms, distance_to_light, green_start, green_end, feasible_range, v_saturation
                    )
                    eta = max(0.1, tl_data["distance"]/(result["velocity"]+0.001)) if result.get("velocity") else 100
                    return self._finalize_split_decision(result, self.corridor_id, eta)

        result = self.split_for_first_green_window(
            pam, related_pcms, distance_to_light, green_start, green_end, feasible_range, v_saturation
        )
        eta = max(0.1, tl_data["distance"]/(result["velocity"]+0.001)) if result.get("velocity") else 100
        return self._finalize_split_decision(result, self.corridor_id, eta)


    def find_feasible_velocity_range(self, distance_full, green_start, green_end):
        if green_end <= 0: return None
        vf_start = max(self.v_min, distance_full / max(green_end,0.3))
        if green_start > 0:
            vf_end = min(self.v_max, distance_full / max(0.3,green_start)) 
        else:
            vf_end = self.v_max
        if vf_start <= vf_end:
            return (vf_start, vf_end)
        return None

    def check_leader_arrival_time(self, distance, chosen_velocity, green_start, green_end, vehicle_velocity):
        if distance < 0: return chosen_velocity, 0
        if chosen_velocity < 0.001: arrival_time = float('inf')
        else: arrival_time = distance / chosen_velocity
        
        actual_arrival_time = distance / max(vehicle_velocity,1)
        delta_time = actual_arrival_time - green_start
        
        if arrival_time < green_start or actual_arrival_time < green_start:
            needed_time = green_start
            if needed_time <= 0: return chosen_velocity, delta_time
            needed_v = distance / needed_time
            if needed_v < chosen_velocity:
                return needed_v * 1.05, delta_time
        
        if arrival_time > green_end and actual_arrival_time > green_end:
            return None, delta_time
        return chosen_velocity, delta_time

    def check_last_arrival_time(self, distance, chosen_velocity, green_end, vehicle_velocity, delta_time_leader):
        if distance < 0: return chosen_velocity
        if chosen_velocity < 0.001: arrival_time = float('inf')
        else: arrival_time = distance / chosen_velocity
        
        actual_arrival_time = distance / max(vehicle_velocity,1)
        # delta_time = actual_arrival_time - arrival_time + delta_time_leader
        
        if arrival_time > green_end or actual_arrival_time > (green_end + delta_time_leader):
            return None
        
        needed_v = chosen_velocity
        if self.corridor_change and chosen_velocity < (distance / max(1,green_end)):
             needed_v = distance / green_end
        return needed_v

    def compute_subplatoon_length(self, pcms_subgroup):
        if not pcms_subgroup: return 0.0
        sorted_sub = sorted(pcms_subgroup, key=lambda p: p.position_in_platoon)
        length = 0.0
        for pcm in sorted_sub[1:]:
            length += pcm.distance_to_front
        return length

    def check_trailing_vehicle_arrival(self, distance_to_stop_line, leader_velocity, green_end, vf_end, delta_time_leader,v_saturation):
        if self.platoon_manager.pam.platoon_speed < v_saturation and leader_velocity < vf_end:
             delta_time_leader = 1
        if distance_to_stop_line < 0: return True
        if leader_velocity < 0.001: arrival_time = float('inf')
        else: arrival_time = (distance_to_stop_line / leader_velocity) + delta_time_leader
        
        if arrival_time < (green_end - 5.0): return True
        return False

    def split_for_first_green_window(self, pam, pcms, distance_to_light, green_start, green_end, feasible_range, v_saturation):
        (vf_start, vf_end) = feasible_range
        sorted_pcms = sorted(pcms, key=lambda x: x.position_in_platoon)
        leader_pcm = next((p for p in sorted_pcms if p.vehicle_id == pam.leader_id), None)
        if not leader_pcm:
            return {"mode": "NONE", "velocity": None, "front_group": [], "rear_group": [],"sub_platoon": None}

        front_group = [leader_pcm]
        rear_candidates = [p for p in sorted_pcms if p != leader_pcm]

        if not self.corridor_change:
            updated_v_leader,delta_time_leader = self.check_leader_arrival_time(
                distance=distance_to_light - pam.platoon_length,
                chosen_velocity=pam.platoon_speed,
                green_start=green_start,
                green_end=green_end,
                vehicle_velocity=leader_pcm.target_speed
            )
        else:
            updated_v_leader = pam.platoon_speed
            delta_time_leader = 0
            
        if not updated_v_leader:
            return {"mode": "NONE", "velocity": None, "front_group": [], "rear_group": [], "sub_platoon": None}

        chosen_v_for_front = updated_v_leader
        feasible_front_group = list(front_group)

        for next_vehicle in rear_candidates:
            candidate_front = feasible_front_group + [next_vehicle]
            sub_len = self.compute_subplatoon_length(candidate_front)
            dist_sub = (distance_to_light - self.platoon_manager.pam.platoon_length) + sub_len
            bool_sat, trailing_v_saturation = self._check_saturation_flow_for_trailing_vehicles(sub_len, green_start, green_end, chosen_v_for_front, dist_sub,v_saturation)
            can_pass = self.check_trailing_vehicle_arrival(
                distance_to_stop_line=dist_sub,
                leader_velocity=chosen_v_for_front,
                green_end=green_end,
                vf_end=vf_end,
                delta_time_leader=delta_time_leader,
                v_saturation=trailing_v_saturation
            )
            if not can_pass:
                break
            feasible_front_group = candidate_front

        rear_group = [v for v in sorted_pcms if v not in feasible_front_group]
        if not rear_group:
            return {
                "mode": "ENTIRE",
                "velocity": chosen_v_for_front,
                "front_group": feasible_front_group,
                "rear_group": [],
                "sub_platoon": None
            }

        pam.split_decision_cntr += 1
        if pam.split_decision_cntr >= 3:
            self.split_counter += 1
            sub_platoon = self.platoon_manager.split_platoon(
                platoon=feasible_front_group,
                new_group=rear_group,
                new_platoon_id=self.split_counter,
                tl_id=self.corridor_id,
                eta_to_light= max(0.1,distance_to_light / chosen_v_for_front)
            )
            pam.split_decision_cntr = 0
            return {
                "mode": "SPLIT",
                "velocity": chosen_v_for_front,
                "front_group": feasible_front_group,
                "rear_group": rear_group,
                "sub_platoon": sub_platoon
            }
        else:
            return {
                "mode": "WAIT",
                "velocity": chosen_v_for_front,
                "front_group": feasible_front_group,
                "rear_group": rear_group,
                "sub_platoon": []}

    def _compute_feasible_green_windows(self):
        valid_tls = {k: v for k, v in self.traffic_lights.items() if 'distance' in v and v['distance'] > -50}
        if not valid_tls: return []
        
        self.corridor_id, tl_data = min(
            valid_tls.items(), key=lambda item: item[1]['distance'] if item[1]['distance'] > 0 else 99999)
        
        distance_to_stop = tl_data['distance']
        remaining_time   = tl_data['remaining_time']
        red_time         = tl_data['red_time']
        green_time       = tl_data['green_time']
        total_cycle      = red_time + green_time

        feasible_windows = []
        current_phase = tl_data['current_state']

        for i in range(4):
            if current_phase == TrafficLightState.Red:
                green_start = remaining_time + i * total_cycle
                green_end = green_start + green_time
            else:
                if i == 0:
                    green_start = 0
                    green_end = remaining_time
                else:
                    green_start = remaining_time + (i - 1) * total_cycle + red_time
                    green_end = remaining_time + i * total_cycle
            
            dist = distance_to_stop
            v_start = max(self.v_min, dist/ max(green_end, 1))
            v_end = min(self.v_max, dist/ max(green_start, 1))
            
            if (v_start <= v_end) and (v_start < 13.0):
                feasible_windows.append((v_start, v_end, green_start, green_end))
        
        if not feasible_windows and self.corridor_change:
             feasible_windows.append((self.platoon_manager.pam.platoon_speed, self.platoon_manager.pam.platoon_speed, green_start, green_end))
             
        return feasible_windows

    def _check_entire_platoon_feasibility(self, feasible_range, distance_to_light, green_start, green_end, related_pcms, pam, v_saturation):
        (v_lower_bound, v_upper_bound) = feasible_range
        leader_pcm = related_pcms[0]
        last_pcm = related_pcms[-1]
        
        if not leader_pcm: return {"status": "NONE", "velocity": None}
        leader_distance = distance_to_light - pam.platoon_length
        
        if not self.corridor_change:
            updated_v_leader,delta_time_leader = self.check_leader_arrival_time(
                distance=leader_distance,
                chosen_velocity=v_upper_bound,
                green_start=green_start,
                green_end=green_end,
                vehicle_velocity=leader_pcm.target_speed
            )
        else:
            updated_v_leader = pam.platoon_speed
            delta_time_leader = 0
            
        if updated_v_leader is None: return {"status": "NONE", "velocity": None}

        last_vehicle_distance = distance_to_light
        updated_v_last = self.check_last_arrival_time(
            distance=last_vehicle_distance,
            chosen_velocity=updated_v_leader,
            green_end=green_end,
            vehicle_velocity=last_pcm.target_speed,
            delta_time_leader=delta_time_leader,
        )

        if updated_v_last is None:
            return {"status": "SPLIT", "velocity": updated_v_leader*0.5}

        candidate_vel = 0.5 * (updated_v_leader + updated_v_last)
        if candidate_vel >= v_lower_bound:
            return {"status": "ENTIRE", "velocity": updated_v_leader if not self.corridor_change else updated_v_last}
        else:
            return {"status": "SPLIT", "velocity": updated_v_leader}

    def _check_saturation_flow(self, platoon_length, green_start, green_end, vf_start ,distance_to_light):
        time_window = green_end - green_start
        if green_start == 0:
            return True, distance_to_light / green_end

        required_speed = platoon_length / time_window
        return (required_speed <= self.v_max-1), required_speed

    def _check_saturation_flow_for_trailing_vehicles(self, platoon_length, green_start, green_end, vf_start,distance_to_light,v_saturation):
        time_window = green_end - green_start
        if green_start == 0:
            return True, distance_to_light / green_end

        required_speed = platoon_length / time_window
        return (required_speed <= self.v_max-1 and required_speed>=v_saturation), required_speed

    def _finalize_split_decision(self, result, traffic_light_key,eta_to_light):
        mode = result["mode"]
        if mode == "NONE":
            self.ref_v = self.v_min
            return self.ref_v, result, traffic_light_key,eta_to_light

        if mode in ["ENTIRE", "SPLIT", "WAIT"]:
            self.ref_v = result["velocity"]
            return self.ref_v, result, traffic_light_key,eta_to_light
        
        self.ref_v = self.v_min
        return self.ref_v, result, traffic_light_key,eta_to_light
