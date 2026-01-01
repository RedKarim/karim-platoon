# Ported PlatoonManager
from .platoon_messages import PCM, PAM

class PlatoonManager:
    def __init__(self, vehicles, leader_id, traffic_manager, behaviour_agents,platoon_id):
        self.traffic_manager = traffic_manager  # Reference to VehicleTrafficManager
        self.vehicles = vehicles  # list of vehicles
        self.leader_id = leader_id
        self.platoon_id = platoon_id
        self.behind_vehicles = behaviour_agents  # Initialize if needed
        self.pcms = []
        self.pam = None
        self.initialize_platoon()
    
    def initialize_platoon(self):
        # Initialize PCM and PAM
        # Note: assuming vehicle objects have 'distance_to_front' attribute set externally or default to 10
        self.pcms = [PCM(vehicle_id=vehicle.id, desired_acceleration=0.0, desired_spacing=10.0,
                        platoon_id=self.platoon_id, position_in_platoon=idx, target_speed=0.0, 
                        distance_to_front=getattr(vehicle, 'distance_to_front', 10.0))
                     for idx, vehicle in enumerate(self.vehicles)]
        self.pam = PAM(
            leader_id=self.leader_id,
            platoon_id=self.platoon_id,
            platoon_size=len(self.vehicles),
            platoon_speed=0.0,
            vehicle_ids=[v.id for v in self.vehicles],
            platoon_position=(0,0),
            eta_to_light=0,
            platoon_length =sum([v.distance_to_front for v in self.pcms]),
            status="stable",
            leaving_vehicles=[],
            split_decision_cntr=0,
            corridor_id=13
        )
        print(self.pcms)
        print(self.pam)

    def update_platoon(self, platoon_status,tl_id,ref_v,eta_to_light,distances=None):
        # distances is a list of each behind vehicle's gap
        platoon_length = sum(distances[1:]) if distances else 0.0 # leaving out the leader
        leaving_vehicles = [p.vehicle_id for p in platoon_status['rear_group']]
        status = platoon_status['mode']
        # print(f"[PLATOON MANAGER] Distances: {distances}")
        leader_vehicle = None
        pam = self.pam
        pcm_list = []

        # Find and store the leader vehicle
        for idx, vehicle in enumerate(self.vehicles):
            if vehicle.id == self.leader_id:
                leader_vehicle = vehicle
                speed = leader_vehicle.get_velocity()
                current_speed = (speed.x**2 + speed.y**2 + speed.z**2)**0.5
                break
        
        if leader_vehicle:
            # Create PAM for leader
            pam = PAM(
                leader_id=self.leader_id,
                platoon_id=self.platoon_id,
                platoon_size=len(self.vehicles),
                platoon_speed=ref_v,
                vehicle_ids=[v.id for v in self.vehicles],
                platoon_position=(leader_vehicle.get_location().x, leader_vehicle.get_location().y),
                eta_to_light=eta_to_light,  # Could be computed from traffic light manager
                platoon_length=platoon_length,
                status=status,
                leaving_vehicles=leaving_vehicles,
                split_decision_cntr=pam.split_decision_cntr,
                corridor_id=tl_id)
            print(f"[PLATOON MANAGER/Update Platoon] Publishing PAM: {pam}")
            self.pam = pam
        # Create PCM for each vehicle
        for position_in_platoon, vehicle in enumerate(self.vehicles):
            speed = vehicle.get_velocity()
            current_speed = (speed.x**2 + speed.y**2 + speed.z**2)**0.5
            pcm = PCM(
                vehicle_id=vehicle.id,
                desired_acceleration=0.0,  # Could be determined by logic
                desired_spacing=10.0,      # Example spacing
                platoon_id=self.platoon_id,
                position_in_platoon=position_in_platoon,
                target_speed=current_speed,
                distance_to_front=distances[position_in_platoon] if distances else 0.0
            )
            # print(f"Publishing PCM: {pcm}")
            pcm_list.append(pcm)
        self.pcms = pcm_list
        # print(f"[PLATOON MANAGER/Update Platoon] Publishing PCMs: {pcm_list}")

    def split_platoon(self, platoon, new_group, new_platoon_id,tl_id,eta_to_light):
        print("-------------[PLATOON MANAGER]----------------")
        print("[PLATOON MANAGER] Splitting Platoon")
        # In mock, behaviour agents are just dicts or objects. 
        # CAUTION: 'vehicle' in 'sub_platoon_vehicles' logic relies on vehicle.id.
        
        sub_platoon_vehicles=[vehicle for vehicle in self.vehicles if vehicle.id in [v.vehicle_id for v in new_group]]
        sub_platoon_pcms = [pcm for pcm in self.pcms if pcm.vehicle_id in [v.vehicle_id for v in new_group]]
        
        # 'behind_vehicles' structure depends on TrafficManager. 
        # Assuming behind_vehicles is list of dict or obj with 'id' field
        sub_platoon_behind_vehicles = [vehicle for vehicle in self.behind_vehicles if vehicle['id'] in [v.vehicle_id for v in new_group]]
        
        # Update the current platoon to only include the front group
        self.vehicles = [vehicle for vehicle in self.vehicles if vehicle.id in [v.vehicle_id for v in platoon]]
        self.pcms = [pcm for pcm in self.pcms if pcm.vehicle_id in [v.vehicle_id for v in platoon]]
        self.behind_vehicles = [vehicle for vehicle in self.behind_vehicles if vehicle['id'] in [v.vehicle_id for v in platoon]]
        
        # Create a new PlatoonManager for the rear group
        if new_group:
            rear_leader_id = new_group[0].vehicle_id
            new_platoon_manager = PlatoonManager(
                vehicles=sub_platoon_vehicles,
                leader_id=rear_leader_id,
                traffic_manager=self.traffic_manager,
                behaviour_agents=sub_platoon_behind_vehicles,
                platoon_id=new_platoon_id
            )
            # In original code: new_platoon_manager.ego_vehicle = sub_platoon_behind_vehicles[0]
            # new_platoon_manager.ego_vehicle["vehicle"].get_location()
            # We assume sub_platoon_behind_vehicles[0] has key "vehicle"
            
            new_platoon_manager.ego_vehicle = sub_platoon_behind_vehicles[0]
            platoon_position=new_platoon_manager.ego_vehicle["vehicle"].get_location()
            print(f"[PLATOON MANAGER] Platoon Created with Leader Ego: {rear_leader_id}")
            # Initialize PCM and PAM for the new platoon
            new_platoon_manager.pcms = sub_platoon_pcms
            for p,pcm in enumerate(new_platoon_manager.pcms):
                pcm.platoon_id = new_platoon_manager.platoon_id
                pcm.position_in_platoon = p
                pcm.distance_to_front = 100 if rear_leader_id == pcm.vehicle_id else sub_platoon_pcms[p].distance_to_front

            new_platoon_manager.pam = PAM(
                leader_id=rear_leader_id,
                platoon_id=new_platoon_manager.platoon_id,
                platoon_size=len(new_platoon_manager.vehicles),
                platoon_speed=self.pam.platoon_speed*0.9,
                vehicle_ids=[v.id for v in new_platoon_manager.vehicles],
                platoon_position=(platoon_position.x,platoon_position.y),
                eta_to_light=self.pam.eta_to_light + eta_to_light*2,
                platoon_length=sum([0 if rear_leader_id == pcm.vehicle_id else pcm.distance_to_front for pcm in new_platoon_manager.pcms]),
                status="leaving",
                leaving_vehicles=[],
                split_decision_cntr=0,
                corridor_id=tl_id
            )
            print(f"[PLATOON MANAGER] Subplatoon PAM:{new_platoon_manager.pam}")
            print("-------------------------------------------------------")
            print(f"[PLATOON MANAGER] Subplatoon PCM:{new_platoon_manager.pcms}")
            # Add the new platoon to the list of platoons
            return new_platoon_manager
        return None
    
    def cleanup(self):
        """Destroy all vehicles managed by this platoon."""
        for vehicle in self.vehicles:
            if hasattr(vehicle, 'destroy'):
                vehicle.destroy()
        self.vehicles.clear()
        print("PlatoonManager: All vehicles destroyed and cleared.")
