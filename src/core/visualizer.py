import pygame
import numpy as np
import math
from .traffic_light import TrafficLightState
from .primitives import Transform as T, Location as L, Vector3D as V

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900

# Layout
TOP_VIEW_HEIGHT = 200 # 2D Map height
BOTTOM_VIEW_HEIGHT = SCREEN_HEIGHT - TOP_VIEW_HEIGHT

VIEWPORT_WIDTH = SCREEN_WIDTH # For 3D view (full width)

FOV = 60
NEAR_PLANE = 0.5
FAR_PLANE = 1000.0

# Colors
SKY_COLOR = (135, 206, 235)  # Sky Blue
GROUND_COLOR = (34, 139, 34) # Forest Green
BACKGROUND_COLOR = (30, 30, 30) # Dark Gray for 2D Map
ROAD_COLOR = (50, 50, 50)    # Gray
MARKING_COLOR = (255, 255, 255)
EGO_COLOR = (0, 100, 255)
FOLLOWER_COLOR = (200, 50, 50) # Slightly darker red
TL_POLE_COLOR = (50, 50, 50)
TIRE_COLOR = (20, 20, 20)
CABIN_COLOR = (0, 70, 180) # Darker blue for ego cabin
CABIN_FOLLOWER_COLOR = (180, 40, 40) # Darker red for follower cabin

class Camera3D:
    """Chase camera for 3D view - optimized for straight road."""
    
    def __init__(self, fov=60, width=VIEWPORT_WIDTH, height=BOTTOM_VIEW_HEIGHT):
        self.position = np.array([0.0, 0.0, 5.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0]) # Z-up
        
        self.width = width
        self.height = height
        self.aspect = width / height
        self.fov_rad = math.radians(fov)
        self.f = 1.0 / math.tan(self.fov_rad / 2)
        
        # Follow parameters - optimized for straight road
        self.distance = 20.0  # Distance behind
        self.height_offset = 6.0  # Height above
        self.smooth_speed = 0.15  # Faster tracking
    
    def update_follow(self, vehicle_transform):
        """Update camera to follow vehicle along straight road."""
        v_loc = vehicle_transform.location
        v_rot = vehicle_transform.rotation
        
        # Vehicle position
        target_pos = np.array([v_loc.x, v_loc.y, v_loc.z])
        
        # For straight road (heading ~0), camera is behind on -X axis
        yaw_rad = math.radians(v_rot.yaw)
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0])
        
        desired_pos = target_pos - forward * self.distance
        desired_pos[2] += self.height_offset
        
        # Smooth update
        self.position = self.position * (1 - self.smooth_speed) + desired_pos * self.smooth_speed
        
        # Look ahead of the car
        look_target = target_pos + forward * 10.0
        self.target = look_target

    def get_view_matrix(self):
        """Create view matrix using LookAt."""
        z_axis = self.position - self.target
        norm_z = np.linalg.norm(z_axis)
        if norm_z < 1e-6:
            z_axis = np.array([0, 0, 1])
        else:
            z_axis = z_axis / norm_z
        
        x_axis = np.cross(self.up, z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / norm_x
            
        y_axis = np.cross(z_axis, x_axis)
        
        view = np.eye(4)
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[0, 3] = -np.dot(x_axis, self.position)
        view[1, 3] = -np.dot(y_axis, self.position)
        view[2, 3] = -np.dot(z_axis, self.position)
        
        return view

    def project_points(self, points_3d):
        """Project 3D points to 2D screen coordinates."""
        num_points = len(points_3d)
        if num_points == 0: 
            return None, np.array([])
        
        # Homogeneous coordinates
        points_4d = np.hstack((points_3d, np.ones((num_points, 1))))
        
        # View Transform
        view_mat = self.get_view_matrix()
        points_view = points_4d @ view_mat.T
        
        # Check visibility
        z_vals = points_view[:, 2]
        valid_mask = z_vals < -NEAR_PLANE
        
        projected = np.zeros((num_points, 2))
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None, valid_mask
            
        p_valid = points_view[valid_indices]
        x = p_valid[:, 0]
        y = p_valid[:, 1]
        z = p_valid[:, 2]
        
        x_ndc = (x / -z) * (self.f / self.aspect)
        y_ndc = (y / -z) * self.f
        
        screen_x = (x_ndc + 1.0) * self.width * 0.5
        screen_y = (1.0 - y_ndc) * self.height * 0.5
        
        projected[valid_indices, 0] = screen_x
        projected[valid_indices, 1] = screen_y
        
        return projected, valid_mask


class Visualizer:
    """Visualizer for straight road simulation with split 2D (Top) / 3D (Bottom) view."""
    
    def __init__(self, waypoints):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Platoon Simulation - 1500m Straight Road")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14)
        self.large_font = pygame.font.SysFont('Arial', 20)
        self.waypoints = waypoints
        self.running = True
        
        # Surfaces
        self.surface_3d = pygame.Surface((SCREEN_WIDTH, BOTTOM_VIEW_HEIGHT))
        self.surface_2d = pygame.Surface((SCREEN_WIDTH, TOP_VIEW_HEIGHT))
        
        self.road_width = 7.5
        
        # 3D Setup
        self.road_geometry = self._build_road_mesh()
        self.camera = Camera3D(width=SCREEN_WIDTH, height=BOTTOM_VIEW_HEIGHT)
        
        # 2D view parameters - fixed wide view or scrolling
        self.view_window = 250.0  # Show 250m window to see platoon easily
        
        # Interactive Control
        self.tracked_vehicle_id = None # None means Ego
        self.buttons = {} 
        self._init_buttons()
        self.available_leaders = []
        
    def _init_buttons(self):
        """Initialize UI buttons."""
        y = TOP_VIEW_HEIGHT + 10
        w, h = 100, 30
        self.buttons['next'] = pygame.Rect(SCREEN_WIDTH - 120, y, w, h)
        self.buttons['ego'] = pygame.Rect(SCREEN_WIDTH - 230, y, w, h)
        
    def _build_road_mesh(self):
        """Build road mesh for 3D rendering - straight road."""
        road_quads = []
        if not self.waypoints: return np.array([])
        wps = np.array([[w.x, w.y, getattr(w, 'z', 0)] for w in self.waypoints])
        num = len(wps)
        if num < 2: return np.array([])
        half_w = self.road_width / 2.0
        left_edge = []; right_edge = []
        for i in range(num):
            p0 = wps[i]
            l = p0.copy(); l[1] -= half_w
            r = p0.copy(); r[1] += half_w
            left_edge.append(l); right_edge.append(r)
        step = max(1, len(left_edge) // 500)
        geometry = []
        for i in range(0, len(left_edge) - step, step):
            next_i = min(i + step, len(left_edge) - 1)
            quad = np.array([left_edge[i], right_edge[i], right_edge[next_i], left_edge[next_i]])
            geometry.append(quad)
        return np.array(geometry) if geometry else np.array([])

    def draw_road_3d(self, surface):
        """Draw road in 3D view."""
        if len(self.road_geometry) == 0: return
        cam_pos = self.camera.position
        centers = np.mean(self.road_geometry, axis=1)
        dists = np.linalg.norm(centers - cam_pos, axis=1)
        visible_indices = np.where(dists < 300)[0] # Visible dist
        if len(visible_indices) == 0: return
        visible_quads = self.road_geometry[visible_indices]
        M = len(visible_quads)
        points_flat = visible_quads.reshape(-1, 3)
        projected, mask = self.camera.project_points(points_flat)
        if projected is None: return
        mask_quads = mask.reshape(M, 4)
        valid_quads = np.all(mask_quads, axis=1)
        quads_2d = projected.reshape(M, 4, 2)
        final_quads = quads_2d[valid_quads]
        valid_dists = dists[visible_indices][valid_quads]
        sorted_order = np.argsort(valid_dists)[::-1]
        for idx in sorted_order:
            poly = final_quads[idx]
            pygame.draw.polygon(surface, ROAD_COLOR, poly)
        self._draw_center_line_3d(surface)

    def _draw_center_line_3d(self, surface):
        cam_x = self.camera.position[0]
        # Draw dashed line
        for x in range(int(cam_x - 100), int(cam_x + 300), 10):
            if x < 0 or x > 1500: continue
            start = np.array([[x, 0, 0.01]])
            end = np.array([[x + 5, 0, 0.01]])
            start_proj, start_mask = self.camera.project_points(start)
            end_proj, end_mask = self.camera.project_points(end)
            if start_proj is not None and end_proj is not None and start_mask[0] and end_mask[0]:
                pygame.draw.line(surface, MARKING_COLOR, 
                               (int(start_proj[0, 0]), int(start_proj[0, 1])),
                               (int(end_proj[0, 0]), int(end_proj[0, 1])), 2)

    def _project_poly(self, poly_3d):
        projected, mask = self.camera.project_points(poly_3d)
        if projected is None or not np.all(mask):
            return None
        return projected

    def draw_box_mesh(self, surface, transform, extent, color, outline_color=(0,0,0)):
        """Generic box drawer."""
        l, w, h = extent[0], extent[1], extent[2]
        corners = np.array([
            [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h], # Bottom
            [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h]      # Top
        ])
        yaw_rad = math.radians(transform.rotation.yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        rotated_corners = corners @ R.T
        world_corners = rotated_corners + np.array([transform.location.x, transform.location.y, transform.location.z])
        
        projected = self._project_poly(world_corners)
        if projected is None: return

        # Draw faces (Painter's algo simple order: Bottom, Top, Sides... ideally sort by normal, but car is convex)
        faces = [
            [0,1,2,3], # Bottom
            [4,5,6,7], # Top
            [0,1,5,4], # Front
            [1,2,6,5], # Right
            [2,3,7,6], # Back
            [3,0,4,7]  # Left
        ]
        
        # Sort faces by depth
        # center of face
        face_depths = []
        cam_pos = self.camera.position
        for i, face_idxs in enumerate(faces):
            center = np.mean(world_corners[face_idxs], axis=0)
            dist = np.linalg.norm(center - cam_pos)
            face_depths.append((dist, i))
        
        face_depths.sort(key=lambda x: x[0], reverse=True)
        
        for dist, idx in face_depths:
            face = faces[idx]
            poly = projected[face]
            pygame.draw.polygon(surface, color, poly)
            if outline_color:
                pygame.draw.polygon(surface, outline_color, poly, 1)

    def draw_vehicle_realistic(self, surface, vehicle, is_ego=False):
        """Draw a more detailed vehicle model with wheels and cabin."""
        transform = vehicle.get_transform()
        loc = transform.location
        
        # Dimensions
        length = 4.6
        width = 1.8
        height_chassis = 0.6
        height_cabin = 0.5
        
        chassis_color = EGO_COLOR if is_ego else FOLLOWER_COLOR
        cabin_color = CABIN_COLOR if is_ego else CABIN_FOLLOWER_COLOR
        
        # 1. Wheels (4 boxes)
        wheel_radius = 0.35
        wheel_width = 0.3
        wheel_offsets = [
            (1.4, 0.9), (1.4, -0.9), (-1.4, 0.9), (-1.4, -0.9)
        ]
        
        # Helper to rotate offsets
        yaw_rad = math.radians(transform.rotation.yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        
        def get_world_pos(lx, ly, lz):
            rx = lx * cy - ly * sy
            ry = lx * sy + ly * cy
            return L(loc.x + rx, loc.y + ry, loc.z + lz)

        # Draw Wheels
        for wx, wy in wheel_offsets:
            w_pos = get_world_pos(wx, wy, wheel_radius)
            # Wheel is a box for now, maybe rotated? 
            # Simple box
            w_trans = T(w_pos, transform.rotation)
            self.draw_box_mesh(surface, w_trans, (wheel_radius, wheel_width/2, wheel_radius), TIRE_COLOR, None)

        # 2. Chassis
        c_pos = get_world_pos(0, 0, wheel_radius + height_chassis/2)
        c_trans = T(c_pos, transform.rotation)
        self.draw_box_mesh(surface, c_trans, (length/2, width/2, height_chassis/2), chassis_color, (0,0,0))
        
        # 3. Cabin (Upper part)
        cab_pos = get_world_pos(-0.3, 0, wheel_radius + height_chassis + height_cabin/2)
        cab_trans = T(cab_pos, transform.rotation)
        self.draw_box_mesh(surface, cab_trans, (length/3, width/2.2, height_cabin/2), cabin_color, (0,0,0))
        
        # 4. Headlights (Yellow quads on front face of Chassis)
        # Simplified: Just small boxes
        hl_pos_l = get_world_pos(length/2 + 0.05, 0.6, wheel_radius + height_chassis/2)
        hl_pos_r = get_world_pos(length/2 + 0.05, -0.6, wheel_radius + height_chassis/2)
        self.draw_box_mesh(surface, T(hl_pos_l, transform.rotation), (0.05, 0.2, 0.1), (255, 255, 200), None)
        self.draw_box_mesh(surface, T(hl_pos_r, transform.rotation), (0.05, 0.2, 0.1), (255, 255, 200), None)
        
        # Taillights (Red)
        tl_pos_l = get_world_pos(-(length/2 + 0.05), 0.6, wheel_radius + height_chassis/2)
        tl_pos_r = get_world_pos(-(length/2 + 0.05), -0.6, wheel_radius + height_chassis/2)
        self.draw_box_mesh(surface, T(tl_pos_l, transform.rotation), (0.05, 0.2, 0.1), (200, 0, 0), None)
        self.draw_box_mesh(surface, T(tl_pos_r, transform.rotation), (0.05, 0.2, 0.1), (200, 0, 0), None)

    def draw_traffic_light_realistic(self, surface, tl_data, tl_id):
        """Draw realistic traffic light with pole and lamps."""
        actor = tl_data['actor']
        state = tl_data['current_state']
        transform = actor.get_transform()
        
        loc = transform.location
        
        # 1. Pole (Tall thin box)
        pole_height = 6.0
        pole_pos = L(loc.x, loc.y, pole_height/2)
        pole_trans = T(pole_pos, transform.rotation)
        self.draw_box_mesh(surface, pole_trans, (0.2, 0.2, pole_height/2), TL_POLE_COLOR, None)
        
        # 3. Housing (Vertical Box) mounting the lights
        housing_h = 1.0
        housing_w = 0.4
        housing_d = 0.4
        housing_z = pole_height - 1.5
        housing_pos = L(loc.x, loc.y, housing_z)
        housing_trans = T(housing_pos, transform.rotation)
        self.draw_box_mesh(surface, housing_trans, (housing_d/2, housing_w/2, housing_h/2), (30, 30, 30))
        
        # 4. Lamps (Circles/Quads on face)
        lamp_colors = {
            'Red': (50, 0, 0),
            'Yellow': (50, 50, 0),
            'Green': (0, 50, 0)
        }
        
        active_colors = {
            'Red': (255, 0, 0),
            'Yellow': (255, 255, 0),
            'Green': (0, 255, 0)
        }
        
        current_color_name = 'Red'
        if state == TrafficLightState.Green: current_color_name = 'Green'
        elif state == TrafficLightState.Yellow: current_color_name = 'Yellow'
        
        # Draw 3 lamps
        lamps = [('Red', 0.25), ('Yellow', 0.0), ('Green', -0.25)]
        
        yaw_rad = math.radians(transform.rotation.yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        
        for name, z_off in lamps:
            col = active_colors[name] if name == current_color_name else lamp_colors[name]
            # Local offset
            lx = -housing_d/2 - 0.05
            ly = 0
            lz = z_off
            
            # Rotate
            rx = lx * cy - ly * sy
            ry = lx * sy + ly * cy
            
            l_pos = L(housing_pos.x + rx, housing_pos.y + ry, housing_pos.z + lz)
            l_trans = T(l_pos, transform.rotation)
            
            # Draw small flat box as light
            self.draw_box_mesh(surface, l_trans, (0.02, 0.12, 0.12), col, None)

    def draw_2d_view(self, surface, ego_vehicle, traffic_vehicles, traffic_lights, leader_ids=None):
        """Draw 2D view at top, centered on platoon."""
        surface.fill(BACKGROUND_COLOR)
        
        # Calculate center of platoon (ego + followers)
        locations = [ego_vehicle.get_location().x]
        for v in traffic_vehicles:
            locations.append(v['vehicle'].get_location().x)
        
        center_x = sum(locations) / len(locations)
        
        view_start = center_x - self.view_window * 0.5
        
        scale_x = SCREEN_WIDTH / self.view_window
        road_y = TOP_VIEW_HEIGHT // 2
        
        # Draw road
        pygame.draw.rect(surface, ROAD_COLOR, (0, road_y - 30, SCREEN_WIDTH, 60))
        pygame.draw.line(surface, MARKING_COLOR, (0, road_y - 30), (SCREEN_WIDTH, road_y - 30), 2)
        pygame.draw.line(surface, MARKING_COLOR, (0, road_y + 30), (SCREEN_WIDTH, road_y + 30), 2)
        
        # Dashes
        for x in range(0, SCREEN_WIDTH, 40):
            pygame.draw.line(surface, MARKING_COLOR, (x, road_y), (x + 20, road_y), 2)
            
        # Markers
        start_marker = int(view_start // 100) * 100
        for d in range(start_marker, int(view_start + self.view_window) + 100, 100):
            screen_x = (d - view_start) * scale_x
            if 0 <= screen_x <= SCREEN_WIDTH:
                pygame.draw.line(surface, (80, 80, 80), (int(screen_x), road_y - 40), (int(screen_x), road_y + 40), 1)
                text = self.font.render(f"{d}m", True, (180, 180, 180))
                surface.blit(text, (int(screen_x) - 10, road_y - 50))

        # Traffic Lights
        for tl_id, tl_data in traffic_lights.items():
            tl_actor = tl_data.get('actor')
            state = tl_data.get('current_state')
            if tl_actor:
                loc = tl_actor.get_location()
                screen_x = (loc.x - view_start) * scale_x
                if 0 <= screen_x <= SCREEN_WIDTH:
                    # Color based on state
                    c = (100, 100, 100)
                    if state == TrafficLightState.Red: c = (255, 0, 0)
                    elif state == TrafficLightState.Green: c = (0, 255, 0)
                    elif state == TrafficLightState.Yellow: c = (255, 255, 0)
                    
                    pygame.draw.line(surface, TL_POLE_COLOR, (int(screen_x), road_y - 30), (int(screen_x), road_y - 60), 3)
                    pygame.draw.circle(surface, c, (int(screen_x), road_y - 65), 8)
                    lbl = self.font.render(f"TL{tl_id}", True, c)
                    surface.blit(lbl, (int(screen_x) - 10, road_y - 80))

        # Vehicles
        def draw_veh(v, col, lbl):
            loc = v.get_location()
            sx = (loc.x - view_start) * scale_x
            if 0 <= sx <= SCREEN_WIDTH:
                rect = pygame.Rect(int(sx) - 10, road_y - 6, 20, 12)
                pygame.draw.rect(surface, col, rect)
                pygame.draw.rect(surface, (0,0,0), rect, 1)
                if lbl:
                    t = self.font.render(lbl, True, (255, 255, 255))
                    surface.blit(t, (int(sx) - 5, road_y + 15))

        for i, v_data in enumerate(traffic_vehicles):
            v_id = v_data.get('id', v_data['vehicle'].id)
            color = FOLLOWER_COLOR
            label = str(i + 1)
            
            if leader_ids and v_id in leader_ids:
                color = EGO_COLOR
                label += " (L)"
                
            draw_veh(v_data['vehicle'], color, label)
        draw_veh(ego_vehicle, EGO_COLOR, "L")

    def _update_controls(self, ego_id, leader_ids):
        """Update buttons and input."""
        # Setup available targets
        self.available_leaders = [ego_id]
        if leader_ids:
            self.available_leaders.extend([lid for lid in leader_ids if lid != ego_id])
            
    def update(self, ego_vehicle, traffic_vehicles, traffic_lights, tick_info, leader_ids=None):
        if not self.running: return
        
        # Helpers
        ego_id = ego_vehicle.id
        self._update_controls(ego_id, leader_ids)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit(); return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_n:
                    # Switch leader
                    curr_idx = -1
                    if self.tracked_vehicle_id in self.available_leaders:
                        curr_idx = self.available_leaders.index(self.tracked_vehicle_id)
                    
                    next_idx = (curr_idx + 1) % len(self.available_leaders)
                    self.tracked_vehicle_id = self.available_leaders[next_idx]
                elif event.key == pygame.K_e:
                    self.tracked_vehicle_id = ego_id
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if self.buttons['next'].collidepoint(mx, my):
                    # Next
                    curr_idx = -1
                    if self.tracked_vehicle_id in self.available_leaders:
                        curr_idx = self.available_leaders.index(self.tracked_vehicle_id)
                    next_idx = (curr_idx + 1) % len(self.available_leaders)
                    self.tracked_vehicle_id = self.available_leaders[next_idx]
                elif self.buttons['ego'].collidepoint(mx, my):
                    # Ego
                    self.tracked_vehicle_id = ego_id

        # Determine target vehicle
        target_v = ego_vehicle
        if self.tracked_vehicle_id and self.tracked_vehicle_id != ego_id:
            # Find in followers
            found = False
            for v_data in traffic_vehicles:
                if v_data['vehicle'].id == self.tracked_vehicle_id:
                    target_v = v_data['vehicle']
                    found = True
                    break
            if not found:
                self.tracked_vehicle_id = ego_id # Fallback
                target_v = ego_vehicle
                
        # Update Camera
        self.camera.update_follow(target_v.get_transform())
        
        # 3D Render
        self.surface_3d.fill(SKY_COLOR)
        # Horizon / Sky logic? Just fill sky, then draw ground rect
        pygame.draw.rect(self.surface_3d, GROUND_COLOR, (0, BOTTOM_VIEW_HEIGHT//2, SCREEN_WIDTH, BOTTOM_VIEW_HEIGHT//2))
        self.draw_road_3d(self.surface_3d)
        
        # Draw Objects 3D
        # Sort objects by distance to camera to handle simple occlusion
        cam_pos = self.camera.position
        
        render_queue = []
        
        # Traffic Lights
        for tl_id, tl_data in traffic_lights.items():
            tl_actor = tl_data.get('actor')
            if tl_actor:
                loc = tl_actor.get_location()
                dist = np.linalg.norm(np.array([loc.x, loc.y, loc.z]) - cam_pos)
                if dist < 200: # Cull far
                    render_queue.append((dist, 'TL', tl_data, tl_id))
                    
        # Vehicles
        dist = np.linalg.norm(np.array([ego_vehicle.get_location().x, ego_vehicle.get_location().y, 0]) - cam_pos)
        render_queue.append((dist, 'VEH', ego_vehicle, True)) # is_ego
        
        for v_data in traffic_vehicles:
            v_id = v_data.get('id', v_data['vehicle'].id)
            v = v_data['vehicle']
            dist = np.linalg.norm(np.array([v.get_location().x, v.get_location().y, 0]) - cam_pos)
            if dist < 200:
                render_queue.append((dist, 'VEH', v, False))
                
        # Sort back to front (larger distance first) for Painter's Algorithm
        render_queue.sort(key=lambda x: x[0], reverse=True)
        
        for item in render_queue:
            if item[1] == 'TL':
                self.draw_traffic_light_realistic(self.surface_3d, item[2], item[3])
            elif item[1] == 'VEH':
                self.draw_vehicle_realistic(self.surface_3d, item[2], item[3])

        # 2D Render
        self.draw_2d_view(self.surface_2d, ego_vehicle, traffic_vehicles, traffic_lights, leader_ids)

        # Blit to Screen
        self.screen.blit(self.surface_2d, (0, 0))
        self.screen.blit(self.surface_3d, (0, TOP_VIEW_HEIGHT))
        
        # Separator
        pygame.draw.line(self.screen, (255, 255, 255), (0, TOP_VIEW_HEIGHT), (SCREEN_WIDTH, TOP_VIEW_HEIGHT), 2)
        
        # HUD Controls
        # Draw Buttons
        mouse_pos = pygame.mouse.get_pos()
        
        # Ego Button (Blue)
        col = (0, 100, 200) if self.buttons['ego'].collidepoint(mouse_pos) else (0, 80, 160)
        pygame.draw.rect(self.screen, col, self.buttons['ego'])
        pygame.draw.rect(self.screen, (255, 255, 255), self.buttons['ego'], 2)
        txt = self.font.render("EGO View", True, (255, 255, 255))
        self.screen.blit(txt, (self.buttons['ego'].x + 10, self.buttons['ego'].y + 5))
        
        # Next Button (Green)
        col = (0, 200, 100) if self.buttons['next'].collidepoint(mouse_pos) else (0, 160, 80)
        pygame.draw.rect(self.screen, col, self.buttons['next'])
        pygame.draw.rect(self.screen, (255, 255, 255), self.buttons['next'], 2)
        txt = self.font.render("Next Leader", True, (255, 255, 255))
        self.screen.blit(txt, (self.buttons['next'].x + 10, self.buttons['next'].y + 5))
        
        # HUD Info
        tracked_name = "Ego (Leader)"
        if self.tracked_vehicle_id and self.tracked_vehicle_id != ego_id:
             tracked_name = f"Vehicle {self.tracked_vehicle_id}"
             
        info = [
            f"Tick: {tick_info['tick']}", 
            f"Speed: {tick_info['speed']*3.6:.1f} km/h",
            f"Camera Tracking: {tracked_name}",
            f"(Press N or Click Button to Switch)"
        ]
        for i, line in enumerate(info):
            t = self.large_font.render(line, True, (255, 255, 255))
            self.screen.blit(t, (10, TOP_VIEW_HEIGHT + 10 + i * 25))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
