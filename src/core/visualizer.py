import pygame
import numpy as np
import math
from .traffic_light import TrafficLightState

# Constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 800
VIEWPORT_WIDTH = SCREEN_WIDTH // 2
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
FOLLOWER_COLOR = (255, 50, 50)
TL_POLE_COLOR = (100, 100, 100)

class Camera3D:
    """Chase camera for 3D view - optimized for straight road."""
    
    def __init__(self, fov=60, width=VIEWPORT_WIDTH, height=SCREEN_HEIGHT):
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
        self.smooth_speed = 0.15  # Faster tracking for straight road

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
        
        # Check visibility (points with negative Z in view space are in front)
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
    """Visualizer for straight road simulation with split 3D/2D view."""
    
    def __init__(self, waypoints):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Platoon Simulation - 1500m Straight Road")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.large_font = pygame.font.SysFont('Arial', 24)
        self.waypoints = waypoints
        self.running = True
        
        # Surfaces
        self.surface_3d = pygame.Surface((VIEWPORT_WIDTH, SCREEN_HEIGHT))
        self.surface_2d = pygame.Surface((VIEWPORT_WIDTH, SCREEN_HEIGHT))
        
        # Road parameters for straight road
        self.road_width = 7.5  # meters
        self.road_length = 1500.0  # meters
        
        # Calculate road bounds
        if waypoints:
            self.min_x = min(w.x for w in self.waypoints)
            self.max_x = max(w.x for w in self.waypoints)
            self.min_y = min(w.y for w in self.waypoints) - self.road_width
            self.max_y = max(w.y for w in self.waypoints) + self.road_width
        else:
            self.min_x, self.max_x = 0, 1500
            self.min_y, self.max_y = -10, 10
        
        # 3D Setup
        self.road_geometry = self._build_road_mesh()
        self.camera = Camera3D(width=VIEWPORT_WIDTH, height=SCREEN_HEIGHT)
        
        # 2D view parameters - scrolling horizontal view
        self.view_window = 200.0  # Show 200m of road at a time
        self.ego_screen_x = VIEWPORT_WIDTH * 0.3  # Ego at 30% from left
        
    def _build_road_mesh(self):
        """Build road mesh for 3D rendering - straight road."""
        road_quads = []
        
        if not self.waypoints:
            return np.array([])
        
        wps = np.array([[w.x, w.y, getattr(w, 'z', 0)] for w in self.waypoints])
        num = len(wps)
        
        if num < 2:
            return np.array([])
        
        half_w = self.road_width / 2.0
        
        left_edge = []
        right_edge = []
        
        # For straight road along X axis, left/right are +/- Y
        for i in range(num):
            p0 = wps[i]
            # For straight road, right vector is (0, 1, 0)
            right = np.array([0, 1, 0])
            
            l = p0.copy()
            l[1] -= half_w
            r = p0.copy()
            r[1] += half_w
            
            left_edge.append(l)
            right_edge.append(r)
            
        # Build Quads - sample every 10 waypoints for efficiency
        step = max(1, len(left_edge) // 500)
        geometry = []
        
        for i in range(0, len(left_edge) - step, step):
            next_i = min(i + step, len(left_edge) - 1)
            
            quad = np.array([
                left_edge[i],
                right_edge[i],
                right_edge[next_i],
                left_edge[next_i]
            ])
            geometry.append(quad)
            
        return np.array(geometry) if geometry else np.array([])

    def draw_road_3d(self, surface):
        """Draw road in 3D view."""
        if len(self.road_geometry) == 0:
            return
            
        cam_pos = self.camera.position
        
        # Calculate centers of quads for distance-based culling
        centers = np.mean(self.road_geometry, axis=1)
        dists = np.linalg.norm(centers - cam_pos, axis=1)
        
        # Draw only within 200m
        visible_indices = np.where(dists < 200)[0]
        
        if len(visible_indices) == 0:
            return
            
        visible_quads = self.road_geometry[visible_indices]
        
        # Project all vertices
        M = len(visible_quads)
        points_flat = visible_quads.reshape(-1, 3)
        projected, mask = self.camera.project_points(points_flat)
        
        if projected is None:
            return
        
        # Draw quads where all 4 points are valid
        mask_quads = mask.reshape(M, 4)
        valid_quads = np.all(mask_quads, axis=1)
        quads_2d = projected.reshape(M, 4, 2)
        final_quads = quads_2d[valid_quads]
        
        # Sort by distance (furthest first)
        valid_dists = dists[visible_indices][valid_quads]
        sorted_order = np.argsort(valid_dists)[::-1]
        
        # Draw road surface
        for idx in sorted_order:
            poly = final_quads[idx]
            pygame.draw.polygon(surface, ROAD_COLOR, poly)
            
        # Draw center dashed line
        self._draw_center_line_3d(surface)

    def _draw_center_line_3d(self, surface):
        """Draw dashed center line on road in 3D."""
        cam_x = self.camera.position[0]
        
        # Draw dashes every 10m
        for x in range(int(cam_x - 100), int(cam_x + 150), 10):
            if x < 0 or x > 1500:
                continue
                
            # Each dash is 5m long
            start = np.array([[x, 0, 0.01]])
            end = np.array([[x + 5, 0, 0.01]])
            
            start_proj, start_mask = self.camera.project_points(start)
            end_proj, end_mask = self.camera.project_points(end)
            
            if start_proj is not None and end_proj is not None and start_mask[0] and end_mask[0]:
                pygame.draw.line(surface, MARKING_COLOR, 
                               (int(start_proj[0, 0]), int(start_proj[0, 1])),
                               (int(end_proj[0, 0]), int(end_proj[0, 1])), 2)

    def draw_box_3d(self, surface, transform, extent, color):
        """Draw 3D bounding box for vehicles."""
        l = extent.x
        w = extent.y
        h = extent.z
        
        corners = np.array([
            [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h],  # Bottom
            [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h]       # Top
        ])
        
        yaw_rad = math.radians(transform.rotation.yaw)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        R = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        
        rotated_corners = corners @ R.T
        world_corners = rotated_corners + np.array([transform.location.x, transform.location.y, transform.location.z])
        
        projected, mask = self.camera.project_points(world_corners)
        
        if projected is None or not np.all(mask):
            return
        
        faces = [
            [0,1,2,3], [4,5,6,7],  # Bottom, Top
            [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]  # Sides
        ]
        
        for face in faces:
            poly = projected[face]
            pygame.draw.polygon(surface, color, poly)
            pygame.draw.polygon(surface, (0,0,0), poly, 1)

    def draw_2d_view(self, surface, ego_vehicle, traffic_vehicles, traffic_lights):
        """Draw scrolling 2D top-down view of straight road."""
        surface.fill(BACKGROUND_COLOR)
        
        # Get ego position for scrolling
        ego_loc = ego_vehicle.get_location()
        ego_x = ego_loc.x
        
        # Calculate view window
        view_start = ego_x - self.view_window * 0.3
        view_end = ego_x + self.view_window * 0.7
        
        # Scale for 2D view
        scale_x = VIEWPORT_WIDTH / self.view_window
        road_y = SCREEN_HEIGHT // 2  # Road at center
        
        # Draw road background
        pygame.draw.rect(surface, ROAD_COLOR, 
                        (0, road_y - 40, VIEWPORT_WIDTH, 80))
        
        # Draw road edge lines
        pygame.draw.line(surface, MARKING_COLOR, 
                        (0, road_y - 40), (VIEWPORT_WIDTH, road_y - 40), 2)
        pygame.draw.line(surface, MARKING_COLOR, 
                        (0, road_y + 40), (VIEWPORT_WIDTH, road_y + 40), 2)
        
        # Draw center dashed line
        for x in range(0, VIEWPORT_WIDTH, 30):
            pygame.draw.line(surface, MARKING_COLOR,
                           (x, road_y), (x + 15, road_y), 2)
        
        # Draw distance markers
        for d in range(0, 1600, 100):
            screen_x = (d - view_start) * scale_x
            if 0 <= screen_x <= VIEWPORT_WIDTH:
                pygame.draw.line(surface, (80, 80, 80), 
                               (int(screen_x), road_y - 50), 
                               (int(screen_x), road_y + 50), 1)
                text = self.font.render(f"{d}m", True, (150, 150, 150))
                surface.blit(text, (int(screen_x) - 15, road_y + 55))
        
        # Draw traffic lights
        for tl_id, tl_data in traffic_lights.items():
            if isinstance(tl_data, dict):
                tl_actor = tl_data.get('actor')
                state = tl_data.get('current_state')
                
                if tl_actor:
                    tl_loc = tl_actor.get_location()
                    screen_x = (tl_loc.x - view_start) * scale_x
                    
                    if 0 <= screen_x <= VIEWPORT_WIDTH:
                        # Traffic light color
                        if state == TrafficLightState.Red:
                            color = (255, 0, 0)
                        elif state == TrafficLightState.Green:
                            color = (0, 255, 0)
                        elif state == TrafficLightState.Yellow:
                            color = (255, 255, 0)
                        else:
                            color = (100, 100, 100)
                        
                        # Draw traffic light pole and signal
                        pygame.draw.line(surface, TL_POLE_COLOR,
                                       (int(screen_x), road_y - 40),
                                       (int(screen_x), road_y - 70), 3)
                        pygame.draw.circle(surface, color, 
                                         (int(screen_x), road_y - 80), 10)
                        
                        # Label
                        label = self.font.render(f"TL{tl_id}", True, color)
                        surface.blit(label, (int(screen_x) - 15, road_y - 100))
        
        # Draw vehicles
        def draw_vehicle_2d(vehicle, color, label=None):
            loc = vehicle.get_location()
            screen_x = (loc.x - view_start) * scale_x
            
            if 0 <= screen_x <= VIEWPORT_WIDTH:
                # Draw vehicle rectangle
                veh_width = 30
                veh_height = 15
                rect = pygame.Rect(int(screen_x) - veh_width//2, 
                                  road_y - veh_height//2,
                                  veh_width, veh_height)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0, 0, 0), rect, 2)
                
                # Draw direction indicator
                pygame.draw.rect(surface, (255, 255, 255),
                               (int(screen_x) + veh_width//2 - 8, road_y - 3, 6, 6))
                
                if label:
                    text = self.font.render(label, True, (255, 255, 255))
                    surface.blit(text, (int(screen_x) - 10, road_y + 20))
        
        # Draw follower vehicles first (so ego appears on top)
        for i, v_data in enumerate(traffic_vehicles):
            draw_vehicle_2d(v_data['vehicle'], FOLLOWER_COLOR, f"F{i+1}")
        
        # Draw ego vehicle
        draw_vehicle_2d(ego_vehicle, EGO_COLOR, "EGO")
        
        # Draw current position info
        pos_text = self.large_font.render(f"Position: {ego_x:.1f}m / 1500m", True, (255, 255, 255))
        surface.blit(pos_text, (10, 10))
        
        # Progress bar
        progress = min(1.0, ego_x / 1500.0)
        bar_width = VIEWPORT_WIDTH - 40
        pygame.draw.rect(surface, (60, 60, 60), (20, 40, bar_width, 20))
        pygame.draw.rect(surface, (0, 200, 0), (20, 40, int(bar_width * progress), 20))
        pygame.draw.rect(surface, (255, 255, 255), (20, 40, bar_width, 20), 2)

    def update(self, ego_vehicle, traffic_vehicles, traffic_lights, tick_info):
        """Update visualization with current simulation state."""
        if not self.running:
            return
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return

        # Update Camera
        self.camera.update_follow(ego_vehicle.get_transform())
        
        # --- 3D RENDER ---
        self.surface_3d.fill(SKY_COLOR)
        
        # Draw ground plane
        pygame.draw.rect(self.surface_3d, GROUND_COLOR, 
                        (0, SCREEN_HEIGHT//2, VIEWPORT_WIDTH, SCREEN_HEIGHT//2))
        
        # Road 3D
        self.draw_road_3d(self.surface_3d)
        
        # Traffic Lights 3D
        for tl_id, tl_data in traffic_lights.items():
            if isinstance(tl_data, dict):
                tl_actor = tl_data.get('actor')
                state = tl_data.get('current_state')
                
                if tl_actor:
                    if state == TrafficLightState.Red:
                        c = (255, 0, 0)
                    elif state == TrafficLightState.Green:
                        c = (0, 255, 0)
                    elif state == TrafficLightState.Yellow:
                        c = (255, 255, 0)
                    else:
                        c = (100, 100, 100)
                    
                    t = tl_actor.get_transform()
                    from .primitives import Transform as T, Location as L, Rotation as R, Vector3D as V
                    
                    # Light (Floating for visibility)
                    light_trans = T(L(t.location.x, t.location.y, t.location.z + 5.0), t.rotation)
                    # Pole
                    pole_trans = T(L(t.location.x, t.location.y, t.location.z + 2.5), t.rotation)
                    self.draw_box_3d(self.surface_3d, pole_trans, V(0.2, 0.2, 2.5), TL_POLE_COLOR)
                    self.draw_box_3d(self.surface_3d, light_trans, V(0.5, 0.5, 0.8), c)

        # Vehicles 3D
        from .primitives import Vector3D
        self.draw_box_3d(self.surface_3d, ego_vehicle.get_transform(), 
                        Vector3D(2.3, 1.0, 0.8), EGO_COLOR)
            
        for v_data in traffic_vehicles:
            v = v_data['vehicle']
            if hasattr(v, 'bounding_box'):
                self.draw_box_3d(self.surface_3d, v.get_transform(), 
                               v.bounding_box.extent, FOLLOWER_COLOR)
            else:
                self.draw_box_3d(self.surface_3d, v.get_transform(), 
                               Vector3D(2.3, 1.0, 0.8), FOLLOWER_COLOR)

        # --- 2D RENDER ---
        self.draw_2d_view(self.surface_2d, ego_vehicle, traffic_vehicles, traffic_lights)

        # --- COMPOSITE ---
        self.screen.blit(self.surface_3d, (0, 0))
        self.screen.blit(self.surface_2d, (VIEWPORT_WIDTH, 0))
        
        # Separator Line
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (VIEWPORT_WIDTH, 0), (VIEWPORT_WIDTH, SCREEN_HEIGHT), 2)

        # Draw HUD
        info_lines = [
            f"Tick: {tick_info['tick']}",
            f"Time: {tick_info['time']:.2f} s",
            f"Speed: {tick_info['speed']*3.6:.1f} km/h",
            "3D Chase View | 2D Top-Down Map"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (0, 0, 0))
            text_s = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_s, (11, 11 + i * 25))
            self.screen.blit(text, (10, 10 + i * 25))

        pygame.display.flip()
        self.clock.tick(30)  # Cap at 30 FPS
        
    def close(self):
        """Clean up pygame."""
        pygame.quit()
