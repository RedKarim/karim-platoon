import pygame
import numpy as np
import math
from .traffic_light import TrafficLightState

# Constants
SCREEN_WIDTH = 2000
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

class Camera3D:
    def __init__(self, fov=60, width=VIEWPORT_WIDTH, height=SCREEN_HEIGHT):
        self.position = np.array([0.0, 0.0, 5.0])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0]) # Z-up
        
        self.width = width
        self.height = height
        self.aspect = width / height
        self.fov_rad = math.radians(fov)
        self.f = 1.0 / math.tan(self.fov_rad / 2)
        
        # Follow parameters
        self.distance = 15.0 # Distance behind
        self.height_offset = 5.0 # Height above
        self.pitch_offset = -0.1 # Look down slightly
        self.smooth_speed = 0.1

    def update_follow(self, vehicle_transform):
        # Extract vehicle data
        v_loc = vehicle_transform.location
        v_rot = vehicle_transform.rotation
        
        # Calculate target position (Vehicle position)
        target_pos = np.array([v_loc.x, v_loc.y, v_loc.z])
        
        # Calculate desired camera position
        # Behind the car: -Forward vector
        yaw_rad = math.radians(v_rot.yaw)
        # Forward vector in XY plane
        forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0])
        
        desired_pos = target_pos - forward * self.distance
        desired_pos[2] += self.height_offset
        
        # Smooth update
        self.position = self.position * (1 - self.smooth_speed) + desired_pos * self.smooth_speed
        
        # Look at the car (slightly ahead of it)
        look_target = target_pos + forward * 5.0
        self.target = look_target

    def get_view_matrix(self):
        # Gram-Schmidt Process for LookAt Matrix
        z_axis = self.position - self.target
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(self.up, z_axis)
        norm_x = np.linalg.norm(x_axis)
        if norm_x < 1e-6: # degraded case (looking straight down/up)
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / norm_x
            
        y_axis = np.cross(z_axis, x_axis)
        
        # View Matrix (4x4)
        view = np.eye(4)
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[0, 3] = -np.dot(x_axis, self.position)
        view[1, 3] = -np.dot(y_axis, self.position)
        view[2, 3] = -np.dot(z_axis, self.position)
        
        # Convert to Pygame Coordinate System (Y down) inside projection or here?
        # Let's keep Standard 3D (Right Handed usually), then Project to Screen.
        return view

    def project_points(self, points_3d):
        # points_3d: (N, 3) array
        num_points = len(points_3d)
        if num_points == 0: return []
        
        # Homogeneous coordinates
        points_4d = np.hstack((points_3d, np.ones((num_points, 1))))
        
        # View Transform
        view_mat = self.get_view_matrix()
        points_view = points_4d @ view_mat.T # (N, 4)
        
        # Filter points behind camera (z > 0 in view space usually means behind if Camera looks down -Z)
        # Our View Matrix construction: Z is vector from Target to Camera. So Camera is at origin, Target is at -Z.
        # So points with negative Z in view space are in front.
        # NEAR PLANE CULLING
        # Check Z < -NEAR_PLANE
        
        # Perspective Project params
        # x' = x / -z * f / aspect
        # y' = y / -z * f
        
        # Vectorized projection
        z_vals = points_view[:, 2]
        valid_mask = z_vals < -NEAR_PLANE
        
        # We process all, but will filter later or set to None
        projected = np.zeros((num_points, 2))
        
        # Avoid division by zero
        # z_vals[~valid_mask] = -0.0001
        
        # Project
        # Screen Space x: [-1, 1], y: [-1, 1]
        # x_ndc = x / -z * f / aspect
        # y_ndc = y / -z * f
        
        # We iterate to handle masking simpler or just operate on valid
        
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None, valid_mask
            
        p_valid = points_view[valid_indices]
        x = p_valid[:, 0]
        y = p_valid[:, 1]
        z = p_valid[:, 2] # Negative
        
        x_ndc = (x / -z) * (self.f / self.aspect)
        y_ndc = (y / -z) * self.f
        
        # Map to Screen: 
        # x_screen = (x_ndc + 1) * width / 2
        # y_screen = (1 - y_ndc) * height / 2  (Flip Y for screen coords)
        
        screen_x = (x_ndc + 1.0) * self.width * 0.5
        screen_y = (1.0 - y_ndc) * self.height * 0.5
        
        projected[valid_indices, 0] = screen_x
        projected[valid_indices, 1] = screen_y
        
        return projected, valid_mask


class Visualizer:
    def __init__(self, waypoints):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("EcoLead Simulation (Split Screen 3D + 2D)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.waypoints = waypoints
        self.running = True
        
        # Surfaces
        self.surface_3d = pygame.Surface((VIEWPORT_WIDTH, SCREEN_HEIGHT))
        self.surface_2d = pygame.Surface((VIEWPORT_WIDTH, SCREEN_HEIGHT))
        
        # 3D Setup
        self.road_width = 7.5 # meters
        self.road_geometry = self._build_road_mesh() # (N, 4, 3) 
        self.camera = Camera3D(width=VIEWPORT_WIDTH, height=SCREEN_HEIGHT)
        
        # 2D Setup (Pre-calc scale)
        min_x = min(w.x for w in self.waypoints)
        max_x = max(w.x for w in self.waypoints)
        min_y = min(w.y for w in self.waypoints)
        max_y = max(w.y for w in self.waypoints)
        
        margin = 30
        track_width = max_x - min_x
        track_height = max_y - min_y
        
        scale_x = (VIEWPORT_WIDTH - 2 * margin) / track_width if track_width > 0 else 1.0
        scale_y = (SCREEN_HEIGHT - 2 * margin) / track_height if track_height > 0 else 1.0
        self.scale_2d = min(scale_x, scale_y)
        
        self.offset_x_2d = (VIEWPORT_WIDTH - track_width * self.scale_2d) / 2 - min_x * self.scale_2d
        self.offset_y_2d = (SCREEN_HEIGHT - track_height * self.scale_2d) / 2 - min_y * self.scale_2d
        self.min_x = min_x
        self.min_y = min_y
        
        # Pre-calc 2D road points
        self.road_points_2d = [self.world_to_screen_2d(w.x, w.y) for w in self.waypoints]

    def world_to_screen_2d(self, x, y):
        # Map X->Right, Y->Up (Inverted Screen Y)
        # Using pre-calced offset suitable for viewport width 
        margin = 30
        sx = margin + (x - self.min_x) * self.scale_2d 
        sy = SCREEN_HEIGHT - margin - (y - self.min_y) * self.scale_2d
        
        # Recalculate centering exactly as Init did or use stored offset? 
        # The stored offset self.offset_x_2d might be safer if we want exact centering.
        # Let's match init logic logic:
        sx = self.offset_x_2d + x * self.scale_2d
        # Y needs flip relative to bounds?
        # max_y maps to margin?
        # Let's simple invert:
        # sy = self.offset_y_2d + y * self.scale_2d # This would be Non-inverted Y (World Y down)
        # We want World Y up.
        # sy = offset + (max_y - y) * scale ?
        
        # Let's stick to the logic that worked before but mapped to Viewport
        sy = SCREEN_HEIGHT - 30 - (y - self.min_y) * self.scale_2d
        # We should probably center Y better.
        # If we use the centered offset_y_2d computed in init (which assumed inverted?), let's use it.
        # Actually previous logic calculated offset based on track_height. 
        # Let's just use:
        # sx = int(self.offset_x_2d + x * self.scale_2d)
        # sy = int(SCREEN_HEIGHT - (self.offset_y_2d + y * self.scale_2d)) 
        # Wait, min_y should map to bottom...
        
        sx = int(self.offset_x_2d + x * self.scale_2d)
        sy = int(SCREEN_HEIGHT - (self.offset_y_2d + (y - self.min_y) * self.scale_2d) - self.min_y*self.scale_2d) # Error prone
        
        # Reliable way:
        sx = int(30 + (x - self.min_x) * self.scale_2d) # Just left align with margin? No we centered x.
        # Let's use the explicit calculated offsets 
        sx = int(self.offset_x_2d + x * self.scale_2d)
        # Invert Y: 
        # y_from_bottom = (y - self.min_y) * self.scale_2d
        # sy = SCREEN_HEIGHT - self.offset_y_2d - y_from_bottom ?
        # Actually let's just do standard mapping:
        sy = int(SCREEN_HEIGHT - 30 - (y - self.min_y) * self.scale_2d)
        return (sx, sy)

    def _build_road_mesh(self):
        # Convert waypoints to numpy array
        # Waypoints are center points. We need Left and Right edges.
        road_quads = []
        
        # Iterate waypoints
        # We need direction to calculate normal
        
        wps = np.array([[w.x, w.y, w.z] for w in self.waypoints])
        num = len(wps)
        
        # Loop closed? ECOLead route is circular.
        
        half_w = self.road_width / 2.0
        
        left_edge = []
        right_edge = []
        
        for i in range(num):
            p0 = wps[i]
            p1 = wps[(i+1)%num]
            
            # Forward vector
            fwd = p1 - p0
            dist = np.linalg.norm(fwd)
            if dist < 0.001: continue
            fwd /= dist
            
            # Up vector (assume Z up)
            up = np.array([0, 0, 1])
            
            # Right vector
            right = np.cross(fwd, up)
            right /= np.linalg.norm(right)
            
            l = p0 - right * half_w
            r = p0 + right * half_w
            
            left_edge.append(l)
            right_edge.append(r)
            
        # Build Quads
        # Quad i: Left[i], Right[i], Right[i+1], Left[i+1]
        count = len(left_edge)
        geometry = []
        for i in range(count):
            next_i = (i+1)%count
            # Don't connect last to first if distance is huge (not circular loop case handle?)
            # Assuming circular for now
            if np.linalg.norm(wps[i] - wps[next_i]) > 20: continue # Skip wrap if gap
            
            quad = np.array([
                left_edge[i],
                right_edge[i],
                right_edge[next_i],
                left_edge[next_i]
            ])
            geometry.append(quad)
            
        return np.array(geometry)

    def draw_road(self, surface):
        # Project all road points?
        # Optimization: Only project points near camera? 
        # For now, brute force project all vertices? Might be heavy (5000 * 4 points)
        # Let's cull based on distance to camera first.
        
        cam_pos = self.camera.position
        
        # Compute centers of quads
        # We can store this.
        # Simplified: Check distance of first vertex of quad
        
        # 1. Slice visible road
        # Ideally we know which index the car is at, and draw +/- N segments.
        # But here we just filter by distance.
        centers = np.mean(self.road_geometry, axis=1) # (M, 3)
        dists = np.linalg.norm(centers - cam_pos, axis=1)
        
        # Draw only within 150m
        visible_indices = np.where(dists < 200)[0]
        
        visible_quads = self.road_geometry[visible_indices]
        
        # Project
        # Reshape to list of points (M*4, 3)
        M = len(visible_quads)
        if M == 0: return
        
        points_flat = visible_quads.reshape(-1, 3)
        projected, mask = self.camera.project_points(points_flat)
        
        if projected is None: return
        
        # Reshape back and draw
        # mask is (M*4,)
        
        # We need to draw quads where ALL 4 points are valid? Or at least some clipped?
        # Simple approach: If all 4 valid, draw.
        
        mask_quads = mask.reshape(M, 4)
        valid_quads = np.all(mask_quads, axis=1)
        
        quads_2d = projected.reshape(M, 4, 2)
        
        final_quads = quads_2d[valid_quads]
        
        # Painter's Algo: Sort by distance (furthest first)
        # Actually standard Z-buffer is hard in Pygame. 
        # Drawing road furthest to closest is better.
        # dists is already calculated.
        
        valid_indices_sorted = visible_indices[valid_quads]
        sorted_order = np.argsort(dists[valid_indices_sorted])[::-1] # Descending distance
        
        # Draw
        for idx in sorted_order:
            poly = final_quads[idx]
            pygame.draw.polygon(self.screen, ROAD_COLOR, poly)
            # Edge lines
            pygame.draw.lines(surface, (100,100,100), True, poly, 1)

    def draw_box(self, surface, transform, extent, color):
        # Create 8 corners of bounding box in world space
        # Extent is half-size
        l = extent.x
        w = extent.y
        h = extent.z
        
        corners = np.array([
            [l, w, -h], [l, -w, -h], [-l, -w, -h], [-l, w, -h], # Bottom
            [l, w, h], [l, -w, h], [-l, -w, h], [-l, w, h]    # Top
        ])
        # Z is usually up, so -h to h relative to center z?
        # transform.location is usually center of mass/ground? 
        # CARLA: Z is center. So -h to h is correct.
        
        # Rotate and Translate
        yaw_rad = math.radians(transform.rotation.yaw)
        pitch_rad = math.radians(transform.rotation.pitch)
        roll_rad = math.radians(transform.rotation.roll)
        
        # Rotation Matrix (Yaw only for simplicity, or full R)
        # Yaw (around Z)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        R = np.array([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ])
        
        rotated_corners = corners @ R.T
        
        # Translate
        world_corners = rotated_corners + np.array([transform.location.x, transform.location.y, transform.location.z])
        
        # Project
        projected, mask = self.camera.project_points(world_corners)
        
        if projected is None or not np.all(mask): return # Skip if clipped
        
        # Draw Faces
        # Bottom: 0,1,2,3
        # Top: 4,5,6,7
        # Sides: 0,1,5,4; 1,2,6,5; 2,3,7,6; 3,0,4,7
        
        faces = [
            [0,1,2,3], [4,5,6,7],
            [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]
        ]
        
        
        for face in faces:
            poly = projected[face]
            pygame.draw.polygon(surface, color, poly)
            pygame.draw.polygon(surface, (0,0,0), poly, 1) # Outline

    def draw_2d_view(self, surface):
        surface.fill(BACKGROUND_COLOR) # clear
        # Draw Road
        if len(self.road_points_2d) > 1:
            pygame.draw.lines(surface, ROAD_COLOR, False, self.road_points_2d, 5)
            pygame.draw.lines(surface, MARKING_COLOR, False, self.road_points_2d, 1)
            
        # We need access to vehicles and lights... passed in update?
        # We will split update logic.

    def update(self, ego_vehicle, traffic_vehicles, traffic_lights, tick_info):
        if not self.running: return
        
        # Event Loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return

        # Update Camera
        self.camera.update_follow(ego_vehicle.get_transform())
        
        # --- 3D RENDER ---
        self.surface_3d.fill(SKY_COLOR)
        
        # Road 3D
        self.draw_road(self.surface_3d)
        
        # Traffic Lights 3D
        for tl_id, tl_data in traffic_lights.items():
             if isinstance(tl_data, dict):
                tl_actor = tl_data.get('actor')
                state = tl_data.get('current_state')
             else: continue
             
             if tl_actor:
                 c = (50, 50, 50)
                 if state == TrafficLightState.Red: c = (255, 0, 0)
                 elif state == TrafficLightState.Green: c = (0, 255, 0)
                 elif state == TrafficLightState.Yellow: c = (255, 255, 0)
                 
                 t = tl_actor.get_transform()
                 from .primitives import Transform as T, Location as L, Rotation as R, Vector3D as V
                 
                 # Light (Floating for visibility)
                 light_trans = T(L(t.location.x, t.location.y, t.location.z + 4.0), t.rotation)
                 # Pole
                 pole_trans = T(L(t.location.x, t.location.y, t.location.z + 2), t.rotation)
                 self.draw_box(self.surface_3d, pole_trans, V(0.2, 0.2, 2.0), (100,100,100))
                 self.draw_box(self.surface_3d, light_trans, V(0.5, 0.5, 0.8), c)

        # Vehicles 3D
        if hasattr(ego_vehicle, 'bounding_box'):
            self.draw_box(self.surface_3d, ego_vehicle.get_transform(), ego_vehicle.bounding_box.extent, EGO_COLOR)
        else: 
            from .primitives import Vector3D
            self.draw_box(self.surface_3d, ego_vehicle.get_transform(), Vector3D(2.3, 1.0, 0.8), EGO_COLOR)
            
        for v_data in traffic_vehicles:
            v = v_data['vehicle']
            if hasattr(v, 'bounding_box'):
                self.draw_box(self.surface_3d, v.get_transform(), v.bounding_box.extent, FOLLOWER_COLOR)

        # --- 2D RENDER ---
        self.surface_2d.fill(BACKGROUND_COLOR)
        
        # Road 2D
        if len(self.road_points_2d) > 1:
            pygame.draw.lines(self.surface_2d, ROAD_COLOR, False, self.road_points_2d, 10)
            pygame.draw.lines(self.surface_2d, (200,200,200), False, self.road_points_2d, 1)

        # Traffic Lights 2D
        for tl_id, tl_data in traffic_lights.items():
            if isinstance(tl_data, dict):
                tl_actor = tl_data.get('actor')
                state = tl_data.get('current_state')
            else: continue
            if tl_actor:
                 c = (100, 100, 100)
                 if state == TrafficLightState.Red: c = (255, 0, 0)
                 elif state == TrafficLightState.Green: c = (0, 255, 0)
                 elif state == TrafficLightState.Yellow: c = (255, 255, 0)
                 
                 loc = tl_actor.get_location()
                 sx, sy = self.world_to_screen_2d(loc.x, loc.y)
                 pygame.draw.circle(self.surface_2d, c, (sx, sy), 6)
        
        # Vehicles 2D
        def draw_veh_2d(vehicle, color):
            loc = vehicle.get_location()
            sx, sy = self.world_to_screen_2d(loc.x, loc.y)
            yaw = vehicle.get_transform().rotation.yaw
            
            # Simple rect rotated
            w, h = 10, 20 # Screen pixels approx
            surf = pygame.Surface((h, w), pygame.SRCALPHA)
            surf.fill(color)
            pygame.draw.rect(surf, (0,0,0), (int(h*0.7), 0, int(h*0.3), w)) # Head
            
            rot_surf = pygame.transform.rotate(surf, -yaw)
            rect = rot_surf.get_rect(center=(sx, sy))
            self.surface_2d.blit(rot_surf, rect)

        draw_veh_2d(ego_vehicle, EGO_COLOR)
        for v_data in traffic_vehicles:
            draw_veh_2d(v_data['vehicle'], FOLLOWER_COLOR)


        # --- COMPOSITE ---
        self.screen.blit(self.surface_3d, (0, 0))
        self.screen.blit(self.surface_2d, (VIEWPORT_WIDTH, 0))
        
        # Separator Line
        pygame.draw.line(self.screen, (255, 255, 255), (VIEWPORT_WIDTH, 0), (VIEWPORT_WIDTH, SCREEN_HEIGHT), 2)

        # Draw HUD (Overall)
        info_lines = [
            f"Tick: {tick_info['tick']}",
            f"Time: {tick_info['time']:.2f} s",
            f"Speed: {tick_info['speed']*3.6:.1f} km/h",
            "Left: 3D Chase | Right: 2D Map"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (0, 0, 0)) # Black text on Sky
            # Drop shadow
            text_s = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_s, (11, 11 + i * 25))
            self.screen.blit(text, (10, 10 + i * 25))

        pygame.display.flip()
        
    def close(self):
        pygame.quit()
