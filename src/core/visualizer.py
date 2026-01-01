import pygame
import math
import numpy as np
from .traffic_light import TrafficLightState

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SCALE = 1.6 # Dynamic
OFFSET_X = 0
OFFSET_Y = 0

# Colors
ROAD_COLOR = (50, 50, 50)
MARKING_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (30, 30, 30)
EGO_COLOR = (0, 100, 255)
FOLLOWER_COLOR = (255, 100, 100)
TEXT_COLOR = (200, 200, 200)

class Visualizer:
    def __init__(self, waypoints):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("EcoLead Simulation (Karim-Platoon)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.waypoints = waypoints
        self.running = True
        
        # Calculate bounds and scale
        min_x = min(w.x for w in self.waypoints)
        max_x = max(w.x for w in self.waypoints)
        min_y = min(w.y for w in self.waypoints)
        max_y = max(w.y for w in self.waypoints)
        
        margin = 50
        track_width = max_x - min_x
        track_height = max_y - min_y
        
        scale_x = (SCREEN_WIDTH - 2 * margin) / track_width if track_width > 0 else 1.0
        scale_y = (SCREEN_HEIGHT - 2 * margin) / track_height if track_height > 0 else 1.0
        self.scale = min(scale_x, scale_y)
        
        # Center the track
        self.offset_x = (SCREEN_WIDTH - track_width * self.scale) / 2 - min_x * self.scale
        self.offset_y = (SCREEN_HEIGHT - track_height * self.scale) / 2 - min_y * self.scale # Y is inverted? No, let's keep simple transform
        
        # Actually in Pygame Y is down. World Y is likely Up.
        # Let's map min_y to SCREEN_HEIGHT - margin, max_y to margin
        # world_y = min_y -> screen_y = max_screen_y
        # screen_y = SCREEN_HEIGHT - margin - (y - min_y) * scale
        
        self.min_x = min_x
        self.min_y = min_y

    def world_to_screen(self, x, y):
        sx = int(self.offset_x + x * self.scale)
        # Flip Y:
        # We wanted centered. 
        # let's use the standard formula computed in init? 
        # actually let's redo the math in world_to_screen using stored min/max to be safe/clear
        
        # X: min_x maps to margin, max_x maps to width-margin
        # Y: min_y maps to height-margin, max_y maps to margin (flipped)
        
        margin = 50
        sx = margin + (x - self.min_x) * self.scale
        sy = SCREEN_HEIGHT - margin - (y - self.min_y) * self.scale
        return (sx, sy)

    def draw_vehicle(self, vehicle, color):
        loc = vehicle.get_location()
        sx, sy = self.world_to_screen(loc.x, loc.y)
        
        # Get yaw to rotate vehicle rect
        yaw = vehicle.get_transform().rotation.yaw # degrees
        # Pygame rotation is counter-clockwise?
        # Simulation Yaw: 0 is X positive (Right).
        # Screen X is Right. Screen Y is Up (inverted).
        # We need to test rotation direction.
        
        # Create a surface for the car
        car_width = 2.0 * self.scale
        car_length = 4.5 * self.scale
        
        # Create a surface with alpha channel
        car_surf = pygame.Surface((int(car_length), int(car_width)), pygame.SRCALPHA)
        car_surf.fill(color) # Fill with color
        
        # Add a "windshield" to indicate direction
        pygame.draw.rect(car_surf, (0,0,0), (int(car_length*0.6), 0, int(car_length*0.2), int(car_width)))
        
        # Rotate
        # Math angle: 0 is X+. Pygame rotate is CCW.
        # Screen Y is flipped, so maybe rotation behaves differently.
        # Let's try direct mapping first.
        # Standard: angle increases CCW.
        # If Yaw is degrees CCW from X, then we just rotate by yaw.
        # Since we flip Y in world_to_screen, we might need to negate angle.
        rotated_surf = pygame.transform.rotate(car_surf, -yaw) 
        
        rect = rotated_surf.get_rect(center=(sx, sy))
        self.screen.blit(rotated_surf, rect)

    def update(self, ego_vehicle, traffic_vehicles, traffic_lights, tick_info):
        if not self.running:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    pygame.quit()
                    return

        # Center camera on Ego - OPTIONAL
        # If we want to see the WHOLE road as an oval, we should NOT center on ego, but keep fixed view.
        # The user asked for "oval shape road", implying seeing the track.
        # Let's disable camera tracking for now to show the full shape.
        
        # ego_loc = ego_vehicle.get_location()
        # global OFFSET_X, OFFSET_Y
        # OFFSET_X = SCREEN_WIDTH // 2 - ego_loc.x * SCALE
        # OFFSET_Y = SCREEN_HEIGHT // 2 + ego_loc.y * SCALE

        self.screen.fill(BACKGROUND_COLOR)

        # Draw Road (Waypoints)
        if len(self.waypoints) > 1:
            points = [self.world_to_screen(w.x, w.y) for w in self.waypoints]
            
            # Draw Asphalt
            pygame.draw.lines(self.screen, ROAD_COLOR, False, points, 20)
            
            # Draw Dashed Line
            # This is expensive to do segment by segment for 5000 points.
            # Let's sub-sample or just draw a solid thin line for now, or dashed carefully.
            # Solid center line for simplicity but good look
            pygame.draw.lines(self.screen, MARKING_COLOR, False, points, 1)

        # Draw Traffic Lights
        for tl_id, tl_data in traffic_lights.items():
            # Support both object and dict representation from TL manager
            if isinstance(tl_data, dict):
                tl_actor = tl_data.get('actor')
                state = tl_data.get('current_state')
            else:
                tl_actor = tl_data
                state = TrafficLightState.Unknown

            if tl_actor:
                loc = tl_actor.get_location()
                sx, sy = self.world_to_screen(loc.x, loc.y)
                
                # Determine Color
                c = (100, 100, 100) # Off/Gray
                if state == TrafficLightState.Red: c = (255, 0, 0)
                elif state == TrafficLightState.Green: c = (0, 255, 0)
                elif state == TrafficLightState.Yellow: c = (255, 255, 0)
                
                # Draw Pole
                # pygame.draw.line(self.screen, (150,150,150), (sx, sy), (sx+10, sy), 2)
                
                # Draw Light Box
                pygame.draw.circle(self.screen, (0,0,0), (sx, sy), 8)
                pygame.draw.circle(self.screen, c, (sx, sy), 6)
                
                # Draw ID
                label = self.font.render(str(tl_id), True, (255, 255, 255))
                self.screen.blit(label, (sx+10, sy-10))

        # Draw Vehicles
        self.draw_vehicle(ego_vehicle, EGO_COLOR)
        
        for v_data in traffic_vehicles:
            self.draw_vehicle(v_data['vehicle'], FOLLOWER_COLOR) # Changed to FOLLOWER_COLOR

        # Draw HUD
        info_lines = [
            f"Tick: {tick_info['tick']}",
            f"Time: {tick_info['time']:.2f} s",
            f"Ego Speed: {tick_info['speed']:.2f} m/s",
            f"Offset: {OFFSET_X:.0f}, {OFFSET_Y:.0f}"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()
        
        # Limit FPS?
        # self.clock.tick(60)

    def close(self):
        if self.running:
            pygame.quit()
            self.running = False
