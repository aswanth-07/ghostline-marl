from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

from neon_arena.geometry import ray_rect_distance

from neon_arena.config import (
    CORE_RADIUS,
    DASH_ENERGY_MAX,
    EMP_DURATION_STEPS,
    EMP_RADIUS,
    MAX_STEPS,
    PORTAL_RADIUS,
    RAY_LENGTH,
    WORLD_HEIGHT,
    WORLD_WIDTH,
)

SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 800
ARENA_LEFT = 24
ARENA_TOP = 58
ARENA_WIDTH = 1068
ARENA_HEIGHT = 704
PANEL_LEFT = 1122
PANEL_WIDTH = 292

INK = (219, 244, 255)
MUTED = (112, 148, 166)
CYAN = (49, 228, 255)
MAGENTA = (255, 56, 179)
GOLD = (255, 209, 82)
LIME = (125, 255, 159)
RED = (255, 82, 106)
ORANGE = (255, 132, 66)


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    life: float
    color: tuple[int, int, int]
    radius: float


class ArenaRenderer:
    def __init__(self, env: Any):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("CORE RUNNER // RL ARENA")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.env = env
        self.running = True
        self.reset_requested = False
        self.frame = 0
        self.random = random.Random(17)
        self.particles: list[Particle] = []
        self.trail: deque[tuple[float, float]] = deque(maxlen=26)
        self.shake = 0.0
        self.camera_offset = (0.0, 0.0)
        self.font_xs = pygame.font.SysFont("consolas", 15)
        self.font_sm = pygame.font.SysFont("consolas", 18)
        self.font_md = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 31, bold=True)
        self.scale = min(ARENA_WIDTH / WORLD_WIDTH, ARENA_HEIGHT / WORLD_HEIGHT)
        self.world_width = WORLD_WIDTH * self.scale
        self.world_height = WORLD_HEIGHT * self.scale
        self.world_left = ARENA_LEFT + (ARENA_WIDTH - self.world_width) * 0.5
        self.world_top = ARENA_TOP + (ARENA_HEIGHT - self.world_height) * 0.5
        self.show_districts_debug = False
        self.show_drone_debug = False
        self.alarm_surface = pygame.Surface((ARENA_WIDTH, ARENA_HEIGHT), pygame.SRCALPHA)

    def human_action(self) -> tuple[np.ndarray, bool, bool]:
        self._consume_events(allow_reset=True)
        keys = pygame.key.get_pressed()
        x_axis = float(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_a] or keys[pygame.K_LEFT])
        y_axis = float(keys[pygame.K_s] or keys[pygame.K_DOWN]) - float(keys[pygame.K_w] or keys[pygame.K_UP])
        boost = 1.0 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else -1.0
        emp = 1.0 if keys[pygame.K_SPACE] else -1.0
        reset_requested = self.reset_requested
        self.reset_requested = False
        return np.array([x_axis, y_axis, boost, emp], dtype=np.float32), reset_requested, self.running

    def draw(
        self,
        stats: dict[str, Any] | None = None,
        *,
        process_events: bool = True,
        limit_fps: bool = True,
    ) -> bool:
        if process_events:
            self._consume_events(allow_reset=False)
        self.frame += 1
        
        # Dynamically adjust scale and bounds based on current environment size
        self.scale = min(ARENA_WIDTH / self.env.world_width, ARENA_HEIGHT / self.env.world_height)
        self.world_width = self.env.world_width * self.scale
        self.world_height = self.env.world_height * self.scale
        self.world_left = ARENA_LEFT + (ARENA_WIDTH - self.world_width) * 0.5
        self.world_top = ARENA_TOP + (ARENA_HEIGHT - self.world_height) * 0.5
        
        self.trail.append(tuple(float(value) for value in self.env.player))
        self._add_event_particles()
        self._update_particles()
        self._update_camera_feedback()

        self.screen.fill((4, 9, 17))
        self._draw_background()
        self._draw_hazards()
        if getattr(self, "show_districts_debug", False):
            self._draw_districts_debug()
        self._draw_objective_path()
        self._draw_blocks()
        self._draw_gates()
        self._draw_extraction_portal()
        self._draw_terminal()
        self._draw_cores()
        self._draw_loot()
        self._draw_drone_routes()
        self._draw_drone_vision()
        self._draw_cameras()
        self._draw_drones()
        if getattr(self, "show_drone_debug", False):
            self._draw_drone_debug()
        self._draw_projectiles()
        self._draw_player()
        self._draw_particles()
        self._draw_minimap()
        self._draw_header()
        self._draw_panel(stats or {})
        self._draw_result_overlay()
        pygame.display.flip()
        if limit_fps:
            self.clock.tick(48)
        return self.running

    def close(self) -> None:
        pygame.quit()

    def _consume_events(self, *, allow_reset: bool) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif allow_reset and event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.reset_requested = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.show_districts_debug = not self.show_districts_debug
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.show_drone_debug = not self.show_drone_debug

    def _world(self, point: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        return (
            round(self.world_left + self.camera_offset[0] + float(point[0]) * self.scale),
            round(self.world_top + self.camera_offset[1] + float(point[1]) * self.scale),
        )

    def _draw_background(self) -> None:
        if self.env.alarm_active:
            pulse = 28 + round(16 * math.sin(self.frame * 0.18))
            self.alarm_surface.fill((255, 30, 66, pulse))
            self.screen.blit(self.alarm_surface, (ARENA_LEFT, ARENA_TOP))
        pygame.draw.rect(
            self.screen,
            (7, 18, 28),
            pygame.Rect(round(self.world_left), round(self.world_top), round(self.world_width), round(self.world_height)),
        )
        for x in range(0, round(self.env.world_width) + 1, 40):
            start = self._world((x, 0))
            end = self._world((x, self.env.world_height))
            pygame.draw.line(self.screen, (12, 39, 54), start, end, 1)
        for y in range(0, round(self.env.world_height) + 1, 40):
            start = self._world((0, y))
            end = self._world((self.env.world_width, y))
            pygame.draw.line(self.screen, (12, 39, 54), start, end, 1)

        border = pygame.Rect(round(self.world_left), round(self.world_top), round(self.world_width), round(self.world_height))
        border_color = RED if self.env.alarm_active and self.frame % 28 < 14 else (24, 90, 112)
        pygame.draw.rect(self.screen, border_color, border, 2)

    def _draw_objective_path(self) -> None:
        start = self._world(self.env.player)
        end = self._world(self.env.objective)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = max(1.0, math.hypot(dx, dy))
        segments = max(1, int(distance // 22))
        color = GOLD if self.env.cores_collected >= len(self.env.cores) else CYAN
        for index in range(0, segments, 2):
            first = index / segments
            second = min(1.0, (index + 0.72) / segments)
            point_a = (round(start[0] + dx * first), round(start[1] + dy * first))
            point_b = (round(start[0] + dx * second), round(start[1] + dy * second))
            pygame.draw.line(self.screen, (*color, 100), point_a, point_b, 1)

    def _draw_blocks(self) -> None:
        # Pass 1: Glow
        for block in self.env.city_blocks:
            is_breakable = getattr(self.env, "breakable_walls", None) and block in self.env.breakable_walls
            if is_breakable:
                continue
                
            rect = pygame.Rect(
                round(self.world_left + block.x * self.scale),
                round(self.world_top + block.y * self.scale),
                round(block.width * self.scale),
                round(block.height * self.scale),
            )
            
            # Optimized local glow surface to avoid full-screen surface creation lag
            glow_surf = pygame.Surface((rect.width + 24, rect.height + 24), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (23, 182, 218, 28), pygame.Rect(6, 6, rect.width + 12, rect.height + 12), 5, border_radius=5)
            self.screen.blit(glow_surf, (rect.x - 12, rect.y - 12))

        # Pass 2: Borders & Highlights
        for block in self.env.city_blocks:
            is_breakable = getattr(self.env, "breakable_walls", None) and block in self.env.breakable_walls
            if is_breakable:
                continue
                
            rect = pygame.Rect(
                round(self.world_left + block.x * self.scale),
                round(self.world_top + block.y * self.scale),
                round(block.width * self.scale),
                round(block.height * self.scale),
            )
            
            pygame.draw.rect(self.screen, (31, 112, 133), rect, 2, border_radius=4)
            
            # Check if there is a block directly above this one
            has_block_above = False
            for other in self.env.city_blocks:
                if other == block:
                    continue
                if getattr(self.env, "breakable_walls", None) and other in self.env.breakable_walls:
                    continue
                if abs((other.y + other.height) - block.y) < 1.0:
                    if other.x < block.x + block.width - 1.0 and other.x + other.width > block.x + 1.0:
                        has_block_above = True
                        break
            
            if not has_block_above:
                pygame.draw.line(self.screen, (64, 221, 238), rect.topleft, rect.topright, 2)

        # Pass 3: Solid Interiors (deflated by 2)
        for block in self.env.city_blocks:
            is_breakable = getattr(self.env, "breakable_walls", None) and block in self.env.breakable_walls
            if is_breakable:
                continue
                
            rect = pygame.Rect(
                round(self.world_left + block.x * self.scale),
                round(self.world_top + block.y * self.scale),
                round(block.width * self.scale),
                round(block.height * self.scale),
            )
            
            inner_rect = rect.inflate(-2, -2)
            if inner_rect.width > 0 and inner_rect.height > 0:
                pygame.draw.rect(self.screen, (9, 24, 35), inner_rect, border_radius=3)

        # Pass 3.5: Cover shared boundaries to make wall intersections seamless
        for i, block_a in enumerate(self.env.city_blocks):
            if getattr(self.env, "breakable_walls", None) and block_a in self.env.breakable_walls:
                continue
            for block_b in self.env.city_blocks[i+1:]:
                if getattr(self.env, "breakable_walls", None) and block_b in self.env.breakable_walls:
                    continue
                
                # Check horizontal adjacency: block_a left touches block_b right, or vice versa
                shares_vert_wall = False
                coord = 0.0
                if abs((block_a.x + block_a.width) - block_b.x) < 1.0:
                    shares_vert_wall = True
                    coord = block_a.x + block_a.width
                elif abs((block_b.x + block_b.width) - block_a.x) < 1.0:
                    shares_vert_wall = True
                    coord = block_b.x + block_b.width
                    
                if shares_vert_wall:
                    y_start = max(block_a.y, block_b.y)
                    y_end = min(block_a.y + block_a.height, block_b.y + block_b.height)
                    if y_end - y_start > 1.0:
                        cov_rect = pygame.Rect(
                            round(self.world_left + (coord - 2) * self.scale),
                            round(self.world_top + y_start * self.scale),
                            max(1, round(4 * self.scale)),
                            round((y_end - y_start) * self.scale)
                        )
                        pygame.draw.rect(self.screen, (9, 24, 35), cov_rect)
                        
                # Check vertical adjacency: block_a bottom touches block_b top, or vice versa
                shares_horiz_wall = False
                if abs((block_a.y + block_a.height) - block_b.y) < 1.0:
                    shares_horiz_wall = True
                    coord = block_a.y + block_a.height
                elif abs((block_b.y + block_b.height) - block_a.y) < 1.0:
                    shares_horiz_wall = True
                    coord = block_b.y + block_b.height
                    
                if shares_horiz_wall:
                    x_start = max(block_a.x, block_b.x)
                    x_end = min(block_a.x + block_a.width, block_b.x + block_b.width)
                    if x_end - x_start > 1.0:
                        cov_rect = pygame.Rect(
                            round(self.world_left + x_start * self.scale),
                            round(self.world_top + (coord - 2) * self.scale),
                            round((x_end - x_start) * self.scale),
                            max(1, round(4 * self.scale))
                        )
                        pygame.draw.rect(self.screen, (9, 24, 35), cov_rect)

        # Pass 4: Breakable walls and active charges (drawn on top)
        for block in self.env.city_blocks:
            is_breakable = getattr(self.env, "breakable_walls", None) and block in self.env.breakable_walls
            if not is_breakable:
                continue
                
            rect = pygame.Rect(
                round(self.world_left + block.x * self.scale),
                round(self.world_top + block.y * self.scale),
                round(block.width * self.scale),
                round(block.height * self.scale),
            )
            
            if rect.width > 0 and rect.height > 0:
                temp_surf = pygame.Surface((rect.width, rect.height))
                temp_surf.fill((20, 20, 20))
                stripe_color = (255, 209, 82)
                for offset in range(-rect.height, rect.width + 10, 15):
                    pygame.draw.line(temp_surf, stripe_color, (offset, 0), (offset + rect.height, rect.height), 4)
                self.screen.blit(temp_surf, rect.topleft)

            if getattr(self.env, "active_charge_wall", None) == block:
                timer = getattr(self.env, "active_charge_timer", 0)
                blink = (timer // 6) % 2 == 0
                if blink:
                    overlay_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    overlay_surf.fill((255, 0, 0, 120))
                    self.screen.blit(overlay_surf, rect.topleft)
                
                charge_center = self._world(self.env.active_charge_pos)
                pygame.draw.rect(self.screen, (0, 0, 0), (charge_center[0] - 12, charge_center[1] - 12, 24, 24))
                pygame.draw.rect(self.screen, (255, 0, 0), (charge_center[0] - 12, charge_center[1] - 12, 24, 24), 2)
                if (self.frame // 10) % 2 == 0:
                    pygame.draw.circle(self.screen, (255, 0, 0), charge_center, 6)
                else:
                    pygame.draw.circle(self.screen, (150, 0, 0), charge_center, 4)
                
                secs = timer / 60.0
                secs_text = self.font_xs.render(f"{secs:.2f}s", True, (255, 255, 255))
                self.screen.blit(secs_text, (charge_center[0] - secs_text.get_width() // 2, charge_center[1] - 30))

            pygame.draw.rect(self.screen, (255, 209, 82), rect, 2, border_radius=4)

    def _draw_extraction_portal(self) -> None:
        center = self._world(self.env.extraction_point)
        active = self.env.cores_collected >= len(self.env.cores)
        color = GOLD if active else (74, 92, 100)
        pulse = 1.0 + 0.16 * math.sin(self.frame * 0.11)
        for index in range(4):
            radius = round((PORTAL_RADIUS + index * 9) * self.scale * pulse)
            pygame.draw.circle(self.screen, color, center, radius, 2)
        if active:
            self._draw_glow_circle(center, round(PORTAL_RADIUS * self.scale), GOLD)
        label = self.font_xs.render("EXTRACTION", True, color)
        self.screen.blit(label, (center[0] - label.get_width() // 2, center[1] + 40))

    def _draw_terminal(self) -> None:
        center = self._world(self.env.terminal_point)
        if self.env.emp_timer > 0:
            progress = self.env.emp_timer / EMP_DURATION_STEPS
            radius = round((EMP_RADIUS * (1.0 - progress * 0.45)) * self.scale)
            player_center = self._world(self.env.player)
            pygame.draw.circle(self.screen, (90, 238, 255), player_center, radius, 2)
            self._draw_glow_circle(player_center, 20, CYAN)
        if not self.env.terminal_available:
            label_text = "EMP READY" if self.env.emp_available else "TERMINAL USED"
            label = self.font_xs.render(label_text, True, LIME if self.env.emp_available else MUTED)
            self.screen.blit(label, (center[0] - label.get_width() // 2, center[1] + 25))
            return
        pulse = 1.0 + 0.12 * math.sin(self.frame * 0.15)
        radius = round(18 * pulse)
        self._draw_glow_circle(center, radius + 4, LIME)
        points = []
        for index in range(6):
            angle = math.tau * index / 6.0 + math.pi / 6.0
            points.append((round(center[0] + math.cos(angle) * radius), round(center[1] + math.sin(angle) * radius)))
        pygame.draw.polygon(self.screen, (12, 48, 38), points)
        pygame.draw.polygon(self.screen, LIME, points, 2)
        self.screen.blit(self.font_xs.render("EMP", True, LIME), (center[0] - 13, center[1] - 8))

    def _draw_cores(self) -> None:
        for index, core in enumerate(self.env.cores):
            if index in self.env.collected_core_indices:
                continue
            center = self._world(core)
            radius = round(CORE_RADIUS * self.scale * (1.0 + 0.14 * math.sin(self.frame * 0.12 + index)))
            self._draw_glow_circle(center, radius + 5, CYAN)
            points = (
                (center[0], center[1] - radius),
                (center[0] + radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] - radius, center[1]),
            )
            pygame.draw.polygon(self.screen, (9, 42, 57), points)
            pygame.draw.polygon(self.screen, CYAN, points, 2)
            pygame.draw.line(self.screen, (212, 255, 255), points[0], points[2], 1)

    def _draw_drone_routes(self) -> None:
        pass

    def _draw_drone_vision(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for drone in self.env.drones:
            stunned = self.env.drone_is_emp_stunned(drone)
            if stunned and self.frame % 10 < 4:
                # Disrupted vision flickering
                continue
            center = self._world(drone.position)
            direction = self.env._drone_forward(drone)
            angle = math.atan2(float(direction[1]), float(direction[0]))
            cone_length = self.env._drone_vision_range(drone)
            cone_width = 0.55 if drone.role in ("hunter", "sentry") else 0.42
            color = RED if drone.lock_steps > 0 else (ORANGE if drone.role == "sentry" else MAGENTA)
            if stunned:
                color = CYAN
                
            points = self._get_vision_cone_points(drone.position, angle, cone_width * 2.0, cone_length)
            
            poly_opacity = 8 if stunned else 24
            line_opacity = 25 if stunned else 80
            pygame.draw.polygon(overlay, (*color, poly_opacity), points)
            pygame.draw.line(overlay, (*color, line_opacity), points[0], points[1], 1)
            pygame.draw.line(overlay, (*color, line_opacity), points[0], points[-1], 1)
            pygame.draw.lines(overlay, (*color, line_opacity // 2), False, points[1:], 1)
            
            # Sentry targeting laser: draw from sentry to player center when locked on/acquiring
            if drone.role == "sentry" and not stunned and (drone.lock_steps > 0 or self.env._drone_can_see_player(drone)):
                player_center = self._world(self.env.player)
                pygame.draw.line(overlay, (255, 30, 66, 180), center, player_center, 2)
        self.screen.blit(overlay, (0, 0))

    def _draw_drones(self) -> None:
        for index, drone in enumerate(self.env.drones):
            center = self._world(drone.position)
            forward = self.env._drone_forward(drone)
            angle = math.atan2(float(forward[1]), float(forward[0]))
            color = RED if drone.lock_steps > 0 else (ORANGE if drone.role == "sentry" else MAGENTA)
            stunned = self.env.drone_is_emp_stunned(drone)
            if stunned:
                color = CYAN
                # Spawn visual electric spark particles occasionally
                if self.frame % 3 == 0:
                    spark_angle = self.random.random() * math.tau
                    spark_vel = np.array([math.cos(spark_angle), math.sin(spark_angle)], dtype=np.float32) * (1.0 + self.random.random() * 2.0)
                    self.particles.append(
                        Particle(
                            position=drone.position.copy(),
                            velocity=spark_vel,
                            life=0.3 + self.random.random() * 0.4,
                            color=CYAN if self.random.random() > 0.35 else GOLD,
                            radius=1.5 + self.random.random() * 2.0,
                        )
                    )
            self._draw_glow_circle(center, 17, color)
            points = self._triangle(center, angle, 15, 10)
            pygame.draw.polygon(self.screen, (52, 13, 46), points)
            pygame.draw.polygon(self.screen, color, points, 2)
            scan_radius = 29 + round(5 * math.sin(self.frame * 0.1 + index))
            pygame.draw.circle(self.screen, color, center, scan_radius, 1)
            if drone.role == "hunter":
                self.screen.blit(self.font_xs.render("HUNTER", True, color), (center[0] - 26, center[1] - 34))
            elif drone.role == "sentry":
                self.screen.blit(self.font_xs.render("SENTRY", True, color), (center[0] - 24, center[1] - 34))

    def _draw_projectiles(self) -> None:
        for projectile in self.env.projectiles:
            center = self._world(projectile.position)
            pygame.draw.circle(self.screen, (255, 232, 120), center, 7)
            pygame.draw.circle(self.screen, ORANGE, center, 12, 1)
            tail = projectile.position - projectile.velocity * 2.0
            pygame.draw.line(self.screen, ORANGE, self._world(tail), center, 2)

    def _draw_player(self) -> None:
        for index, trail_point in enumerate(self.trail):
            center = self._world(trail_point)
            progress = index / max(1, len(self.trail) - 1)
            radius = max(1, round(7 * progress))
            color = (18, round(90 + 105 * progress), round(118 + 120 * progress))
            if getattr(self.env, "in_vent", False):
                color = (10, 45, 60)
            pygame.draw.circle(self.screen, color, center, radius)

        center = self._world(self.env.player)
        angle = math.atan2(float(self.env.heading[1]), float(self.env.heading[0]))
        
        if getattr(self.env, "in_vent", False):
            surf = pygame.Surface((46, 46), pygame.SRCALPHA)
            pygame.draw.circle(surf, (125, 255, 159, 100), (23, 23), 23)
            pygame.draw.polygon(surf, (213, 255, 255, 180), [(23 + round(12 * math.cos(angle + a)), 23 + round(12 * math.sin(angle + a))) for a in (0, 2.42, -2.42)])
            self.screen.blit(surf, (center[0] - 23, center[1] - 23))
        else:
            self._draw_glow_circle(center, 23, CYAN)
            points = self._triangle(center, angle, 19, 13)
            pygame.draw.polygon(self.screen, (8, 42, 60), points)
            pygame.draw.polygon(self.screen, (213, 255, 255), points, 2)
            pygame.draw.circle(self.screen, CYAN, center, 4)

    def _draw_particles(self) -> None:
        for particle in self.particles:
            center = self._world(particle.position)
            radius = max(1, round(particle.radius * self.scale))
            pygame.draw.circle(self.screen, particle.color, center, radius)

    def _draw_header(self) -> None:
        title_text = "CORE RUNNER"
        if getattr(self.env, "campaign_stage", None) is not None:
            stage_name = self.env.campaign_stage_name.upper()
            title_text += f" - STAGE {self.env.campaign_stage}/12: {stage_name}"
        title = self.font_lg.render(title_text, True, INK)
        subtitle = self.font_sm.render("// SECURITY POLICY LAB", True, CYAN)
        self.screen.blit(title, (24, 13))
        self.screen.blit(subtitle, (title.get_width() + 45, 24))

    def _draw_panel(self, stats: dict[str, Any]) -> None:
        pygame.draw.rect(self.screen, (7, 17, 27), pygame.Rect(PANEL_LEFT, ARENA_TOP, PANEL_WIDTH, ARENA_HEIGHT))
        pygame.draw.rect(self.screen, (28, 89, 106), pygame.Rect(PANEL_LEFT, ARENA_TOP, PANEL_WIDTH, ARENA_HEIGHT), 1)
        x = PANEL_LEFT + 18
        y = ARENA_TOP + 17

        self._text("LIVE ARENA TELEMETRY", x, y, self.font_md, CYAN)
        y += 42
        self._text("OBJECTIVE", x, y, self.font_xs, MUTED)
        y += 18
        self._text(self.env.objective_label, x, y, self.font_md, GOLD)
        y += 44

        self._text("ENERGY CORES", x, y, self.font_xs, MUTED)
        y += 19
        for index in range(len(self.env.cores)):
            center = (x + 13 + index * 33, y + 10)
            color = CYAN if index < self.env.cores_collected else (42, 80, 92)
            pygame.draw.circle(self.screen, color, center, 10, 2)
        y += 42

        self._bar(x, y, "SHIELD", self.env.shield / 3.0, LIME if self.env.shield > 1 else RED)
        y += 43
        heat_color = RED if self.env.heat_ratio > 0.72 else ORANGE if self.env.heat_ratio > 0.36 else GOLD
        self._bar(x, y, "SECURITY HEAT", self.env.heat_ratio, heat_color)
        y += 43
        dash_value = self.env.dash_energy / DASH_ENERGY_MAX
        if self.env.dash_cooldown > 0:
            self._bar(x, y, f"BOOST COOLDOWN [{self.env.dash_cooldown}]", dash_value, RED)
        else:
            self._bar(x, y, "BOOST ENERGY", dash_value, CYAN)
        y += 46

        self._text(f"EPISODE STEP  {self.env.step_count:04d}", x, y, self.font_sm, INK)
        y += 27
        self._text(f"REWARD        {self.env.episode_reward:7.2f}", x, y, self.font_sm, INK)
        y += 48

        if self.env.alarm_active:
            self._text("ALARM PHASE", x, y, self.font_md, RED)
            y += 28
            bonus = max(0, MAX_STEPS - self.env.step_count)
            self._text(f"ESCAPE BONUS  {bonus * 2:>8}", x, y, self.font_sm, ORANGE)
            y += 34
        if self.env.emp_timer > 0:
            self._text(f"EMP FIELD     {self.env.emp_timer:>8}", x, y, self.font_sm, CYAN)
            y += 30
        elif self.env.emp_available:
            self._text("EMP READY     SPACE", x, y, self.font_sm, LIME)
            y += 30
        if self.env.hunter_lock_count > 0 or self.env.alarm_active:
            self._text(f"HUNTER RANGE  {self.env.hunter_lock_range:>8.0f}", x, y, self.font_sm, RED)
            y += 30

        if stats:
            self._text("PPO TRAINING", x, y, self.font_md, MAGENTA)
            y += 33
            self._text(f"TOTAL STEPS   {int(stats.get('steps', 0)):>8,}", x, y, self.font_sm, INK)
            y += 24
            self._text(f"FPS           {float(stats.get('fps', 0.0)):>8.0f}", x, y, self.font_sm, INK)
            y += 24
            reward = stats.get("mean_reward")
            reward_label = "--" if reward is None else f"{float(reward):.2f}"
            self._text(f"MEAN RETURN   {reward_label:>8}", x, y, self.font_sm, INK)
            y += 24
            recording_label = stats.get("recording_label")
            spectator_label = str(recording_label) if recording_label else "SPECTATOR: ENV 00"
            self._text(spectator_label[:27], x, y, self.font_xs, MUTED)
            y += 29
            self._draw_mlp_inspector(stats, x, y)
        else:
            if getattr(self.env, "campaign_stage", None) is not None:
                self._text(f"STAGE {self.env.campaign_stage}: {self.env.campaign_stage_name.upper()}", x, y, self.font_xs, MAGENTA)
                y += 24
            self._text("HUMAN CONTROL", x, y, self.font_md, MAGENTA)
            y += 31
            self._text("WASD     MOVE", x, y, self.font_sm, INK)
            y += 23
            self._text("SHIFT    BOOST", x, y, self.font_sm, INK)
            y += 23
            self._text("SPACE    EMP", x, y, self.font_sm, INK)
            y += 23
            self._text("R        RESET", x, y, self.font_sm, INK)

        footer = self.font_xs.render("VECTOR POLICY // 84 OBS // 13 ACT", True, MUTED)
        self.screen.blit(footer, (x, ARENA_TOP + ARENA_HEIGHT - 18))

    def _draw_mlp_inspector(self, stats: dict[str, Any], x: int, y: int) -> None:
        mlp = stats.get("mlp")
        if not isinstance(mlp, dict) or not mlp:
            return

        self._text("MLP POLICY", x, y, self.font_md, CYAN)
        y += 24
        self._text(str(mlp.get("arch", "75 -> 256 -> 256 -> 13")), x, y, self.font_xs, MUTED)
        y += 18
        params = int(mlp.get("params", 0))
        self._text(f"PARAMS {params:>10,}", x, y, self.font_xs, MUTED)
        y += 18

        layers = [
            ("OBS", mlp.get("obs", [])),
            ("H1", (mlp.get("policy_layers") or [[]])[0]),
            ("H2", (mlp.get("policy_layers") or [[], []])[-1]),
            ("ACT", mlp.get("actions", [])),
        ]
        layer_x = [x + 8, x + 79, x + 150, x + 221]
        top = y + 4
        height = 62
        previous_points: list[tuple[int, int]] = []
        for layer_index, (label, values) in enumerate(layers):
            values = list(values)[:10]
            if not values:
                values = [0.0]
            points = []
            for value_index, value in enumerate(values):
                node_y = round(top + value_index * height / max(1, len(values) - 1))
                points.append((layer_x[layer_index], node_y))
                color = self._activation_color(float(value))
                pygame.draw.circle(self.screen, color, (layer_x[layer_index], node_y), 4)
            self._text(label, layer_x[layer_index] - 13, top + height + 6, self.font_xs, MUTED)
            if previous_points:
                for start in previous_points[:: max(1, len(previous_points) // 4)]:
                    for end in points[:: max(1, len(points) // 4)]:
                        pygame.draw.line(self.screen, (22, 64, 78), start, end, 1)
                for point in points:
                    color = self._activation_color(0.2)
                    pygame.draw.circle(self.screen, color, point, 4)
            previous_points = points

        y = top + height + 25
        actions = list(mlp.get("actions", []))
        for index, label in enumerate(("X", "Y", "BOOST", "EMP")):
            value = float(actions[index]) if index < len(actions) else 0.0
            self._mini_bar(x, y + index * 14, label, value)
        value_estimate = float(mlp.get("value", 0.0))
        self._text(f"VALUE {value_estimate:>9.2f}", x + 132, y + 14, self.font_xs, GOLD)

    def _activation_color(self, value: float) -> tuple[int, int, int]:
        strength = min(1.0, abs(value))
        if value >= 0.0:
            return (round(35 + 190 * strength), round(95 + 145 * strength), 255)
        return (255, round(70 + 80 * strength), round(150 + 80 * strength))

    def _mini_bar(self, x: int, y: int, label: str, value: float) -> None:
        value = max(-1.0, min(1.0, value))
        self._text(label, x, y - 3, self.font_xs, MUTED)
        rect = pygame.Rect(x + 43, y, 74, 8)
        pygame.draw.rect(self.screen, (22, 43, 53), rect)
        center_x = rect.x + rect.width // 2
        pygame.draw.line(self.screen, (82, 122, 135), (center_x, rect.y - 2), (center_x, rect.bottom + 2), 1)
        if value >= 0.0:
            fill = pygame.Rect(center_x, rect.y, round(rect.width * 0.5 * value), rect.height)
        else:
            fill = pygame.Rect(center_x + round(rect.width * 0.5 * value), rect.y, round(rect.width * 0.5 * abs(value)), rect.height)
        pygame.draw.rect(self.screen, CYAN if value >= 0.0 else MAGENTA, fill)
        pygame.draw.rect(self.screen, (61, 114, 126), rect, 1)

    def _draw_result_overlay(self) -> None:
        if not self.env.last_result:
            return
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        rect = pygame.Rect(420, 190, 600, 340)
        pygame.draw.rect(self.screen, (8, 19, 28), rect, border_radius=6)
        pygame.draw.rect(self.screen, CYAN if self.env.extracted else RED, rect, 2, border_radius=6)
        result = self.env.last_result
        color = GOLD if result["status"] == "EXTRACTED" else RED
        title = self.font_lg.render(str(result["status"]), True, color)
        self.screen.blit(title, (rect.centerx - title.get_width() // 2, rect.y + 36))
        rows = (
            f"SCORE      {int(result['score']):>7}",
            f"CORES      {int(result['cores'])}/3",
            f"HITS       {int(result['hits'])}",
            f"BOOSTS     {int(result.get('boosts', result['dashes']))}",
            f"STEPS      {int(result['steps'])}",
            f"RL RETURN  {float(result['reward']):>7.2f}",
        )
        y = rect.y + 105
        for row in rows:
            text = self.font_md.render(row, True, INK)
            self.screen.blit(text, (rect.x + 170, y))
            y += 34
        
        if self.env.extracted and getattr(self.env, "campaign_stage", None) is not None:
            from neon_arena.progression import load_unlocked_stage
            unlocked = load_unlocked_stage()
            if self.env.campaign_stage < unlocked:
                cleared_text = f"STAGE {self.env.campaign_stage} CLEARED! UNLOCKED STAGE {unlocked}"
                cleared_lbl = self.font_sm.render(cleared_text, True, GOLD)
                self.screen.blit(cleared_lbl, (rect.centerx - cleared_lbl.get_width() // 2, rect.bottom - 68))
                
        hint = self.font_sm.render("next run starts automatically // R resets during play", True, MUTED)
        self.screen.blit(hint, (rect.centerx - hint.get_width() // 2, rect.bottom - 42))

    def _bar(self, x: int, y: int, label: str, value: float, color: tuple[int, int, int]) -> None:
        value = max(0.0, min(1.0, value))
        self._text(label, x, y, self.font_xs, MUTED)
        rect = pygame.Rect(x, y + 19, PANEL_WIDTH - 36, 10)
        pygame.draw.rect(self.screen, (22, 43, 53), rect)
        pygame.draw.rect(self.screen, color, pygame.Rect(rect.x, rect.y, round(rect.width * value), rect.height))
        pygame.draw.rect(self.screen, (61, 114, 126), rect, 1)

    def _text(
        self,
        text: str,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int],
    ) -> None:
        self.screen.blit(font.render(text, True, color), (x, y))

    def _draw_glow_circle(self, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
        r = radius + 14
        size = r * 2
        overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(overlay, (*color, 22), (r, r), r)
        pygame.draw.circle(overlay, (*color, 35), (r, r), radius + 8)
        self.screen.blit(overlay, (center[0] - r, center[1] - r))

    def _draw_glow_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        min_x = min(start[0], end[0]) - 10
        max_x = max(start[0], end[0]) + 10
        min_y = min(start[1], end[1]) - 10
        max_y = max(start[1], end[1]) + 10
        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        local_start = (start[0] - min_x, start[1] - min_y)
        local_end = (end[0] - min_x, end[1] - min_y)
        pygame.draw.line(overlay, (*color, 40), local_start, local_end, width + 6)
        self.screen.blit(overlay, (min_x, min_y))
        pygame.draw.line(self.screen, color, start, end, width)

    def _triangle(
        self,
        center: tuple[int, int],
        angle: float,
        length: int,
        width: int,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        nose = (round(center[0] + math.cos(angle) * length), round(center[1] + math.sin(angle) * length))
        left = (
            round(center[0] + math.cos(angle + 2.42) * width),
            round(center[1] + math.sin(angle + 2.42) * width),
        )
        right = (
            round(center[0] + math.cos(angle - 2.42) * width),
            round(center[1] + math.sin(angle - 2.42) * width),
        )
        return nose, left, right

    def _get_vision_cone_points(
        self,
        origin: np.ndarray | tuple[float, float],
        angle: float,
        fov: float,
        max_dist: float,
    ) -> list[tuple[int, int]]:
        origin_np = np.array(origin, dtype=np.float32)
        points = [self._world(origin)]
        
        num_rays = 16
        half_fov = fov / 2.0
        
        for i in range(num_rays + 1):
            ray_angle = angle - half_fov + (i / num_rays) * fov
            ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)], dtype=np.float32)
            
            closest_dist = max_dist
            for block in self.env.city_blocks:
                is_breakable = getattr(self.env, "breakable_walls", None) and block in self.env.breakable_walls
                if is_breakable:
                    continue
                dist = ray_rect_distance(origin_np, ray_dir, block)
                if dist is not None and dist < closest_dist:
                    closest_dist = dist
            
            hit_pos = origin_np + ray_dir * closest_dist
            points.append(self._world(hit_pos))
            
        return points

    def _add_event_particles(self) -> None:
        for event_name, position in self.env.pop_events():
            config = {
                "dash": (CYAN, 9, 2.4),
                "impact": (GOLD, 10, 2.7),
                "danger": (RED, 18, 3.4),
                "camera_detection": (RED, 12, 2.2),
                "core": (CYAN, 26, 4.0),
                "extract": (GOLD, 40, 5.0),
                "alarm": (RED, 42, 4.4),
                "emp": (LIME, 34, 3.8),
                "emp_pickup": (LIME, 20, 2.9),
                "evade": (GOLD, 12, 2.2),
                "breach": (ORANGE, 35, 5.5),
                "plant_charge": (RED, 8, 1.5),
            }
            color, count, speed = config.get(event_name, (INK, 7, 2.0))
            shake_config = {
                "danger": 4.0,
                "alarm": 4.0,
                "extract": 6.0,
                "breach": 1.5,
            }
            if event_name in shake_config:
                self.shake = max(self.shake, shake_config[event_name])
            for _ in range(count):
                angle = self.random.random() * math.tau
                velocity = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                velocity *= speed * (0.45 + self.random.random())
                
                # Custom particle coloring/life for breaches to make them look like debris/explosions
                p_color = color
                p_radius = 2.0 + self.random.random() * 3.0
                p_life = 0.6 + self.random.random() * 0.65
                if event_name == "breach":
                    if self.random.random() < 0.4:
                        p_color = (120, 120, 120) if self.random.random() < 0.5 else (80, 80, 80)
                        p_radius = 3.0 + self.random.random() * 4.0
                        p_life = 1.0 + self.random.random() * 0.8
                    else:
                        p_color = ORANGE if self.random.random() < 0.7 else (255, 230, 80)

                self.particles.append(
                    Particle(
                        position=np.array(position, dtype=np.float32),
                        velocity=velocity,
                        life=p_life,
                        color=p_color,
                        radius=p_radius,
                    )
                )

    def _update_particles(self) -> None:
        active = []
        for particle in self.particles:
            particle.position += particle.velocity
            particle.velocity *= 0.92
            particle.life -= 0.035
            particle.radius *= 0.975
            if particle.life > 0.0:
                active.append(particle)
        self.particles = active

    def _update_camera_feedback(self) -> None:
        if self.shake > 0.1:
            self.camera_offset = (
                self.random.uniform(-self.shake, self.shake),
                self.random.uniform(-self.shake, self.shake),
            )
            self.shake *= 0.84
        else:
            self.camera_offset = (0.0, 0.0)
            self.shake = 0.0

    def _draw_gates(self) -> None:
        # Draw cable link between security gates and terminal
        if self.env.terminal_available:
            for gate in self.env.gates:
                if gate.type in ("security", "terminal") and not gate.is_open:
                    gate_center = (
                        round(self.world_left + (gate.rect.x + gate.rect.width * 0.5) * self.scale),
                        round(self.world_top + (gate.rect.y + gate.rect.height * 0.5) * self.scale),
                    )
                    term_center = self._world(self.env.terminal_point)
                    pygame.draw.line(self.screen, (24, 78, 62), gate_center, term_center, 1)

        for gate in self.env.gates:
            if not gate.is_open:
                rect = pygame.Rect(
                    round(self.world_left + gate.rect.x * self.scale),
                    round(self.world_top + gate.rect.y * self.scale),
                    round(gate.rect.width * self.scale),
                    round(gate.rect.height * self.scale),
                )
                if gate.type == "blue":
                    color = CYAN
                elif gate.type == "red":
                    color = MAGENTA
                elif gate.type == "security":
                    color = LIME
                else:  # terminal or other
                    color = RED
                
                # Pulse laser opacity
                laser_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                laser_surface.fill((*color, 45 + round(20 * math.sin(self.frame * 0.22))))
                self.screen.blit(laser_surface, rect.topleft)
                pygame.draw.rect(self.screen, color, rect, 2)
                
                if rect.width > rect.height:
                    pygame.draw.line(self.screen, color, rect.midleft, rect.midright, 1)
                else:
                    pygame.draw.line(self.screen, color, rect.midtop, rect.midbottom, 1)

                # Draw instructions above closed gates
                label_text = {
                    "blue": "COLLECT 1 CORE TO UNLOCK",
                    "red": "SECURE ALL CORES TO UNLOCK",
                    "security": "HACK TERMINAL TO UNLOCK",
                    "terminal": "HACK TERMINAL TO UNLOCK"
                }.get(gate.type, "")
                if label_text:
                    lbl = self.font_xs.render(label_text, True, color)
                    self.screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.y - 18))

    def _draw_cameras(self) -> None:
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        for camera in self.env.cameras:
            center = self._world(camera.position)
            angle = camera.base_angle + camera.current_angle
            
            stunned = self.env.camera_is_emp_stunned(camera)
            detected = self.env._player_seen_by_camera(camera)
            color = CYAN if stunned else (RED if detected else ORANGE)
            
            pygame.draw.circle(self.screen, (24, 45, 60), center, 8)
            pygame.draw.circle(self.screen, color, center, 8, 2)
            
            lens_end = (
                round(center[0] + math.cos(angle) * 12),
                round(center[1] + math.sin(angle) * 12),
            )
            pygame.draw.line(self.screen, color, center, lens_end, 3)
            
            if stunned and self.frame % 10 < 4:
                continue
            
            cone_length = 280.0
            cone_width = camera.fov / 2.0
            
            points = self._get_vision_cone_points(camera.position, angle, camera.fov, cone_length)
            
            poly_opacity = 22 if detected else (6 if stunned else 8)
            line_opacity = 70 if detected else (15 if stunned else 30)
            pygame.draw.polygon(overlay, (*color, poly_opacity), points)
            pygame.draw.line(overlay, (*color, line_opacity), points[0], points[1], 1)
            pygame.draw.line(overlay, (*color, line_opacity), points[0], points[-1], 1)
            pygame.draw.lines(overlay, (*color, line_opacity // 2), False, points[1:], 1)
        self.screen.blit(overlay, (0, 0))

    def _draw_loot(self) -> None:
        for item in self.env.loot:
            if item.collected:
                continue
            center = self._world(item.position)
            pulse = 1.0 + 0.15 * math.sin(self.frame * 0.15)
            radius = round(10 * pulse)
            
            if item.type == "data_shard":
                color = CYAN
                points = (
                    (center[0], center[1] - radius),
                    (center[0] + round(radius * 0.7), center[1]),
                    (center[0], center[1] + radius),
                    (center[0] - round(radius * 0.7), center[1]),
                )
            elif item.type == "gold_cache":
                color = GOLD
                points = []
                for i in range(6):
                    ang = math.tau * i / 6.0 + self.frame * 0.05
                    points.append((round(center[0] + math.cos(ang) * radius), round(center[1] + math.sin(ang) * radius)))
            else:  # black_box
                color = MAGENTA
                points = (
                    (center[0] - radius, center[1] - radius),
                    (center[0] + radius, center[1] - radius),
                    (center[0] + radius, center[1] + radius),
                    (center[0] - radius, center[1] + radius),
                )
                
            self._draw_glow_circle(center, radius + 3, color)
            pygame.draw.polygon(self.screen, (10, 20, 30), points)
            pygame.draw.polygon(self.screen, color, points, 2)
            
            label = self.font_xs.render("LOOT", True, color)
            self.screen.blit(label, (center[0] - label.get_width() // 2, center[1] - radius - 15))


    def _draw_minimap(self) -> None:
        minimap_w = 160
        minimap_h = 101
        mx = ARENA_LEFT + ARENA_WIDTH - minimap_w - 15
        my = ARENA_TOP + 15
        
        minimap_surf = pygame.Surface((minimap_w, minimap_h), pygame.SRCALPHA)
        minimap_surf.fill((5, 12, 22, 180))
        pygame.draw.rect(minimap_surf, (30, 90, 110, 220), pygame.Rect(0, 0, minimap_w, minimap_h), 1)
        
        scale_x = minimap_w / self.env.world_width
        scale_y = minimap_h / self.env.world_height
        
        for block in self.env.city_blocks:
            bx = block.x * scale_x
            by = block.y * scale_y
            bw = block.width * scale_x
            bh = block.height * scale_y
            is_breakable = getattr(self.env, "breakable_walls", None) and block in self.env.breakable_walls
            if is_breakable:
                pygame.draw.rect(minimap_surf, (255, 209, 82, 180), pygame.Rect(bx, by, bw, bh))
                pygame.draw.rect(minimap_surf, (255, 209, 82, 230), pygame.Rect(bx, by, bw, bh), 1)
            else:
                pygame.draw.rect(minimap_surf, (20, 45, 60, 160), pygame.Rect(bx, by, bw, bh))
                pygame.draw.rect(minimap_surf, (40, 95, 115, 200), pygame.Rect(bx, by, bw, bh), 1)
                
        # Draw vents on minimap
        if hasattr(self.env, "vents"):
            for vent in self.env.vents:
                ax = vent.pos_a[0] * scale_x
                ay = vent.pos_a[1] * scale_y
                bx = vent.pos_b[0] * scale_x
                by = vent.pos_b[1] * scale_y
                pygame.draw.line(minimap_surf, (125, 255, 159, 100), (round(ax), round(ay)), (round(bx), round(by)), 1)
                pygame.draw.circle(minimap_surf, LIME, (round(ax), round(ay)), 2)
                pygame.draw.circle(minimap_surf, LIME, (round(bx), round(by)), 2)
            
        # Draw shock tiles on minimap
        for tile in getattr(self.env, "shock_tiles", []):
            tx = tile.rect.x * scale_x
            ty = tile.rect.y * scale_y
            tw = tile.rect.width * scale_x
            th = tile.rect.height * scale_y
            if tile.state == "SAFE":
                color = (0, 120, 255, 100)
            elif tile.state == "WARN":
                color = (255, 130, 0, 150)
            else:
                color = (255, 30, 30, 200)
            pygame.draw.rect(minimap_surf, color, pygame.Rect(tx, ty, tw, th))

        for gate in self.env.gates:
            if not gate.is_open:
                gx = gate.rect.x * scale_x
                gy = gate.rect.y * scale_y
                gw = gate.rect.width * scale_x
                gh = gate.rect.height * scale_y
                color = CYAN if gate.type == "blue" else (MAGENTA if gate.type == "red" else (LIME if gate.type == "security" else RED))
                pygame.draw.rect(minimap_surf, (*color, 230), pygame.Rect(gx, gy, gw, gh))
                
        tx = self.env.terminal_point[0] * scale_x
        ty = self.env.terminal_point[1] * scale_y
        t_color = LIME if self.env.terminal_available else MUTED
        pygame.draw.circle(minimap_surf, t_color, (round(tx), round(ty)), 3)
        
        for i, core in enumerate(self.env.cores):
            if i not in self.env.collected_core_indices:
                cx = core[0] * scale_x
                cy = core[1] * scale_y
                pygame.draw.circle(minimap_surf, CYAN, (round(cx), round(cy)), 3)
                
        ex = self.env.extraction_point[0] * scale_x
        ey = self.env.extraction_point[1] * scale_y
        e_color = GOLD if self.env.cores_collected >= len(self.env.cores) else MUTED
        pygame.draw.circle(minimap_surf, e_color, (round(ex), round(ey)), 4, 1)
        
        for drone in self.env.drones:
            dx = drone.position[0] * scale_x
            dy = drone.position[1] * scale_y
            d_color = RED if drone.lock_steps > 0 else (ORANGE if drone.role == "sentry" else MAGENTA)
            if self.env.drone_is_emp_stunned(drone):
                d_color = CYAN
            pygame.draw.circle(minimap_surf, d_color, (round(dx), round(dy)), 2)
            
        px = self.env.player[0] * scale_x
        py = self.env.player[1] * scale_y
        pygame.draw.circle(minimap_surf, (255, 255, 255), (round(px), round(py)), 3)
        
        self.screen.blit(minimap_surf, (mx, my))

    def _draw_districts_debug(self) -> None:
        colors = {
            "MARKET": (30, 255, 100, 20),
            "SPAWN": (30, 255, 100, 20),
            "CONNECT": (100, 100, 150, 15),
            "PLAZA": (0, 200, 255, 12),
            "SEC-HUB": (255, 100, 0, 18),
            "VAULT": (255, 0, 255, 18),
            "EXTRACT": (0, 255, 0, 18)
        }
        
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        
        procedural_rooms = getattr(self.env, "procedural_rooms", [])
        room_nodes = getattr(self.env, "room_nodes", [])
        
        curr_room = -1
        obj_room = -1
        active_target = None
        if hasattr(self.env, "_get_room_id_at_position"):
            curr_room = self.env._get_room_id_at_position(self.env.player)
            active_target = self.env.objective
            obj_room = self.env._get_room_id_at_position(active_target)
            
        if procedural_rooms and room_nodes:
            for r_idx, (room, role) in enumerate(procedural_rooms):
                # Tint current room and objective room uniquely
                if r_idx == curr_room:
                    color = (0, 255, 255, 45)  # Cyan tint for current room
                elif r_idx == obj_room:
                    color = (255, 0, 255, 45)  # Magenta tint for objective room
                else:
                    color = colors.get(role, (255, 255, 255, 10))
                    
                top_left = self._world((room.x, room.y))
                bottom_right = self._world((room.x + room.width, room.y + room.height))
                
                rect = pygame.Rect(
                    top_left[0], 
                    top_left[1], 
                    bottom_right[0] - top_left[0], 
                    bottom_right[1] - top_left[1]
                )
                pygame.draw.rect(overlay, color, rect)
                pygame.draw.rect(overlay, (color[0], color[1], color[2], 120), rect, 1)
                
                # Show role and room ID
                label = f"{role} #{r_idx}"
                text_surf = self.font_xs.render(label, True, (color[0], color[1], color[2]))
                center_x = rect.x + rect.width // 2 - text_surf.get_width() // 2
                center_y = rect.y + rect.height // 2 - text_surf.get_height() // 2
                overlay.blit(text_surf, (center_x, center_y))
                
            # Draw Dijkstra/BFS path from current room to objective room
            if curr_room != -1 and obj_room != -1:
                if "large_proc" in self.env.curriculum_stage and hasattr(self.env, "_dijkstra_room_path"):
                    path = self.env._dijkstra_room_path(curr_room, obj_room)
                elif hasattr(self.env, "_bfs_room_path"):
                    path = self.env._bfs_room_path(curr_room, obj_room)
                else:
                    path = []
                if len(path) >= 2:
                    # Construct a doorway-connected path from player through doors to active target
                    pts = [self.env.player]
                    for idx in range(len(path) - 1):
                        r_a = path[idx]
                        r_b = path[idx+1]
                        door_center = None
                        for edge in self.env.door_edges:
                            if (edge.room_a == r_a and edge.room_b == r_b) or (edge.room_a == r_b and edge.room_b == r_a):
                                door_center = edge.center
                                break
                        if door_center is not None:
                            pts.append(np.array(door_center, dtype=np.float32))
                    pts.append(active_target)
                    
                    # Draw yellow path lines connecting these points
                    for idx in range(len(pts) - 1):
                        pygame.draw.line(overlay, (255, 255, 0, 180), self._world(pts[idx]), self._world(pts[idx+1]), 4)
                        
            # Draw actual A* path waypoints to the next target
            if hasattr(self.env, "_get_route_waypoints") and hasattr(self.env, "dist_map"):
                target = self.env._get_next_route_target() if "large_proc" in self.env.curriculum_stage else self.env.objective
                path_waypoints = self.env._get_route_waypoints(self.env.player, target, self.env.dist_map)
                if path_waypoints:
                    prev_pt = self._world(self.env.player)
                    for pt in path_waypoints:
                        curr_pt = self._world(pt)
                        pygame.draw.line(overlay, (49, 228, 255, 180), prev_pt, curr_pt, 2)
                        pygame.draw.circle(overlay, (49, 228, 255, 180), curr_pt, 3)
                        prev_pt = curr_pt
                        
            # Draw next doorway target marker
            if curr_room != obj_room and hasattr(self.env, "_get_next_route_target"):
                target = self.env._get_next_route_target()
                pygame.draw.circle(overlay, (255, 255, 0, 220), self._world(target), 14, 3)
                
        else:
            role_grid = [
                ["MARKET",  "CONNECT", "SEC-HUB", "CONNECT", "VAULT"],
                ["CONNECT", "PLAZA",   "CONNECT", "PLAZA",   "CONNECT"],
                ["MARKET",  "CONNECT", "PLAZA",   "CONNECT", "EXTRACT"]
            ]
            col_width = 300.0
            row_height = 316.666
            for r in range(3):
                for c in range(5):
                    role = role_grid[r][c]
                    color = colors.get(role, (255, 255, 255, 10))
                    x_min = c * col_width
                    y_min = r * row_height
                    
                    top_left = self._world((x_min, y_min))
                    bottom_right = self._world((x_min + col_width, y_min + row_height))
                    
                    rect = pygame.Rect(
                        top_left[0], 
                        top_left[1], 
                        bottom_right[0] - top_left[0], 
                        bottom_right[1] - top_left[1]
                    )
                    pygame.draw.rect(overlay, color, rect)
                    pygame.draw.rect(overlay, (color[0], color[1], color[2], 80), rect, 1)
                    
                    text_surf = self.font_xs.render(role, True, (color[0], color[1], color[2]))
                    center_x = rect.x + rect.width // 2 - text_surf.get_width() // 2
                    center_y = rect.y + rect.height // 2 - text_surf.get_height() // 2
                    overlay.blit(text_surf, (center_x, center_y))
                    
        self.screen.blit(overlay, (0, 0))

    def _draw_hazards(self) -> None:
        # 1. Draw Coolant Spills
        for spill in getattr(self.env, "coolant_spills", []):
            rect = pygame.Rect(
                round(self.world_left + spill.rect.x * self.scale),
                round(self.world_top + spill.rect.y * self.scale),
                round(spill.rect.width * self.scale),
                round(spill.rect.height * self.scale),
            )
            puddle_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            puddle_color = (0, 180, 220, 50 + round(15 * math.sin(self.frame * 0.1)))
            pygame.draw.rect(puddle_surface, puddle_color, (0, 0, rect.width, rect.height), border_radius=10)
            pygame.draw.rect(puddle_surface, (0, 220, 255, 100), (0, 0, rect.width, rect.height), width=2, border_radius=10)
            self.screen.blit(puddle_surface, rect.topleft)

        # 2. Draw Shock Tiles
        for tile in getattr(self.env, "shock_tiles", []):
            rect = pygame.Rect(
                round(self.world_left + tile.rect.x * self.scale),
                round(self.world_top + tile.rect.y * self.scale),
                round(tile.rect.width * self.scale),
                round(tile.rect.height * self.scale),
            )
            tile_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            
            if tile.state == "SAFE":
                color = (0, 120, 255, 30)
                border_color = (0, 160, 255, 90)
            elif tile.state == "WARN":
                alpha = 40 + round(35 * math.sin(self.frame * 0.3))
                color = (255, 130, 0, alpha)
                border_color = (255, 160, 0, 150)
            else: # ACTIVE
                alpha = 70 + round(50 * math.sin(self.frame * 0.5))
                color = (255, 30, 30, alpha)
                border_color = (255, 80, 80, 200)
                
            pygame.draw.rect(tile_surface, color, (0, 0, rect.width, rect.height))
            pygame.draw.rect(tile_surface, border_color, (0, 0, rect.width, rect.height), width=2)
            
            if tile.state == "ACTIVE":
                pygame.draw.line(tile_surface, (255, 255, 255, 200), (0, rect.height//2), (rect.width, rect.height//2), 2)
                pygame.draw.line(tile_surface, (255, 255, 255, 200), (rect.width//2, 0), (rect.width//2, rect.height), 2)
            elif tile.state == "WARN":
                pygame.draw.rect(tile_surface, (255, 160, 0, 200), (rect.width//2 - 2, 8, 4, rect.height - 24))
                pygame.draw.circle(tile_surface, (255, 160, 0, 200), (rect.width//2, rect.height - 10), 3)
                
            self.screen.blit(tile_surface, rect.topleft)

    def _draw_drone_debug(self) -> None:
        nodes = getattr(self.env, "semantic_patrol_nodes", [])
        for node in nodes:
            pos = self._world(node)
            pygame.draw.circle(self.screen, (0, 255, 255), pos, 4)
            
        for drone in self.env.drones:
            state = getattr(drone, "state", "PATROL")
            pos = self._world(drone.position)
            state_text = self.font_xs.render(state, True, (255, 255, 255))
            self.screen.blit(state_text, (pos[0] - state_text.get_width() // 2, pos[1] - 25))



