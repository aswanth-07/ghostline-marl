from __future__ import annotations

import math

import numpy as np

from neon_arena.config import Rect


def length(vector: np.ndarray) -> float:
    vx, vy = vector[0], vector[1]
    return math.sqrt(vx * vx + vy * vy)


def normalized(vector: np.ndarray) -> np.ndarray:
    vx, vy = vector[0], vector[1]
    magnitude = math.sqrt(vx * vx + vy * vy)
    if magnitude <= 1e-8:
        return np.zeros(2, dtype=np.float32)
    return np.array([vx / magnitude, vy / magnitude], dtype=np.float32)


def clamp_magnitude(vector: np.ndarray, maximum: float) -> np.ndarray:
    vx, vy = vector[0], vector[1]
    magnitude = math.sqrt(vx * vx + vy * vy)
    if magnitude <= maximum:
        return np.array([vx, vy], dtype=np.float32)
    ratio = maximum / magnitude
    return np.array([vx * ratio, vy * ratio], dtype=np.float32)


def circle_rect_overlap(position: np.ndarray, radius: float, rect: Rect) -> bool:
    px, py = position[0], position[1]
    rx, ry = rect.x, rect.y
    r_right = rx + rect.width
    r_bottom = ry + rect.height
    
    if px < rx:
        nearest_x = rx
    elif px > r_right:
        nearest_x = r_right
    else:
        nearest_x = px
        
    if py < ry:
        nearest_y = ry
    elif py > r_bottom:
        nearest_y = r_bottom
    else:
        nearest_y = py
        
    dx = px - nearest_x
    dy = py - nearest_y
    return dx * dx + dy * dy < radius * radius


def ray_rect_distance(origin: np.ndarray, direction: np.ndarray, rect: Rect) -> float | None:
    ox, oy = origin[0], origin[1]
    dx, dy = direction[0], direction[1]
    rx, ry = rect.x, rect.y
    r_right = rx + rect.width
    r_bottom = ry + rect.height
    
    t_min = -math.inf
    t_max = math.inf
    
    # Axis 0 (x)
    if abs(dx) < 1e-8:
        if ox < rx or ox > r_right:
            return None
    else:
        inv_d = 1.0 / dx
        first = (rx - ox) * inv_d
        second = (r_right - ox) * inv_d
        if first > second:
            first, second = second, first
        if first > t_min:
            t_min = first
        if second < t_max:
            t_max = second
        if t_max < t_min:
            return None
            
    # Axis 1 (y)
    if abs(dy) < 1e-8:
        if oy < ry or oy > r_bottom:
            return None
    else:
        inv_d = 1.0 / dy
        first = (ry - oy) * inv_d
        second = (r_bottom - oy) * inv_d
        if first > second:
            first, second = second, first
        if first > t_min:
            t_min = first
        if second < t_max:
            t_max = second
        if t_max < t_min:
            return None
            
    if t_max < 0:
        return None
    return t_min if t_min > 0.0 else 0.0
