from __future__ import annotations

import json

import numpy as np

from ghostline.progression import (
    DEFAULT_BINDINGS,
    load_progression,
    load_settings,
    normalize_settings,
    record_run,
    save_settings,
    telemetry_path,
)
from ghostline.simulation import GhostlineSimulation
from ghostline.types import GuardGrade, GuardMode, SimEvent


def test_accessibility_profile_is_normalized_and_persistent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    settings = load_settings()
    settings["audio"].update({"master": 4.0, "music": 0.32, "sfx": -1.0})
    settings["accessibility"].update({"high_contrast": True, "hud_scale": 1.31, "timer_assist": True})
    settings["bindings"]["dash"] = "q"
    saved = save_settings(settings)

    assert saved["audio"] == {"enabled": True, "master": 1.0, "music": 0.32, "sfx": 0.0}
    assert saved["accessibility"]["high_contrast"] is True
    assert saved["accessibility"]["hud_scale"] == 1.25
    assert saved["accessibility"]["timer_assist"] is True
    assert load_settings()["bindings"]["dash"] == "q"
    assert load_progression()["highest_unlocked_tier"] == 1


def test_missing_or_invalid_settings_migrate_to_safe_defaults() -> None:
    normalized = normalize_settings({"audio": "bad", "bindings": {"move_up": "", "dash": "Q"}})
    assert normalized["audio"]["master"] == 0.75
    assert normalized["bindings"]["move_up"] == DEFAULT_BINDINGS["move_up"]
    assert normalized["bindings"]["dash"] == "q"


def test_timer_assist_extends_only_when_explicitly_enabled() -> None:
    from ghostline.app import apply_human_timer_assist

    standard = GhostlineSimulation(seed=14, tier=4)
    assisted = GhostlineSimulation(seed=14, tier=4)
    original = standard.level.mission_seconds
    apply_human_timer_assist(standard, False)
    apply_human_timer_assist(assisted, True)
    assert standard.level.mission_seconds == original
    assert assisted.level.mission_seconds == round(original * 1.35)


def test_run_telemetry_writes_jsonl_and_compact_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    record = record_run(
        {
            "controller": "human",
            "policy": "keyboard",
            "seed": 712,
            "tier": 3,
            "success": True,
            "duration_seconds": 48.25,
            "max_trace": 31.0,
            "detections": 1,
            "damage": 0,
            "path_efficiency": 0.72,
            "path": [[0.0, 32.0, 32.0], [1.0, 48.0, 32.0]],
            "trace_curve": [[0.0, 0.0], [1.0, 4.2]],
        }
    )
    lines = telemetry_path().read_text(encoding="utf-8").splitlines()
    persisted = json.loads(lines[-1])

    assert record["schema_version"] == 1
    assert persisted["path"][-1] == [1.0, 48.0, 32.0]
    assert persisted["trace_curve"][-1] == [1.0, 4.2]
    assert load_progression()["recent_runs"][-1]["seed"] == 712


def test_rebinding_swaps_conflicts_without_orphaning_an_action() -> None:
    from ghostline.app import GameApp

    app = GameApp.__new__(GameApp)
    app.settings = {"bindings": dict(DEFAULT_BINDINGS)}
    app._persist_settings = lambda: None
    old_dash = app.settings["bindings"]["dash"]
    old_pulse = app.settings["bindings"]["pulse"]
    app._assign_binding("dash", old_pulse)

    assert app.settings["bindings"]["dash"] == old_pulse
    assert app.settings["bindings"]["pulse"] == old_dash
    app._assign_binding("confirm", "escape")
    assert app.settings["bindings"]["confirm"] == "escape"
    assert app.settings["bindings"]["back"] == "return"
    # Cross-context sharing remains intentional (Pause and Back default to Escape).
    assert app.settings["bindings"]["pause"] == "escape"


def test_audio_master_music_and_sfx_groups_are_independent(monkeypatch) -> None:
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.audio import AudioDirector

    audio = AudioDirector(enabled=True)
    assert audio.ready
    audio.set_mix(master=0.5, music=0.2, sfx=0.8)
    menu_volume = audio.sounds["menu"].get_volume()
    ambient_volume = audio.ambient.get_volume()
    audio.set_mix(music=1.0, sfx=0.1)

    assert audio.sounds["menu"].get_volume() < menu_volume
    assert audio.ambient.get_volume() > ambient_volume
    audio.close()


def test_renderer_accessibility_modes_and_captions_are_headless_safe(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=42, tier=6)
    renderer = GhostlineRenderer(sim, visible=False)
    default = renderer.draw(return_array=True)
    renderer.apply_accessibility(
        {
            "high_contrast": True,
            "color_safe": True,
            "reduced_motion": True,
            "reduced_flashes": True,
            "sound_captions": True,
            "hud_scale": 1.5,
        }
    )
    renderer.ingest_events(
        [
            SimEvent("detected", tuple(sim.player)),
            SimEvent("damage", tuple(sim.player)),
            SimEvent("pulse", tuple(sim.player)),
        ]
    )
    accessible = renderer.draw(
        return_array=True,
        lab_stats={"policy": "TEST ONNX", "action": "NE +DASH", "latency_ms": 2.2, "hidden": "9.4"},
    )
    renderer.close()

    assert default.shape == accessible.shape == (360, 640, 3)
    assert default.dtype == accessible.dtype == np.uint8
    assert float(np.mean(np.abs(default.astype(np.int16) - accessible.astype(np.int16)))) > 2.0
    assert renderer.reduced_motion is True
    assert renderer.sound_captions_enabled is True


def test_menu_uses_flat_gameplay_schematic_without_loading_key_art(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=7, tier=1), visible=False)
    title_frame = renderer.draw_screen(title="GHOSTLINE", items=["PLAY", "QUIT"], return_array=True)
    briefing_frame = renderer.draw_screen(title="TIER 4 // COUNTERMEASURE", items=["DEPLOY"], return_array=True)
    renderer.close()

    assert not hasattr(renderer, "_key_art")
    assert title_frame.shape == briefing_frame.shape == (360, 640, 3)
    # The opening is deliberately crisp, palette-limited 2D pixel art rather
    # than the retired pseudo-isometric illustration.
    assert 30 < len(np.unique(title_frame.reshape(-1, 3), axis=0)) < 80


def test_world_render_is_never_dimmed_by_square_exploration_tiles(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    hidden_sim = GhostlineSimulation(seed=2_000_004, tier=6)
    revealed_sim = GhostlineSimulation(seed=2_000_004, tier=6)
    hidden_sim.explored[:] = False
    revealed_sim.explored[:] = True
    hidden = GhostlineRenderer(hidden_sim, visible=False)
    revealed = GhostlineRenderer(revealed_sim, visible=False)
    hidden.apply_accessibility({"reduced_motion": True})
    revealed.apply_accessibility({"reduced_motion": True})

    hidden_frame = hidden.draw(return_array=True)
    revealed_frame = revealed.draw(return_array=True)
    hidden.close()
    revealed.close()

    # Exploration remains a policy/minimap concept, but it must not grey out
    # furniture or stamp a visibly tiled fog layer over the playfield.
    assert np.array_equal(hidden_frame, revealed_frame)


def test_menu_dossier_wraps_long_accessibility_copy_inside_panel(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=7, tier=1), visible=False)
    calls: list[tuple[str, int, int, object]] = []
    original_text = renderer._text

    def tracked_text(text, x, y, font, color):
        calls.append((str(text), int(x), int(y), font))
        original_text(text, x, y, font, color)

    monkeypatch.setattr(renderer, "_text", tracked_text)
    renderer.draw_screen(
        title="ACCESSIBILITY",
        panel=[
            "TIMER ASSIST adds 35% to human contract windows and is recorded in telemetry.",
            "REDUCED MOTION disables shake, trails, and moving UI art.",
        ],
        return_array=True,
    )
    renderer.close()

    panel_copy = [(text, y, font) for text, x, y, font in calls if x == 386 and y >= 135]
    assert len(panel_copy) >= 4
    assert all(font.size(text)[0] <= 208 for text, _, font in panel_copy)
    assert max(y + font.get_height() for _, y, font in panel_copy) <= 324


def test_environment_atlas_uses_nearest_scaling_and_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import ATLAS_REGIONS, GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    sim.explored[:] = True
    renderer = GhostlineRenderer(sim, visible=False)
    assert renderer._environment_atlas is not None
    desk = renderer._atlas_sprite("desk", 58)
    assert desk is not None and desk.get_width() == 58
    region = ATLAS_REGIONS["desk"]
    source = renderer._environment_atlas.subsurface(region.rect)
    source_colors = set(map(tuple, pygame.surfarray.array3d(source).reshape(-1, 3)))
    scaled_colors = set(map(tuple, pygame.surfarray.array3d(desk).reshape(-1, 3)))
    assert scaled_colors <= source_colors

    atlas_frame = renderer.draw(return_array=True)
    renderer._environment_atlas = None
    fallback_frame = renderer.draw(return_array=True)
    renderer.close()
    assert atlas_frame.shape == fallback_frame.shape == (360, 640, 3)
    assert float(np.mean(np.abs(atlas_frame.astype(np.int16) - fallback_frame.astype(np.int16)))) > 0.25


def test_release_assets_exclude_source_drafts_but_keep_alpha_atlas(tmp_path) -> None:
    from ghostline.packaging import _release_assets

    visual = tmp_path / "assets" / "visual"
    visual.mkdir(parents=True)
    for name in (
        "ghostline-environment-atlas-v1.png",
        "ghostline-environment-atlas-source-v1.png",
        "ghostline-character-security-atlas-v1.png",
        "ghostline-character-security-atlas-source-v1.png",
        "ghostline-diagonal-locomotion-v2.png",
        "ghostline-diagonal-locomotion-source-v2.png",
        "ghostline-key-art-menu.png",
        "ghostline-key-art-source.png",
    ):
        (visual / name).write_bytes(b"asset")
    (tmp_path / "assets" / "licenses.json").write_text(
        json.dumps(
            {
                "project": "Ghostline",
                "runtime_distribution": {
                    "license": "MIT",
                    "files": [
                        "assets/visual/ghostline-environment-atlas-v1.png",
                        "assets/visual/ghostline-character-security-atlas-v1.png",
                        "assets/visual/ghostline-diagonal-locomotion-v2.png",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    selected = {source.name for source, _ in _release_assets(tmp_path)}
    assert selected == {
        "licenses.json",
        "ghostline-environment-atlas-v1.png",
        "ghostline-character-security-atlas-v1.png",
        "ghostline-diagonal-locomotion-v2.png",
    }


def test_web_asset_filter_excludes_source_drafts_but_keeps_alpha_atlas() -> None:
    import shutil

    from scripts.build_web import WEB_ASSET_IGNORE_PATTERNS

    filenames = [
        "ghostline-environment-atlas-v1.png",
        "ghostline-environment-atlas-source-v1.png",
        "ghostline-character-security-atlas-v1.png",
        "ghostline-character-security-atlas-source-v1.png",
        "ghostline-diagonal-locomotion-v2.png",
        "ghostline-diagonal-locomotion-source-v2.png",
        "ghostline-key-art-menu.png",
        "ghostline-key-art-source.png",
    ]
    ignored = set(shutil.ignore_patterns(*WEB_ASSET_IGNORE_PATTERNS)("assets/visual", filenames))
    assert ignored == {
        "ghostline-environment-atlas-source-v1.png",
        "ghostline-character-security-atlas-source-v1.png",
        "ghostline-diagonal-locomotion-source-v2.png",
        "ghostline-key-art-menu.png",
        "ghostline-key-art-source.png",
    }


def test_character_security_atlas_has_complete_direction_maps_and_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import (
        CHARACTER_REGIONS,
        DRONE_DIRECTION_REGIONS,
        GUARD_DIRECTION_REGIONS,
        RUNNER_DIRECTION_REGIONS,
        GhostlineRenderer,
    )

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    sim.explored[:] = True
    renderer = GhostlineRenderer(sim, visible=False)
    assert renderer._character_atlas is not None
    assert set(RUNNER_DIRECTION_REGIONS) == set(GUARD_DIRECTION_REGIONS) == set(DRONE_DIRECTION_REGIONS) == set(range(8))
    runner = renderer._character_sprite("runner_s", 32)
    assert runner is not None and runner.get_height() == 32
    source = renderer._character_atlas.subsurface(CHARACTER_REGIONS["runner_s"].rect)
    source_colors = set(map(tuple, pygame.surfarray.array3d(source).reshape(-1, 3)))
    scaled_colors = set(map(tuple, pygame.surfarray.array3d(runner).reshape(-1, 3)))
    assert scaled_colors <= source_colors

    atlas_frame = renderer.draw(return_array=True)
    renderer._character_atlas = None
    fallback_frame = renderer.draw(return_array=True)
    renderer.close()
    assert atlas_frame.shape == fallback_frame.shape == (360, 640, 3)
    assert float(np.mean(np.abs(atlas_frame.astype(np.int16) - fallback_frame.astype(np.int16)))) > 0.01


def test_visible_security_cones_match_real_detection_contract(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import math

    from ghostline.presentation import (
        CAMERA_VIEW_DISTANCE,
        CAMERA_VIEW_HALF_ANGLE,
        GUARD_VIEW_ALERT_DISTANCE,
        GUARD_VIEW_BASE_DISTANCE,
        GUARD_VIEW_HALF_ANGLE,
        GhostlineRenderer,
    )

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    sim.level.cameras[:] = sim.level.cameras[:1]
    sim.level.guards[:] = sim.level.guards[:1]
    sim.level.cameras[0].position = sim.player.copy()
    sim.level.guards[0].position = sim.player.copy()
    renderer = GhostlineRenderer(sim, visible=False)
    calls: list[tuple[float, float, str]] = []

    def capture_cone(*args, **kwargs):
        calls.append((float(args[3]), float(args[4]), str(kwargs["pattern"])))

    monkeypatch.setattr(renderer, "_cone", capture_cone)
    renderer._draw_security_cones()
    renderer.close()

    camera = next(call for call in calls if call[2] == "camera")
    guard = next(call for call in calls if call[2] == "guard")
    assert camera[:2] == (CAMERA_VIEW_DISTANCE, CAMERA_VIEW_HALF_ANGLE)
    assert math.isclose(CAMERA_VIEW_HALF_ANGLE, math.acos(0.72))
    assert guard[0] == GUARD_VIEW_BASE_DISTANCE + GUARD_VIEW_ALERT_DISTANCE * sim.alert_tier
    assert math.isclose(GUARD_VIEW_HALF_ANGLE, math.acos(0.62))

    visibility_calls: list[dict[str, float]] = []

    def capture_visibility(*args, **kwargs):
        visibility_calls.append(kwargs)
        return False

    monkeypatch.setattr(sim, "visible", capture_visibility)
    sim._update_cameras(1.0 / 60.0)
    camera_contract = visibility_calls[-1]
    visibility_calls.clear()
    sim._update_guards(1.0 / 60.0)
    guard_contract = visibility_calls[0]
    assert camera_contract["distance"] == CAMERA_VIEW_DISTANCE
    assert math.isclose(camera_contract["cosine"], math.cos(CAMERA_VIEW_HALF_ANGLE))
    assert guard_contract["distance"] == GUARD_VIEW_BASE_DISTANCE + GUARD_VIEW_ALERT_DISTANCE * sim.alert_tier
    assert math.isclose(guard_contract["cosine"], math.cos(GUARD_VIEW_HALF_ANGLE))


def test_hidden_guard_memory_freezes_instead_of_tracking_through_walls(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import BG, GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    guard = sim.level.guards[0]
    guard.position = sim.player + np.asarray((64.0, 0.0), dtype=np.float32)
    renderer = GhostlineRenderer(sim, visible=False)
    visible = True
    monkeypatch.setattr(sim, "player_can_see", lambda _position, **_kwargs: visible)
    renderer._update_security_memory()
    memory = renderer._security_memory[("guard", guard.guard_id)]
    remembered_position = memory.position.copy()
    remembered_tick = memory.last_seen_tick

    visible = False
    sim.elapsed_ticks += 180
    guard.position += np.asarray((96.0, 64.0), dtype=np.float32)
    guard.facing += 1.25
    renderer._update_security_memory()
    memory = renderer._security_memory[("guard", guard.guard_id)]
    assert np.array_equal(memory.position, remembered_position)
    assert memory.last_seen_tick == remembered_tick

    renderer.logical.fill(BG)
    renderer.camera = sim.player.copy()
    renderer._draw_security_memories()
    remembered_screen = renderer._world(remembered_position)
    patch = np.transpose(pygame.surfarray.array3d(renderer.logical), (1, 0, 2))[
        remembered_screen[1] - 28 : remembered_screen[1] + 40,
        remembered_screen[0] - 40 : remembered_screen[0] + 40,
    ]
    assert patch.size and np.any(patch != np.asarray(BG, dtype=np.uint8))
    renderer.reset_for_sim(GhostlineSimulation(seed=2_000_005, tier=6))
    assert renderer._security_memory == {}
    renderer.close()


def test_audible_guard_replaces_stale_ghost_with_grade_and_status_cue(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    guard = sim.level.guards[0]
    sim.level.guards[:] = [guard]
    guard.grade = GuardGrade.INTERCEPTOR
    guard.position = sim.player + np.asarray((64.0, 0.0), dtype=np.float32)
    visible = True
    monkeypatch.setattr(sim, "player_can_see", lambda _position, **_kwargs: visible)
    renderer = GhostlineRenderer(sim, visible=False)
    renderer._update_security_memory()
    assert ("guard", guard.guard_id) in renderer._security_memory

    visible = False
    guard.mode = GuardMode.CHASE
    guard.velocity[:] = (48.0, 0.0)
    memory_sprites: list[int] = []
    remembered_glyphs: list[str] = []
    labels: list[str] = []
    monkeypatch.setattr(
        renderer,
        "_guard_atlas_sprite",
        lambda *_args, **_kwargs: memory_sprites.append(guard.guard_id),
    )
    monkeypatch.setattr(
        renderer,
        "_draw_threat_glyph",
        lambda _surface, kind, _center, _color: remembered_glyphs.append(kind),
    )
    monkeypatch.setattr(renderer, "_text", lambda value, *_args, **_kwargs: labels.append(str(value)))

    renderer._draw_security_memories()
    renderer._draw_alert_indicators()

    assert renderer._memory_superseded_by_audio(("guard", guard.guard_id))
    assert memory_sprites == []
    assert "guard" not in remembered_glyphs
    assert remembered_glyphs == ["sound"]
    assert "STEPS II / CHASE" in labels
    renderer.close()


def test_returning_guard_uses_calm_pose_without_suspicion_symbol(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    renderer = GhostlineRenderer(sim, visible=False)
    states: list[str] = []
    labels: list[str] = []
    monkeypatch.setattr(renderer, "_guard_atlas_sprite", lambda *_args, **_kwargs: None)

    def actor_sprite(_kind, _facing, _moving, state):
        states.append(state)
        return pygame.Surface((2, 2), pygame.SRCALPHA)

    monkeypatch.setattr(renderer, "_actor_sprite", actor_sprite)
    monkeypatch.setattr(renderer, "_text", lambda value, *_args, **_kwargs: labels.append(str(value)))
    renderer._draw_guard(
        sim.player,
        0.0,
        GuardMode.RETURN,
        np.asarray((24.0, 0.0), dtype=np.float32),
        grade=GuardGrade.STANDARD,
    )

    assert states == ["normal"]
    assert "?" not in labels and "!" not in labels
    renderer.close()


def test_visible_terminal_zone_uses_literal_simulation_hack_radius(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.config import HACK_RADIUS
    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    terminal = sim.level.terminals[0]
    sim.player[:] = terminal.position
    for other in sim.level.terminals[1:]:
        other.completed = True
    renderer = GhostlineRenderer(sim, visible=False)
    renderer.camera = sim.player.copy()
    terminal_center = renderer._world(terminal.position)
    circles: list[tuple[tuple[int, int], int, int]] = []
    original_circle = pygame.draw.circle

    def capture_circle(surface, color, center, radius, width=0, *args, **kwargs):
        circles.append((tuple(center), int(radius), int(width)))
        return original_circle(surface, color, center, radius, width, *args, **kwargs)

    monkeypatch.setattr(pygame.draw, "circle", capture_circle)
    renderer._draw_objectives()

    assert (terminal_center, int(round(HACK_RADIUS)), 1) in circles
    renderer.close()


def test_cone_raycast_stops_before_first_simulation_occluder(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import math

    from ghostline.config import TILE_SIZE
    from ghostline.presentation import GhostlineRenderer
    from ghostline.types import Tile

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    renderer = GhostlineRenderer(sim, visible=False)
    points, ray_distances = renderer._cone_points(sim.player, 0.0, 220.0, 0.75)
    assert len(points) == 66
    assert len(ray_distances) == 65
    result = None
    for angle in np.linspace(0.0, math.tau, 48, endpoint=False):
        direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float32)
        distance = renderer._cone_raycast_distance(sim.player, direction, 420.0)
        if distance < 419.0:
            result = direction, distance
            break
    assert result is not None
    direction, distance = result
    before = sim.player + direction * distance
    after = sim.player + direction * (distance + 1.5)
    before_tile = (int(before[0] // TILE_SIZE), int(before[1] // TILE_SIZE))
    after_tile = (int(after[0] // TILE_SIZE), int(after[1] // TILE_SIZE))
    assert before_tile not in sim._blocked_tiles
    assert sim.level.grid[before_tile[1], before_tile[0]] != Tile.WALL
    assert after_tile in sim._blocked_tiles or sim.level.grid[after_tile[1], after_tile[0]] == Tile.WALL
    renderer.close()


def test_awareness_has_distinct_shape_and_meter_states(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import BG, GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    renderer = GhostlineRenderer(sim, visible=False)
    renderer.logical.fill(BG)
    renderer._draw_awareness_badge(100, 100, kind="camera", pressure=0.42)
    camera_partial = pygame.surfarray.array3d(renderer.logical).copy()
    renderer.logical.fill(BG)
    renderer._draw_awareness_badge(100, 100, kind="guard", pressure=1.0, confirmed=True)
    guard_confirmed = pygame.surfarray.array3d(renderer.logical).copy()
    renderer.close()

    assert np.count_nonzero(np.any(camera_partial != np.asarray(BG), axis=2)) > 30
    assert np.count_nonzero(np.any(guard_confirmed != np.asarray(BG), axis=2)) > 30
    assert not np.array_equal(camera_partial, guard_confirmed)


def test_diagonal_locomotion_preserves_direction_and_animates_integer_frames(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=2_000_004, tier=6), visible=False)
    renderer._time = 0.01
    southeast_contact = renderer._runner_atlas_sprite(np.pi / 4.0, True, "normal")
    renderer._time = 0.11
    southeast_recoil = renderer._runner_atlas_sprite(np.pi / 4.0, True, "normal")
    northeast_recoil = renderer._runner_atlas_sprite(-np.pi / 4.0, True, "normal")

    assert southeast_contact is not None and southeast_recoil is not None and northeast_recoil is not None
    assert southeast_contact.get_height() == southeast_recoil.get_height() == 32
    contact_pixels = pygame.surfarray.array_alpha(southeast_contact)
    recoil_pixels = pygame.surfarray.array_alpha(southeast_recoil)
    northeast_pixels = pygame.surfarray.array_alpha(northeast_recoil)
    assert not np.array_equal(contact_pixels, recoil_pixels)
    assert not np.array_equal(recoil_pixels, northeast_pixels)

    renderer._diagonal_locomotion_atlas = None
    renderer._time = 0.11
    fallback = renderer._runner_atlas_sprite(np.pi / 4.0, True, "normal")
    assert fallback is not None and fallback.get_height() == 36
    renderer.close()


def test_debrief_keeps_earned_badge_names_on_atomic_lines(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from ghostline.app import GameApp
    from ghostline.progression import DEFAULT_BINDINGS
    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_123, tier=6)
    sim.extracted = True
    sim.data = sim.level.quota + 2
    sim.optional_data = 2
    sim.detections = 0
    sim.damage_taken = 0
    renderer = GhostlineRenderer(sim, visible=False)
    app = GameApp.__new__(GameApp)
    app.sim = sim
    app.renderer = renderer
    app.selected_tier = 6
    app._debrief_agent = False
    app._telemetry = {"path_efficiency": 0.86, "idle_fraction": 0.02}
    app.settings = {"bindings": dict(DEFAULT_BINDINGS)}
    monkeypatch.setattr(app, "_events", lambda: [])
    captured: dict[str, object] = {}
    monkeypatch.setattr(renderer, "draw_screen", lambda **kwargs: captured.update(kwargs))

    app._debrief()
    body = captured["body"]
    assert isinstance(body, list)
    assert "BADGES      GHOST // NO DAMAGE" in body
    assert "OPTIONAL DATA // EFFICIENT ROUTE" in body
    assert len(body) == 9
    badge_lines = body[6:8]
    assert all(len(renderer._wrap_text(line, renderer.font, 305)) == 1 for line in badge_lines)

    sim.extracted = False
    sim.fail_reason = "contract_expired"
    captured.clear()
    app._debrief()
    failed_body = captured["body"]
    assert isinstance(failed_body, list)
    assert "FAILURE     CONTRACT EXPIRED" in failed_body
    assert not any(line.startswith("BADGES") for line in failed_body)
    renderer.close()
