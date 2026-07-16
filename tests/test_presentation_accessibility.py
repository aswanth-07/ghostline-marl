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
    ambient_volume = audio.ambient_channel.get_volume()
    audio.set_mix(music=1.0, sfx=0.1)

    assert audio.sounds["menu"].get_volume() < menu_volume
    assert audio.ambient_channel.get_volume() > ambient_volume
    audio.close()


def test_audio_waits_for_gameplay_and_reconnect_replaces_owned_loops(monkeypatch) -> None:
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.audio import AudioDirector

    pygame.mixer.quit()
    first = AudioDirector(enabled=True)
    assert first.ready
    assert first._music_started is False
    assert first.ambient_channel.get_sound() is None
    assert first.tension_channel.get_sound() is None

    first.set_gameplay_active(True)
    first.update(trace=0.0, lockdown=False)
    assert first._music_started is True
    assert first.ambient_channel.get_sound() is first.ambient
    assert first.tension_channel.get_sound() is first.tension

    second = AudioDirector(enabled=True)
    assert first.ready is False
    assert second.ready is True
    assert second._music_started is False
    assert second.ambient_channel.get_sound() is None
    assert second.tension_channel.get_sound() is None
    second.set_gameplay_active(True)
    second.update(trace=50.0, lockdown=False)
    assert pygame.mixer.Channel(0).get_sound() is second.ambient
    assert pygame.mixer.Channel(1).get_sound() is second.tension

    # Closing a retired director cannot stop the replacement's channels.
    first.close()
    assert pygame.mixer.Channel(0).get_sound() is second.ambient
    assert pygame.mixer.Channel(1).get_sound() is second.tension
    second.close()


def test_audio_focus_and_gameplay_state_pause_only_the_owned_score() -> None:
    from ghostline.audio import AudioDirector

    class Channel:
        def __init__(self, sound):
            self.sound = sound
            self.paused = False

        def get_sound(self):
            return self.sound

        def pause(self) -> None:
            self.paused = True

        def unpause(self) -> None:
            self.paused = False

    audio = AudioDirector.__new__(AudioDirector)
    audio.ready = True
    audio.enabled = True
    audio._gameplay_active = True
    audio._focus_active = True
    audio.ambient = object()
    audio.tension = object()
    audio.ambient_channel = Channel(audio.ambient)
    audio.tension_channel = Channel(audio.tension)
    audio.sounds = {}
    audio._sfx_channels = []

    audio.set_focus_active(False)
    assert audio.ambient_channel.paused is True
    assert audio.tension_channel.paused is True
    audio.set_focus_active(True)
    assert audio.ambient_channel.paused is False
    audio.set_gameplay_active(False)
    assert audio.ambient_channel.paused is True


def test_procedural_score_has_baked_headroom_and_no_mains_hum_fundamental() -> None:
    from ghostline.audio import AudioDirector

    audio = AudioDirector.__new__(AudioDirector)
    ambient = audio._ambient_wave(8.0)
    tension = audio._tension_wave(8.0)

    assert float(np.sqrt(np.mean(np.square(ambient)))) < 0.05
    assert float(np.sqrt(np.mean(np.square(tension)))) < 0.05
    assert float(np.max(np.abs(ambient))) < 0.1
    assert float(np.max(np.abs(tension))) < 0.13

    frequencies = np.fft.rfftfreq(len(ambient), d=1.0 / audio._sample_rate)
    ambient_spectrum = np.abs(np.fft.rfft(ambient))
    low_band = float(np.sum(np.square(ambient_spectrum[frequencies < 70.0])))
    total = float(np.sum(np.square(ambient_spectrum)))
    assert low_band / total < 1e-6


def test_renderer_does_not_preinitialize_the_audio_mixer(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.presentation import GhostlineRenderer

    pygame.mixer.quit()
    renderer = GhostlineRenderer(GhostlineSimulation(seed=37, tier=1), visible=False)
    assert pygame.mixer.get_init() is None
    renderer.close()


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


def test_sound_captions_are_opt_in_to_keep_the_default_hud_uncluttered(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from ghostline.presentation import GhostlineRenderer
    from ghostline.progression import DEFAULT_SETTINGS

    renderer = GhostlineRenderer(GhostlineSimulation(seed=42, tier=3), visible=False)
    assert renderer.sound_captions_enabled is False
    assert DEFAULT_SETTINGS["accessibility"]["sound_captions"] is False
    renderer.close()


def test_pointer_selects_the_clicked_contract_instead_of_only_hovering(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    import pygame

    from ghostline.app import GameApp
    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=7, tier=1), visible=False)
    renderer.window = pygame.display.set_mode((640, 360))

    class SilentAudio:
        def menu_move(self) -> None:
            pass

        def menu_confirm(self) -> None:
            pass

    app = GameApp.__new__(GameApp)
    app.renderer = renderer
    app.audio = SilentAudio()
    app.selection = 0
    app.touch_controls_enabled = False
    app.settings = {"bindings": dict(DEFAULT_BINDINGS)}
    clicked = renderer.menu_item_rect(4).center
    event = pygame.event.Event(pygame.MOUSEBUTTONUP, {"pos": clicked, "button": 1})

    assert app._menu_events([event], 6) == "confirm"
    assert app.selection == 4
    renderer.close()


def test_touch_controller_maps_diagonal_move_dash_and_pulse_without_changing_action_contract() -> None:
    from ghostline.app import GameApp
    from ghostline.presentation import TOUCH_JOYSTICK_CENTER

    app = GameApp.__new__(GameApp)
    app._touch_roles = {1: "move", 2: "dash", 3: "pulse"}
    app._touch_points = {
        1: (TOUCH_JOYSTICK_CENTER[0] + 30, TOUCH_JOYSTICK_CENTER[1] - 30),
        2: (0.0, 0.0),
        3: (0.0, 0.0),
    }

    action = app._touch_action()
    assert action.move == 2
    assert action.dash is True
    assert action.pulse is True
    assert 0 <= action.encode() < 36


def test_touch_overlay_is_visible_and_headless_safe(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer, TOUCH_JOYSTICK_CENTER

    renderer = GhostlineRenderer(GhostlineSimulation(seed=8, tier=2), visible=False)
    plain = renderer.draw(return_array=True)
    touch = renderer.draw(
        return_array=True,
        touch_controls={"move_point": TOUCH_JOYSTICK_CENTER, "dash": True, "pulse": False},
    )
    renderer.close()

    assert plain.shape == touch.shape == (360, 640, 3)
    assert not np.array_equal(plain, touch)


def test_touch_hud_uses_a_concise_phone_readable_status_strip(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=8, tier=2), visible=False)
    labels: list[str] = []
    original_text = renderer._text

    def tracked_text(text, x, y, font, color):
        labels.append(str(text))
        original_text(text, x, y, font, color)

    monkeypatch.setattr(renderer, "_text", tracked_text)
    minimap_calls = 0

    def tracked_minimap():
        nonlocal minimap_calls
        minimap_calls += 1

    monkeypatch.setattr(renderer, "_draw_minimap", tracked_minimap)
    renderer.logical.fill((0, 0, 0))
    renderer._draw_hud(None, touch_layout=True)
    renderer.close()

    assert any(label.startswith("T2  SURVEILLANCE") for label in labels)
    assert any(label.startswith("ACQUIRE  ") for label in labels)
    assert "HP" in labels
    assert "TRACE" in labels
    assert "DASH" in labels
    assert "INTEGRITY" not in labels
    assert minimap_calls == 0


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


def test_security_telemetry_keeps_guard_position_live_through_walls(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=2_000_004, tier=6)
    guard = sim.level.guards[0]
    guard.position = sim.player + np.asarray((64.0, 0.0), dtype=np.float32)
    renderer = GhostlineRenderer(sim, visible=False)
    visible = True
    monkeypatch.setattr(sim, "player_can_see", lambda _position, **_kwargs: visible)
    renderer._update_security_memory()
    memory = renderer._security_memory[("guard", guard.guard_id)]
    visible = False
    sim.elapsed_ticks += 180
    guard.position += np.asarray((96.0, 64.0), dtype=np.float32)
    guard.facing += 1.25
    renderer._update_security_memory()
    memory = renderer._security_memory[("guard", guard.guard_id)]
    assert np.array_equal(memory.position, guard.position)
    assert memory.last_seen_tick == sim.elapsed_ticks
    renderer.reset_for_sim(GhostlineSimulation(seed=2_000_005, tier=6))
    assert renderer._security_memory
    assert all(memory.last_seen_tick == 0 for memory in renderer._security_memory.values())
    renderer.close()


def test_dash_noise_uses_one_bounded_presentation_wave(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    from ghostline.presentation import GhostlineRenderer

    sim = GhostlineSimulation(seed=91, tier=1)
    renderer = GhostlineRenderer(sim, visible=False)
    events = [SimEvent("dash_noise", tuple(sim.player), 185.0) for _ in range(120)]
    renderer.ingest_events(events)

    assert sum(effect.kind == "dash_noise" for effect in renderer.effects) == 1
    assert len(renderer.particles) <= 72
    renderer.close()


def test_browser_presentation_fills_intermediate_canvas_sizes() -> None:
    from ghostline.presentation import _presentation_scaled_size

    assert _presentation_scaled_size((927, 521), web_runtime=True) == (927, 521)
    assert _presentation_scaled_size((927, 521), web_runtime=False) == (640, 360)
    assert _presentation_scaled_size((1920, 1080), web_runtime=False) == (1920, 1080)


def test_visible_renderer_redraws_all_text_at_native_output_resolution(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    import pygame

    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=7, tier=1), visible=True)
    renderer.window = pygame.display.set_mode((1280, 720), pygame.HIDDEN)
    renderer.draw_screen(
        title="GHOSTLINE",
        subtitle="NATIVE INTERFACE",
        items=["PLAY CONTRACTS"],
        return_array=False,
    )

    assert renderer._native_text_commands
    # The 46px logical title is rasterized directly at 92px for a 2x window,
    # rather than enlarging glyph pixels from the 640x360 world surface.
    assert 92 in renderer._native_font_cache
    renderer.close()


def test_live_agent_card_nests_below_the_minimap_instead_of_covering_the_route(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    import pygame

    from ghostline.presentation import BG, GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=7, tier=1), visible=False)
    renderer.logical.fill(BG)
    renderer._draw_hud(None)
    baseline = pygame.surfarray.array3d(renderer.logical).copy()

    renderer.logical.fill(BG)
    renderer._draw_hud(
        {"policy": "RECURRENT ONNX POLICY", "action": "NE +DASH", "latency_ms": 0.93}
    )
    live = pygame.surfarray.array3d(renderer.logical).copy()
    changed = np.any(live != baseline, axis=2)
    xs, ys = np.where(changed)

    assert xs.min() >= 529 and xs.max() <= 628
    assert ys.min() >= 69 and ys.max() <= 102
    assert int(changed.sum()) <= 100 * 34
    assert changed.mean() < 0.02
    renderer.close()


def test_headless_capture_rasterizes_hud_text_at_native_720p(monkeypatch) -> None:
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")

    from ghostline.presentation import GhostlineRenderer

    renderer = GhostlineRenderer(GhostlineSimulation(seed=7, tier=1), visible=False)
    frame = renderer.draw_screen(
        title="GHOSTLINE",
        subtitle="NATIVE CAPTURE",
        items=["PLAY CONTRACTS"],
        return_array=True,
        output_size=(1280, 720),
    )

    assert frame.shape == (720, 1280, 3)
    assert 92 in renderer._native_font_cache
    renderer.close()


def test_watch_agent_showcase_uses_distinct_validation_seeds() -> None:
    from ghostline.app import AGENT_SHOWCASE_SEEDS
    from ghostline.seeds import FINAL_TEST_SEED_START, VALIDATION_SEED_END, VALIDATION_SEED_START

    assert set(AGENT_SHOWCASE_SEEDS) == set(range(1, 7))
    assert len(set(AGENT_SHOWCASE_SEEDS.values())) == 6
    assert all(VALIDATION_SEED_START <= seed <= VALIDATION_SEED_END for seed in AGENT_SHOWCASE_SEEDS.values())
    assert all(seed < FINAL_TEST_SEED_START for seed in AGENT_SHOWCASE_SEEDS.values())


def test_runtime_policy_lookup_is_independent_of_launch_directory(tmp_path, monkeypatch) -> None:
    from ghostline.app import GameApp

    monkeypatch.chdir(tmp_path)
    app = GameApp.__new__(GameApp)
    app.learned_policy = None
    app.policy_name = "FAIR SCRIPTED BASELINE"
    app._load_runtime_policy()

    assert app.learned_policy is not None
    assert app.policy_name == "RECURRENT ONNX POLICY"


def test_occluded_guard_uses_live_sprite_instead_of_stale_ghost(monkeypatch) -> None:
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
    guard.position += np.asarray((48.0, 24.0), dtype=np.float32)
    drawn_positions: list[np.ndarray] = []
    monkeypatch.setattr(
        renderer,
        "_draw_guard",
        lambda position, *_args, **_kwargs: drawn_positions.append(position.copy()),
    )
    renderer._draw_security()

    assert len(drawn_positions) == 1
    assert np.array_equal(drawn_positions[0], guard.position)
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
