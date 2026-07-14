---
title: Ghostline Assets
updated: 2026-07-13
status: active
---

# Visual and audio workflow

Ghostline uses a high-detail top-down three-quarter pixel style on a 32 px world grid. Gameplay renders to a 640x360 logical canvas and uses exact integer scaling at 1280x720 and 1920x1080, with centered letterboxing for other window shapes.

## Visual disclosure

The AI-assisted cinematic key-art draft under `assets/visual/` is retained only as process provenance. Hands-on review found that its pseudo-isometric depth promised a different game from the shipping top-down 2D playfield, so it is no longer loaded by menus and is excluded from wheel, Windows, and web runtime payloads. Every title, briefing, pause, and debrief screen now uses a code-native flat facility schematic in the same palette, scale language, and security grammar as gameplay. `assets/licenses.json` records both the retirement and the original generation process. Project-owned visual and synthesized-audio assets ship under the repository's MIT License.

Generated imagery is never collision, navigation, visibility, or simulation truth.

## Runtime pixel system

- Runner and guards use reviewed atlases for eight-direction facing and state animation, including dedicated four-frame diagonal run cycles; an original runtime-built eight-direction, four-frame pixel path remains the complete fallback.
- Room roles have distinct deterministic floor materials, wall trims, signage, vents, cabling, grates, warning stripes, rugs, vault inlays, and extraction markings.
- Furnishings cover desks, chairs, meeting and coffee tables, sofas, TVs, lab benches, monitors, server racks, consoles, lockers, plants, crates, generators, and vault cases. Animated displays and server lights remain presentation-only.
- The playfield has no square exploration-fog wash: discovered-state still feeds fair observations and the compact map, while floor, furniture, and wall art stays readable at full authored color. Exact guard/camera entities remain line-of-sight gated.
- Security shapes, segmented awareness badges, exact 65-ray occlusion-clipped cones, typed off-screen arrows, sound labels, terminal progress rings, pulse waves, damage vignettes, and lockdown banners communicate state without privileged enemy information. Cameras pair a square with a dashed scan beam; guards pair a triangle with notched cone edges, so both remain distinct in monochrome and color-safe modes.
- The renderer caches the room-role lookup for each generated level while keeping all Pygame state outside simulation and generation.

## Environment atlas v1

`assets/visual/ghostline-environment-atlas-v1.png` is the alpha-clean 1672x941 runtime atlas. The retained `ghostline-environment-atlas-source-v1.png` is a provenance draft and is deliberately excluded from Windows and web release payloads. Runtime crops cover desks, consoles, chairs, plants, sofas, laboratory benches, servers, lockers, cameras, terminals, vault cases, crates, and generators. Each crop carries explicit bounds and a maximum logical width in `presentation.py`; the renderer uses nearest-neighbor scaling, aligns the sprite's bottom pivot to its authored collision footprint, and draws a one-pixel semantic outline in High Contrast mode. Missing atlases and unmapped prop types fall back to the code-native renderer. Collision, navigation, placement, and security logic never read the atlas.

Generation used the approved menu key art as the palette/material/perspective reference and the 640x360 gameplay capture as the practical-scale reference. The final prompt was:

> Create one original clean 16:9 production concept sheet for a modular near-future stealth-facility tileset: separated 32 px-grid top-down three-quarter floor, wall, corner, door, window, trim, office, lounge, laboratory, server, security, vault, and utility assets; dark navy and steel with restrained cyan, amber, and red accents; consistent perspective and practical silhouettes; perfectly flat solid #ff00ff chroma-key background with no shadows, gradients, texture, glow, text, logos, watermark, characters, weapons, overlaps, cropping, or copyrighted franchise references.

The built-in image generator produced the chroma-key source. Cleanup removed the key locally, validated transparent corners and edge coverage, retained the original source non-destructively, and produced the versioned alpha runtime atlas. Representative room, security, fallback, and accessibility renders are kept under `artifacts/visual-qa/atlas-final/`.

## Character and security atlas v1

`assets/visual/ghostline-character-security-atlas-v1.png` is the alpha-clean 1672x941 runtime character sheet. Its non-destructive chroma-key source is retained as `ghostline-character-security-atlas-source-v1.png` and excluded from Windows/web runtime payloads by the same source-draft filters used for environment art.

The sheet supplies cyan runner directions and run/dash/link/damage states, red-accent unarmed guard directions and patrol/suspicious/chase/tackle-strike states, violet response-drone directions and charge/recoil states, and active/suspicious/disabled/damaged wall-camera states. Explicit crop rectangles, mirrored completion for the eight semantic movement directions, nearest-neighbour scaling, and bottom pivots live in `presentation.py`. State selection consumes existing public presentation state only; the atlas cannot alter movement, collision, damage timing, guard decisions, visibility, or policy observations. If the sheet is absent, every actor and device returns to the original procedural sprite path.

The final generation prompt was:

> Create one original clean 16:9 Ghostline character and electronic-security pixel-sprite sheet on a perfectly flat #ff00ff chroma-key background: separated top-down three-quarter cyan runner direction and idle/run/dash/link/damage poses; red-accent unarmed human guard directions and patrol/suspicious/chase/tackle-strike poses; violet response-drone direction/charge/recoil variants; and active, suspicious, disabled, and damaged wall-camera variants. Use practical readable 24-32 px silhouettes, consistent scale, bottom pivots, navy/steel materials, no background shadows or gradients, no text, logos, watermark, weapons, blood, environment tiles, or copyrighted references.

Cleanup used the installed image-generation chroma-key helper with soft matte and despill, then validated transparent corners, edge coverage, practical 640x360 scale, eight-direction mappings, action readability, High Contrast outlines, and the no-atlas fallback. The inspected runtime matrix is under `artifacts/visual-qa/character-final/`.

## Diagonal locomotion atlas v2

`assets/visual/ghostline-diagonal-locomotion-v2.png` removes the earlier presentation shortcut that reused a side-on run strip for diagonal movement. It contains four-frame northeast and southeast cycles for both the cyan runner and red guard; northwest and southwest use deterministic mirroring. Every frame has an explicit alpha-clean crop, nearest-neighbour scaling, and a stable world-foot pivot. North/south movement uses the direction-preserving integer-pixel procedural gait, and east/west retains the authored side run. Reduced Motion freezes locomotion on a single directional pose.

The built-in image generator used the v1 character source solely as a style/proportion reference. The selected sheet was generated on flat magenta, copied into the project, cleaned with the installed chroma-key helper, and inspected at original resolution before integration. The source remains as `ghostline-diagonal-locomotion-source-v2.png` for provenance and is excluded from wheel, Windows, and web runtime bundles. If the v2 derivative cannot load, the renderer automatically uses the direction-preserving procedural gait; simulation facing, collision, navigation, and observations never depend on either asset.

## Accessibility palette

Default hazards use amber for suspicion and red for confirmed danger. Color-safe mode remaps danger to pink and extraction/success to blue while preserving distinct symbols, outlines, captions, and motion-independent shapes. High-contrast mode expands luminance separation after composition. Reduced Motion disables camera look-ahead, shake, afterimages, and moving menu art; Reduced Flashes removes pulsing lockdown intensity and limits event particles.

## Release-scale QA

`scripts/qa_scaled_visuals.py` freezes representative title, briefing, Field Manual, pause, debrief, settings, Accessibility, Agent Lab selection/live, and tier-6 gameplay frames, then presents each through the shipping renderer at 1280x720 and 1920x1080. The world layer remains nearest-neighbour pixel art; the script now requires a non-zero post-scale text difference and native glyph runs in every scene, proving that menus, HUD, captions, and telemetry are rerasterized at output resolution rather than enlarging 640x360 glyph pixels. The earlier 2026-07-13 v4 matrix under `artifacts/visual-qa/flat-vision-grades-v4/` remains the pre-native-text exact-scaling baseline. Reviewed gameplay and locomotion captures are tracked under `assets/screenshots/`; visible-window and Chrome checks remain separate release gates.

The final frozen-gameplay pass is under `artifacts/visual-qa/final-gameplay-readability-2026-07-13/`. Its 20-frame release matrix again recorded exact 2x/3x scaling with zero mismatched pixels, and measured 89.46 FPS at 1280x720 and 77.19 FPS at 1920x1080 over 120 hidden-surface frames. Focused captures verify frozen `LAST` snapshots and uncertainty rings, quantized `STEPS` direction, the 185 px dash-noise ring, `EYE/SOUND/RADIO` guard causes, trace threshold transitions, link-time feedback, fully visible one-cell server fixtures, and whole-name debrief badge lines. Interactive-window and Chrome checks remain separate gates.

## Audio

All sound effects, ambient pads, and the trace-responsive tension layer are synthesized at runtime from original waveforms. Master, music, and effects volumes are independently adjustable. Optional sound captions identify terminal handshakes, alerts, impacts, pulses, drone rotors, and extraction events. There are no external audio attribution requirements; any future imported sound must be added to `assets/licenses.json`.
