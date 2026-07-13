from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from ghostline.types import SimEvent


class AudioDirector:
    """Original procedural SFX and ambient score; no external audio assets."""

    def __init__(self, *, enabled: bool = True):
        self.enabled = enabled
        self.ready = False
        self.sounds: dict[str, object] = {}
        self.master_volume = 0.75
        self.music_volume = 0.55
        self.sfx_volume = 0.85
        self.volume = self.master_volume  # Backwards-compatible public alias.
        self._trace_mix = 0.0
        self._lockdown_mix = 0.0
        self._last_played: dict[str, float] = {}
        self._last_step = 0.0
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44_100, size=-16, channels=2, buffer=512)
            self.pygame = pygame
            self.sounds = self._build_bank()
            self._base_volumes = {key: sound.get_volume() for key, sound in self.sounds.items()}
            self.ambient = self._sound(self._ambient_wave(8.0), volume=1.0)
            self.tension = self._sound(self._tension_wave(8.0), volume=1.0)
            self.ambient.play(loops=-1, fade_ms=800)
            self.tension.play(loops=-1, fade_ms=1000)
            self.set_mix(master=self.master_volume, music=self.music_volume, sfx=self.sfx_volume)
            self.ready = True
        except Exception:
            self.ready = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if self.ready:
            self.pygame.mixer.pause() if not enabled else self.pygame.mixer.unpause()

    def set_volume(self, volume: float) -> None:
        """Set master volume (kept for compatibility with older callers)."""

        self.set_mix(master=volume)

    def set_mix(
        self,
        *,
        master: float | None = None,
        music: float | None = None,
        sfx: float | None = None,
    ) -> None:
        if master is not None:
            self.master_volume = max(0.0, min(1.0, float(master)))
            self.volume = self.master_volume
        if music is not None:
            self.music_volume = max(0.0, min(1.0, float(music)))
        if sfx is not None:
            self.sfx_volume = max(0.0, min(1.0, float(sfx)))
        if not hasattr(self, "_base_volumes"):
            return
        for key, sound in self.sounds.items():
            sound.set_volume(self._base_volumes[key] * self.master_volume * self.sfx_volume)
        self._apply_music_mix()

    def update(self, *, trace: float, lockdown: bool, speed: float = 0.0) -> None:
        """Crossfade the procedural score with security pressure."""

        target_trace = max(0.0, min(1.0, float(trace) / 100.0))
        self._trace_mix += (target_trace - self._trace_mix) * 0.035
        target_lockdown = 1.0 if lockdown else 0.0
        self._lockdown_mix += (target_lockdown - self._lockdown_mix) * 0.05
        self._apply_music_mix()
        if self.ready and self.enabled and speed > 22.0:
            now = time.monotonic()
            cadence = 0.18 if speed > 170.0 else 0.31
            if now - self._last_step >= cadence:
                self.sounds["step"].play()
                self._last_step = now

    def _apply_music_mix(self) -> None:
        if not (hasattr(self, "ambient") and hasattr(self, "tension")):
            return
        music = self.master_volume * self.music_volume
        tension = max(self._trace_mix * 0.8, self._lockdown_mix)
        self.ambient.set_volume(music * (0.13 - 0.035 * tension))
        self.tension.set_volume(music * (0.02 + 0.16 * tension))

    def handle(self, events: Iterable[SimEvent]) -> None:
        if not (self.ready and self.enabled):
            return
        mapping = {
            "dash": "dash",
            "hack_tick": "hack",
            "hack_complete": "complete",
            "quota_met": "quota",
            "detected": "alert",
            "pulse": "pulse",
            "damage": "damage",
            "lockdown": "lockdown",
            "extracted": "extract",
            "failure": "failure",
            "drone_deployed": "alert",
        }
        for event in events:
            key = mapping.get(event.kind)
            if key and key in self.sounds:
                now = time.monotonic()
                cooldown = {"hack": 0.16, "alert": 0.28, "damage": 0.25, "dash": 0.08}.get(key, 0.02)
                if now - self._last_played.get(key, -math.inf) >= cooldown:
                    self.sounds[key].play()
                    self._last_played[key] = now

    def menu_move(self) -> None:
        if self.ready and self.enabled:
            self.sounds["menu"].play()

    def menu_confirm(self) -> None:
        if self.ready and self.enabled:
            self.sounds["confirm"].play()

    def close(self) -> None:
        if self.ready:
            self.pygame.mixer.stop()

    def _build_bank(self) -> dict[str, object]:
        return {
            "menu": self._sound(self._tone(620, 0.045, decay=22), 0.18),
            "confirm": self._sound(self._chord((420, 630), 0.09), 0.22),
            "step": self._sound(self._noise(0.035, 0.25) + self._tone(82, 0.035, decay=34), 0.075),
            "dash": self._sound(self._noise(0.10, 0.55) + self._tone(115, 0.10, decay=18), 0.22),
            "hack": self._sound(self._tone(760, 0.045, decay=30), 0.10),
            "complete": self._sound(self._sequence((440, 660, 880), 0.09), 0.24),
            "quota": self._sound(self._sequence((330, 495, 660, 990), 0.12), 0.28),
            "alert": self._sound(self._sequence((170, 125), 0.18), 0.27),
            "pulse": self._sound(self._tone(72, 0.50, decay=4) + self._noise(0.50, 0.16), 0.28),
            "damage": self._sound(self._noise(0.20, 0.8) + self._tone(85, 0.20, decay=10), 0.30),
            "lockdown": self._sound(self._sequence((95, 95, 74), 0.32), 0.32),
            "extract": self._sound(self._chord((220, 330, 440, 660), 1.0), 0.30),
            "failure": self._sound(self._sequence((220, 165, 110), 0.35), 0.28),
            "focus": self._sound(self._sequence((740, 880), 0.055), 0.13),
        }

    def _sound(self, mono: np.ndarray, volume: float = 0.2):
        stereo = np.column_stack((mono, mono))
        audio = np.asarray(np.clip(stereo * 32767, -32767, 32767), dtype=np.int16)
        sound = self.pygame.sndarray.make_sound(audio)
        sound.set_volume(volume)
        return sound

    @staticmethod
    def _tone(frequency: float, seconds: float, *, decay: float = 5.0) -> np.ndarray:
        samples = max(1, int(44_100 * seconds))
        time = np.arange(samples, dtype=np.float32) / 44_100
        envelope = np.exp(-decay * time)
        return np.sin(math.tau * frequency * time) * envelope

    @staticmethod
    def _noise(seconds: float, amplitude: float) -> np.ndarray:
        rng = np.random.default_rng(9173)
        samples = max(1, int(44_100 * seconds))
        time = np.arange(samples, dtype=np.float32) / 44_100
        return rng.uniform(-1.0, 1.0, samples).astype(np.float32) * amplitude * np.exp(-12 * time)

    def _chord(self, frequencies: tuple[float, ...], seconds: float) -> np.ndarray:
        waves = [self._tone(frequency, seconds, decay=2.5) for frequency in frequencies]
        return sum(waves) / max(1, len(waves))

    def _sequence(self, frequencies: tuple[float, ...], seconds: float) -> np.ndarray:
        return np.concatenate([self._tone(frequency, seconds, decay=8.0) for frequency in frequencies])

    def _ambient_wave(self, seconds: float) -> np.ndarray:
        samples = int(44_100 * seconds)
        time = np.arange(samples, dtype=np.float32) / 44_100
        pad = 0.16 * np.sin(math.tau * 55 * time) + 0.08 * np.sin(math.tau * 82.5 * time)
        pulse = (0.5 + 0.5 * np.sin(math.tau * 0.125 * time)) * 0.05 * np.sin(math.tau * 220 * time)
        return np.asarray(pad + pulse, dtype=np.float32)

    def _tension_wave(self, seconds: float) -> np.ndarray:
        samples = int(44_100 * seconds)
        time = np.arange(samples, dtype=np.float32) / 44_100
        throb = (0.35 + 0.65 * np.square(np.sin(math.tau * 0.75 * time))) * np.sin(math.tau * 46 * time)
        alarm = np.sin(math.tau * 96 * time + 0.8 * np.sin(math.tau * 0.125 * time))
        texture = (np.sin(math.tau * 313 * time) + np.sin(math.tau * 487 * time)) * (0.5 + 0.5 * np.sin(math.tau * 0.25 * time))
        return np.asarray(0.18 * throb + 0.05 * alarm + 0.025 * texture, dtype=np.float32)
