from __future__ import annotations

import math
import sys
import time
from typing import Iterable

import numpy as np

from ghostline.types import SimEvent


class AudioDirector:
    """Original procedural SFX and ambient score; no external audio assets."""

    _active_director: AudioDirector | None = None
    _sample_rate = 44_100
    _ambient_channel_index = 0
    _tension_channel_index = 1

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
        self._music_started = False
        self._gameplay_active = False
        self._focus_active = True
        self._sfx_channels: list[object] = []
        self.pygame = None
        self.ambient = None
        self.tension = None
        self.ambient_channel = None
        self.tension_channel = None
        self.activate()

    def activate(self) -> bool:
        """Initialize one idempotent, channel-owned audio graph.

        Music is deliberately not started here.  The browser constructs the
        game immediately after its focus gate opens; starting both infinite
        score buffers at that point made a low drone the first thing a visitor
        heard.  ``update`` starts them on the first actual gameplay tick.
        """

        if self.ready:
            return True
        try:
            import pygame

            if not pygame.mixer.get_init():
                # A larger browser buffer avoids main-thread WASM work starving
                # WebAudio and repeating a tiny buffer as a harsh buzz.  The
                # desktop path remains responsive enough for UI and footsteps.
                buffer = 2048 if sys.platform in {"emscripten", "wasi"} else 1024
                pygame.mixer.init(
                    frequency=self._sample_rate,
                    size=-16,
                    channels=2,
                    buffer=buffer,
                )
            pygame.mixer.set_num_channels(max(16, pygame.mixer.get_num_channels()))
            # Sound.play() cannot steal the two score channels.
            pygame.mixer.set_reserved(2)
            self.pygame = pygame
            self.sounds = self._build_bank()
            self._base_volumes = {key: sound.get_volume() for key, sound in self.sounds.items()}
            self.ambient = self._sound(self._ambient_wave(8.0), volume=1.0)
            self.tension = self._sound(self._tension_wave(8.0), volume=1.0)
            self.ambient_channel = pygame.mixer.Channel(self._ambient_channel_index)
            self.tension_channel = pygame.mixer.Channel(self._tension_channel_index)
            previous = type(self)._active_director
            if previous is not None and previous is not self:
                previous._retire()
            type(self)._active_director = self
            self.ready = True
            self.set_mix(master=self.master_volume, music=self.music_volume, sfx=self.sfx_volume)
            return True
        except Exception:
            self.ready = False
            return False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        if not self.ready:
            return
        if not enabled:
            for channel in self._owned_channels():
                channel.pause()
            return
        for channel in self._owned_sfx_channels():
            channel.unpause()
        self._sync_music_playback()

    def set_gameplay_active(self, active: bool) -> None:
        """Run the continuous score only while a contract is actively ticking."""

        self._gameplay_active = bool(active)
        self._sync_music_playback()

    def set_focus_active(self, active: bool) -> None:
        """Silence the score and new SFX while the app/tab cannot be heard safely."""

        self._focus_active = bool(active)
        self._sync_music_playback()

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
        self._ensure_music_started()
        self._apply_music_mix()
        if self.ready and self.enabled and self._focus_active and speed > 22.0:
            now = time.monotonic()
            cadence = 0.18 if speed > 170.0 else 0.31
            if now - self._last_step >= cadence:
                self._play("step")
                self._last_step = now

    def _ensure_music_started(self) -> None:
        if not self._music_allowed() or self._music_started:
            return
        self.ambient_channel.play(self.ambient, loops=-1, fade_ms=900)
        self.tension_channel.play(self.tension, loops=-1, fade_ms=1100)
        self._music_started = True

    def _music_allowed(self) -> bool:
        return bool(self.ready and self.enabled and self._gameplay_active and self._focus_active)

    def _sync_music_playback(self) -> None:
        if not self.ready:
            return
        for channel in self._owned_music_channels():
            channel.unpause() if self._music_allowed() else channel.pause()

    def _apply_music_mix(self) -> None:
        if not (self.ready and self.ambient_channel is not None and self.tension_channel is not None):
            return
        music = self.master_volume * self.music_volume
        tension = max(self._trace_mix * 0.8, self._lockdown_mix)
        self.ambient_channel.set_volume(music * (0.16 - 0.055 * tension))
        self.tension_channel.set_volume(music * (0.01 + 0.13 * tension))

    def handle(self, events: Iterable[SimEvent]) -> None:
        if not (self.ready and self.enabled and self._focus_active):
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
                    self._play(key)
                    self._last_played[key] = now

    def menu_move(self) -> None:
        if self.ready and self.enabled and self._focus_active:
            self._play("menu")

    def menu_confirm(self) -> None:
        if self.ready and self.enabled and self._focus_active:
            self._play("confirm")

    def _play(self, key: str) -> None:
        if not (self.ready and self.enabled and self._focus_active):
            return
        channel = self.sounds[key].play()
        if channel is not None:
            self._sfx_channels = [owned for owned in self._sfx_channels if owned.get_busy()]
            self._sfx_channels.append(channel)

    def close(self) -> None:
        self._stop_owned_audio()
        if type(self)._active_director is self:
            type(self)._active_director = None
        self.ready = False

    def _retire(self) -> None:
        """Silence an old instance without touching a replacement's audio."""

        self._stop_owned_audio()
        self.ready = False

    def _owned_channels(self) -> list[object]:
        return [*self._owned_music_channels(), *self._owned_sfx_channels()]

    def _owned_music_channels(self) -> list[object]:
        channels: list[object] = []
        for channel, sound in (
            (self.ambient_channel, self.ambient),
            (self.tension_channel, self.tension),
        ):
            if channel is not None and sound is not None and channel.get_sound() is sound:
                channels.append(channel)
        return channels

    def _owned_sfx_channels(self) -> list[object]:
        channels: list[object] = []
        sounds = tuple(self.sounds.values())
        for channel in self._sfx_channels:
            playing = channel.get_sound()
            if playing is not None and any(playing is sound for sound in sounds):
                channels.append(channel)
        return channels

    def _stop_owned_audio(self) -> None:
        for channel in self._owned_channels():
            channel.stop()
        self._sfx_channels.clear()
        self._music_started = False

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
        samples = int(self._sample_rate * seconds)
        time = np.arange(samples, dtype=np.float32) / self._sample_rate
        # The old 55/82.5 Hz continuous pad read as mains hum on laptop and
        # phone speakers.  This restrained, slowly breathing upper register
        # leaves ample baked-in headroom even if a WebAudio backend ignores a
        # transient channel-volume update during context resume.
        breath = 0.42 + 0.58 * (0.5 + 0.5 * np.sin(math.tau * 0.125 * time - math.pi / 2))
        pad = breath * (
            0.045 * np.sin(math.tau * 110 * time)
            + 0.024 * np.sin(math.tau * 165 * time)
        )
        shimmer = (
            0.5 + 0.5 * np.sin(math.tau * 0.25 * time - math.pi / 2)
        ) * 0.012 * np.sin(math.tau * 330 * time)
        return np.asarray(pad + shimmer, dtype=np.float32)

    def _tension_wave(self, seconds: float) -> np.ndarray:
        samples = int(self._sample_rate * seconds)
        time = np.arange(samples, dtype=np.float32) / self._sample_rate
        throb_envelope = np.power(
            0.5 + 0.5 * np.sin(math.tau * 0.5 * time - math.pi / 2),
            2,
        )
        throb = 0.065 * throb_envelope * np.sin(math.tau * 82.5 * time)
        alarm = 0.022 * np.sin(
            math.tau * 123.75 * time + 0.35 * np.sin(math.tau * 0.125 * time)
        )
        texture = (
            np.sin(math.tau * 313 * time) + np.sin(math.tau * 487 * time)
        ) * (0.5 + 0.5 * np.sin(math.tau * 0.25 * time - math.pi / 2))
        return np.asarray(throb + alarm + 0.009 * texture, dtype=np.float32)
