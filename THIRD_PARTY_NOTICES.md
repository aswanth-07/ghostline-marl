# Ghostline third-party notices

Ghostline's own source code and project-owned visual/audio assets are released
under the MIT License in `LICENSE`. That license does not replace the terms of
the third-party runtime components used by the game.

The Windows release build inventories the Python distributions actually found
inside the PyInstaller archive. It writes their versions and declared licenses
to `Ghostline.manifest.json`, and copies the corresponding license/notice files
to `licenses/` beside `Ghostline.exe`. This includes the Python runtime and the
PyInstaller bootloader notice as well as discovered packages such as NumPy,
Gymnasium, pygame-ce, and (for agent builds) ONNX Runtime and its runtime
dependencies.

The exact dependency versions used for reproducible builds are locked in
`requirements.lock`. Upstream components remain the property of their
respective copyright holders and are distributed under their own terms.

The static web release additionally self-hosts the checksum-locked Pygbag 0.9.3
browser bootstrap and CPython 3.12 WebAssembly runtime, plus BrowserFS 1.4.3 and
ONNX Runtime Web 1.27.0. Its root `THIRD_PARTY_NOTICES.md` is accompanied by:

- Pygbag's MIT text under `licenses/pygbag-0.9.3/`;
- the Python Software Foundation license under `licenses/cpython-3.12/`;
- BrowserFS's MIT text under `licenses/browserfs-1.4.3/`; and
- for agent builds, ONNX Runtime's MIT text and upstream notices under
  `licenses/onnxruntime-web-1.27.0/`.

The web builder validates each self-hosted runtime file and license before it
can enter the release bundle. Ghostline applies one documented source-level
patch to the vendored Pygbag bootstrap: a caught, optional `window.top.blanker`
probe no longer logs an error when the game is embedded by a cross-origin
portfolio. No runtime or Python behavior is changed by that patch.
