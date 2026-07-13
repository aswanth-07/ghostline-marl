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

The static web release additionally ships BrowserFS 1.4.3 and ONNX Runtime Web
1.27.0. Its root `THIRD_PARTY_NOTICES.md` is accompanied by the exact BrowserFS
MIT text under `licenses/browserfs-1.4.3/` and the ONNX Runtime MIT text plus
upstream `ThirdPartyNotices.txt` under `licenses/onnxruntime-web-1.27.0/`.
Those documents are version-locked and checksum-verified by the web builder.
