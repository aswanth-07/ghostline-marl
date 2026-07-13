# Web and Vercel deployment

Ghostline's browser release is a static Pygbag 0.9.3 build. The deterministic Python simulation runs in WebAssembly, while the selected recurrent policy is lazy-loaded through ONNX Runtime Web 1.27.0. Human play remains available if the manifest, runtime, model, WebGPU initialization, or WASM initialization fails.

## Architecture

- `web/main.py` starts the same `GameApp` through its cooperative async loop.
- `web/runtime.py` is the only Python adapter. It exposes tier/seed launch, current-run agent takeover, return-to-human control, and player-equivalent observation serialization. It prefetches inference in the spare frame before each 10 Hz decision so the async JavaScript bridge does not add a full policy-step delay.
- `web/static/policy-bridge.mjs` owns asynchronous inference, legal-action enforcement, persistent GRU state, latency telemetry, and WebGPU-to-WASM fallback.
- `web/static/matched-runs.mjs` admits comparison cards only when both completed runs have the exact same tier and seed. Mismatched contracts are displayed as `NOT COMPARED` with an explicit refusal reason.
- `web/static/embed-bridge.mjs` owns the versioned, origin-scoped portfolio message contract. It never accepts gameplay commands from the parent page.
- `web/ghostline.tmpl` and `web/static/ghostline.css` provide the responsive loading, focus, fullscreen, Agent Lab, and human-versus-agent shell.
- `scripts/build_web.py` SHA-256-locks and self-hosts Pygbag's 0.9.3 CPython 3.12 runtime, locks and verifies the ONNX Runtime and BrowserFS npm tarballs, validates the selected model's ONNX input/output shapes and dtypes plus its v2 environment-source fingerprint, derives the GRU width instead of assuming it, generates content-addressed model filenames, invokes Pygbag, and writes `bundle-report.json`.
- `.vercelignore` excludes local virtual environments, training artifacts, evidence ledgers, QA output, caches, and desktop packages from the remote build upload. Vercel receives only the locked build inputs, selected deployment model, runtime source/assets, license documents, and web shell.
- `vercel.json` explicitly selects the `Other` framework preset (`framework: null`) and disables Vercel's inferred install phase. This prevents the repository's packaging `pyproject.toml` from being mistaken for a Python Function; the locked custom build command remains the only install/build authority.
- The custom command creates `.vercel-venv` and installs through that isolated interpreter. Vercel's uv-managed Python image enforces PEP 668, so the release never mutates the system environment or uses `--break-system-packages`.
- The Pygbag archive is assembled from an explicit twelve-module game-runtime allowlist and the exact three runtime atlases declared by `assets/licenses.json`. Training, evaluation, export, packaging, recording, screenshots, source drafts, retired key art, and unused web derivatives are never copied into the browser stage.

The model is never fetched on ordinary human play. ONNX Runtime and the content-addressed model are requested only after `AGENT TAKEOVER` or `?autoplay=1`.
Campaign progression and settings use the desktop JSON contract inside the Python runtime and are mirrored to browser `localStorage`, so refreshes retain unlocked tiers without introducing a second save schema. Storage denial in a restricted iframe falls back to a fresh in-memory profile.

## Portfolio embed contract

Use `?embed=1&autoplay=0` for the portfolio presentation. Embed mode removes only
the redundant standalone brand header and legal footer; it retains the explicit
audio/focus gate, keyboard help, human/agent controls, contract launcher, live
telemetry, and matched-run cards. At narrow widths the lab moves below the game
instead of being removed. `autoplay=0` never bypasses Chrome's user-activation
gate and never loads the policy without an explicit takeover.

When embedded in a frame, Ghostline sends these display-only messages to the
parent after resolving the parent origin from `document.referrer` and Chrome's
`ancestorOrigins`. It does not use a wildcard target origin and suppresses the
message if the two origin signals disagree:

```json
{"source":"ghostline","version":1,"type":"ready","modelAvailable":true}
{"source":"ghostline","version":1,"type":"run-complete","controller":"agent","tier":6,"seed":2000071,"success":true,"duration":41.25}
```

`modelAvailable: false` identifies a valid human-only fallback, not a failed
game load. `controller` is `human`, `agent`, or `hybrid`; mixed-control results
remain excluded from the in-game matched benchmark. The portfolio must validate
`event.origin`, `source`, `version`, and `type`, and must treat these events as
telemetry only. Ghostline intentionally has no parent-to-game command channel.

The one-shot `ready` event describes the secure web shell, not completion of the
Python game loop. It is emitted as soon as the policy manifest is known, before
the self-hosted runtime reaches Chrome's required user-activation gate. The
parent can therefore reveal Ghostline's own accurate download and audio-focus
progress instead of obscuring it with a competing timeout. `gameReady` remains
false until the deterministic game loop has actually started.

Losing tab or iframe focus pauses an active human mission and never steals focus
back automatically; the player explicitly clicks the game before resuming. A
mission that switches controllers is labeled `hybrid`, including takeover time
and data, and is excluded from the pure human-versus-agent result cards.
After a run ends, its resolved tier and procedural seed are pinned into the
launcher so the other controller replays the identical contract by default.
If inference fails after takeover, the bridge immediately invalidates recurrent
memory and its prior action, emits only neutral action zero, and asks the Python
adapter to close the policy environment and restore human control. The active
run becomes `hybrid`; a failed backend can never keep replaying stale movement.

## Build commands

Install the browser build extra once:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[web]"
```

Build an explicitly non-release, human-only diagnostic:

```powershell
.\.venv\Scripts\python.exe scripts\build_web.py --human-only
```

Build the final release (the selected model is mandatory by default):

```powershell
.\.venv\Scripts\python.exe scripts\build_web.py --model models\ghostline-policy.onnx
```

Recheck the generated payload without rebuilding:

```powershell
.\.venv\Scripts\python.exe scripts\build_web.py --check-only
# Only for an existing diagnostic bundle:
.\.venv\Scripts\python.exe scripts\build_web.py --check-only --human-only
node --test web\tests\*.test.mjs
```

With a bundled model, execute two real ONNX Runtime Web/WASM recurrent transitions from Node before opening Chrome:

```powershell
node web\tests\onnx-wasm-smoke.mjs .web-build\ghostline\build\web
```

`bundle-report.json` distinguishes the complete local human bundle from the lazy WASM and WebGPU agent payloads. The human figure includes `pythons.js`, `cpythonrc.py`, `empty.ogg`, and the CPython 3.12 `main.js`, `main.wasm`, and `main.data`; there is no uncounted external core bootstrap. The build verifies every published runtime hash and fails over 25 MiB for the local human bundle or 50 MiB for the aggregate local human-plus-WASM-agent path.

The local byte figure is deliberately not described as the complete cold
browser transfer. Pygbag's PEP-723 installer still obtains its small cp312
package index plus browser wheels for NumPy, pygame-ce, Gymnasium, and
Gymnasium's pure-Python dependencies from the Pygbag/PyPI package repositories.
Those package requests are distinct from the now-self-hosted core runtime and
must be measured in Chrome. As of the 2026-07 release audit, the wheel bodies
total about 15.0 MB; their repository selection is not part of Ghostline's
static checksum lock. The 24.1 MB local bundle plus those wheel bodies is about
39.1 MB of raw artifacts, so the original under-25-MB total cold-transfer target
is not established by this build and remains an explicit release limitation.
Only production Chrome transfer traces can account for Vercel compression,
browser caching, and the package installer's actual request set.

The production post-build removes Pygbag's unused `ghostline.apk`; Vercel serves
only the browser `ghostline.tar.gz`. The launch gate uses the same flat facility
grid language as the 2D game and does not ship or display the retired
three-quarter-view key art.

The Pygbag runtime is downloaded only during the controlled build, verified
against six reviewed upstream SHA-256 values, and published beneath
`runtime/pygbag-0.9.3/`. Its `pythons.js` output has a second fixed hash after a
single narrow patch removes the erroneous console log from Pygbag's caught
cross-origin `window.top.blanker` probe. The optional top-window blanker is not
part of Ghostline and the patch changes no Python, WebAssembly, input, audio, or
rendering behavior. The shell requests only `snd,gui`; the unused `vtx` feature
is excluded so it cannot import the external terminal bootstrap.

Every web bundle includes `THIRD_PARTY_NOTICES.md`, BrowserFS's MIT license,
and—when the agent runtime is present—the checksum-locked ONNX Runtime license
and full upstream third-party notices. Missing legal documents fail bundle
validation just like a missing WASM binary.

The PEP 723 block in `web/main.py` intentionally uses bare browser repository
names (`numpy`, then `gymnasium`). Pygbag 0.9.3's browser installer resolves
those literal names and does not parse desktop-style `==` constraints. Desktop,
training, packaging, and ONNX dependencies remain exactly locked by
`pyproject.toml` and `requirements.lock`; the browser uses Pygbag's matching
CPython 3.12 WASM wheels (currently NumPy 2.0.2 and pygame-ce 2.5.7) plus the
pure-Python Gymnasium wheel.

## Chrome-only QA

Do not use the Codex in-app Browser for this project. Start the local build server in one terminal:

```powershell
.\.venv\Scripts\python.exe scripts\build_web.py --serve --model models\ghostline-policy.onnx
```

Then open the printed URL in Google Chrome, normally:

```powershell
Start-Process "$env:ProgramFiles\Google\Chrome\Application\chrome.exe" "http://localhost:8000/?autoplay=0"
```

Verify in Chrome DevTools:

1. Loading progress is readable, the audio-authorization gate works, and `FOCUS GAME` restores keyboard input.
2. WASD, Shift, Space, Escape, restart, menus, fullscreen, tier selection, and deterministic seed selection work.
3. `AGENT TAKEOVER` continues the current mission, updates backend/latency telemetry, and `TAKE CONTROL` returns the same mission to the player.
4. Complete human and agent runs on an identical tier/seed and confirm the matched cards appear; then use a different seed and confirm comparison is explicitly refused.
5. Disable WebGPU and confirm WASM fallback; block the model request and confirm human-only fallback. Also interrupt a live inference request and confirm action zero followed by manual-control restoration and a `hybrid` run label.
6. Use the Network panel with cache disabled to record usable-start time and transfer size. Use Performance for 60 FPS and ten policy calls per second.
7. Test both the standalone URL and the portfolio iframe at desktop widths. Keyboard input must remain opt-in through the focus button.
8. In the iframe, confirm one `ready` message reports the bundled-model state and one `run-complete` message is emitted for each terminal contract state. Confirm a different parent/referrer origin receives no message.
9. Confirm all six core files, including `runtime/pygbag-0.9.3/cpython312/main.wasm`, are served from the Ghostline origin and the console contains no `window.top.blanker` error. External package-index/wheel requests are allowed and recorded separately; external `0.9.3/pythons.js`, `0.9.3/cpython312/*`, `vtx.js`, `vt/*`, or `xtermjsixel/*` requests are release failures.

The Pygbag test server does not reproduce Vercel's isolation headers. Repeat policy-threading and embed checks on the Vercel preview. Vercel retains
`Cross-Origin-Embedder-Policy: credentialless` for the standalone threaded-WASM
path, but all CPython bootstrap resources now come from Ghostline's own origin.
The versioned `runtime/` tree receives immutable caching; HTML, the policy
manifest, and the game archive continue to revalidate.

## Vercel release

`vercel.json` installs the locked `[web]` build extra and invokes the strict
model build. A deployment therefore fails closed when
`models/ghostline-policy.onnx` is absent or incompatible; Vercel can never
silently publish the diagnostic human-only bundle. The output includes the MIT
license, publishes `.web-build/ghostline/build/web`, supplies COOP/COEP for
threaded WASM, disables caching for the HTML/manifest, and applies immutable
caching only to versioned runtime and content-addressed model assets. Regular
CI labels its `--human-only` artifact as diagnostic, while the tag/manual
release workflow requires the champion and runs recurrent WASM inference.

```powershell
npx vercel link
npx vercel deploy
npx vercel --prod
```

Recommended portfolio embed:

```html
<iframe
  src="https://YOUR-GHOSTLINE-DEPLOYMENT.vercel.app/?embed=1&amp;autoplay=0"
  title="Play Ghostline or watch its recurrent RL agent"
  allow="autoplay; fullscreen; gamepad; cross-origin-isolated"
  loading="lazy"
  style="width:100%;aspect-ratio:16/10;border:0"
></iframe>
```

Final publication requires a policy exported from the frozen `GhostlineEnv-v2` contract. Never substitute an older smoke checkpoint merely to make the Agent button active.
The ONNX graph must carry `ghostline.contract=GhostlineEnv-v2` and the current
`ghostline.environment_fingerprint`; the builder rejects stale or unlabelled
graphs even when their tensor dimensions happen to match.
