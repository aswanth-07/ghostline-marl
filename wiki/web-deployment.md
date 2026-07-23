# Web and Vercel deployment

Ghostline's browser release is a static Pygbag 0.9.3 build. The deterministic Python simulation runs in WebAssembly, while the selected recurrent policy is lazy-loaded through ONNX Runtime Web 1.27.0. Human play remains available if the manifest, runtime, model, WebGPU initialization, or WASM initialization fails.

## Architecture

- `web/main.py` starts the same `GameApp` through its cooperative async loop.
- `web/runtime.py` is the only Python adapter. It exposes tier/seed launch, current-run agent takeover, exact portfolio-run replay, return-to-human control, touch-device detection, and player-equivalent observation serialization. It queues inference from the exact state at each 10 Hz boundary; if a browser result misses the next render frame, simulation time waits instead of advancing with a fabricated neutral action.
- `web/static/policy-bridge.mjs` owns asynchronous inference, legal-action enforcement, persistent GRU state, latency telemetry, and backend selection. Threaded WASM is the measured release default for this compact recurrent graph; `?backend=webgpu` retains the WebGPU comparison path with automatic WASM fallback.
- `web/static/matched-runs.mjs` admits comparison cards only when both completed runs have the exact same tier and seed. Mismatched contracts are displayed as `NOT COMPARED` with an explicit refusal reason.
- `web/static/embed-bridge.mjs` owns the versioned, origin-scoped portfolio message contract. It never accepts gameplay commands from the parent page.
- `web/ghostline.tmpl` and `web/static/ghostline.css` provide the responsive loading, focus, fullscreen, Agent Lab, and human-versus-agent shell.
- `scripts/build_web.py` SHA-256-locks and self-hosts Pygbag's 0.9.3 CPython 3.12 runtime, locks and verifies the ONNX Runtime and BrowserFS npm tarballs, validates the selected model's ONNX input/output shapes and dtypes plus its v2 environment-source fingerprint, derives the GRU width instead of assuming it, generates content-addressed model filenames, invokes Pygbag, and writes `bundle-report.json`.
- `.vercelignore` excludes local virtual environments, training artifacts, evidence ledgers, QA output, caches, and desktop packages from the remote build upload. Vercel receives only the locked build inputs, selected deployment model, runtime source/assets, license documents, and web shell.
- `vercel.json` explicitly selects the `Other` framework preset (`framework: null`) and disables Vercel's inferred install phase. This prevents the repository's packaging `pyproject.toml` from being mistaken for a Python Function; the locked custom build command remains the only install/build authority.
- The custom command creates `.vercel-venv` and installs through that isolated interpreter. Vercel's uv-managed Python image enforces PEP 668, so the release never mutates the system environment or uses `--break-system-packages`.
- The Pygbag archive is assembled from an explicit twelve-module game-runtime allowlist and the exact three runtime atlases declared by `assets/licenses.json`. Training, evaluation, export, packaging, recording, screenshots, source drafts, retired key art, and unused web derivatives are never copied into the browser stage.

The model is never fetched on ordinary human play. ONNX Runtime and the content-addressed model are requested only after `AGENT TAKEOVER`, `REPLAY PORTFOLIO AGENT RUN`, or `?autoplay=1`. The replay action always starts a fresh tier-6 seed-2,000,000 run, while ordinary takeover preserves an already active human contract.
Campaign progression and settings use the desktop JSON contract inside the Python runtime and are mirrored to browser `localStorage`, so refreshes retain unlocked tiers without introducing a second save schema. Storage denial in a restricted iframe falls back to a fresh in-memory profile.

Coarse-pointer phones enable the in-canvas movement stick, dash, pulse, and pause contacts before the first mission frame; a hybrid touch laptop that still reports a fine primary pointer remains in the desktop layout. The canvas uses `touch-action: none` so browser panning cannot steal a held direction; HTML contract controls remain native pointer/touch targets. Mouse and touch selection inside the Pygame menus is mapped through the same letterbox transform used for rendering. Portrait phones receive an explicit rotate-to-landscape readability gate rather than a silently compressed playfield. Landscape uses dynamic viewport height and safe-area insets, offers fullscreen from the launch gesture, and contains the fixed 16:9 framebuffer instead of stretching it to the phone's physical aspect ratio. Because phones downsample the 1280x720 backing canvas into a smaller CSS viewport, they select browser-quality interpolation instead of nearest-neighbour CSS downscaling; desktop retains the crisp pixel presentation. Contract and telemetry controls remain in a dismissible `RUN SETUP` drawer that behaves as a modal while open, makes the background inert, and restores focus to the initiating control or game canvas when it closes.

## Portfolio embed contract

Use `?embed=1&autoplay=0` for the portfolio presentation. Embed mode removes only
the redundant standalone brand header and legal footer; it retains the explicit
audio/focus gate, human/agent controls, contract launcher, live telemetry, and
matched-run cards. At narrow widths the playfield comes first and the lab moves
into the accessible `RUN SETUP` drawer; no policy or telemetry functionality is
removed. `autoplay=0` never bypasses Chrome's user-activation
gate and never loads the policy without an explicit takeover.

Because the portfolio and game are separate Vercel origins, the iframe includes
`webgpu` in its Permissions Policy `allow` attribute for the optional
`?backend=webgpu` comparison path. The release defaults to threaded ONNX Runtime
Web WASM because the selected 5.83 MB GRU measured substantially faster there
on the target Chrome machine. An unavailable or failed WebGPU request falls
back to the same WASM path.

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
false until the Python adapter has initialized the real canvas and deterministic
game loop. The Pygbag template must never declare `ready` on its own; doing so
would expose its still-1×1 bootstrap canvas as a blank playable surface.

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
JavaScript owns the asynchronous ONNX session load and enqueues a plain
`agent-ready` command only after initialization succeeds. Python performs the
synchronous environment handoff from that command; it never awaits a JavaScript
Promise across the Pygbag boundary.

Takeover uses an explicit three-state handshake: `HUMAN CONTROL`, `AGENT
HANDOFF`, and `AGENT CONTROL`. The shell mirrors model-download progress,
provides a cancellable handoff, and does not claim agent control until the
browser returns the first recurrent inference. Each asynchronous inference is
identified by the bridge's monotonic completion count. Python retains an
outstanding generation until that exact result is ready, rather than clearing
it to `HOLD`. The next observation is captured only after all six ticks of the
previous action complete, so live WASM/WebGPU decisions use the same state
sequence as Python evaluation and recording. A late backend pauses simulation
time at the decision boundary without replaying stale actions or issuing a
duplicate request. The takeover click also satisfies the launch/focus gesture: if
the Python canvas is still starting, the shell enters it automatically when it
becomes ready instead of leaving the policy active behind a second launch gate.

Pygbag writes a device-pixel-ratio-adjusted inline canvas size during startup.
The Ghostline shell deliberately overrides that hint and scales the fixed 16:9
framebuffer across the complete game frame. The WebAssembly presentation path
also scales the logical 640x360 scene to the exact browser canvas dimensions,
so browser zoom and intermediate viewport widths cannot fall back to a small
centered image. Desktop builds retain crisp integer-only scaling.

The standalone shell also exposes an `INTEL PANEL` toggle. Wide-screen users
can collapse the full launcher/telemetry column and give the 16:9 playfield the
workspace; live telemetry and matched-run analysis remain opt-in disclosures.
Entering a contract now collapses that rail automatically, expanding the game
from the setup layout while leaving `SHOW INTEL` available for tier, seed, and
matched-run analysis. The launcher uses the portfolio's neutral-black surfaces,
58 px grid, cyan/magenta interaction tokens, and self-hosted Manrope/JetBrains
Mono Latin variable fonts; their SIL OFL 1.1 texts ship with the static bundle.
The toggle carries `aria-controls`/`aria-expanded`, restores focus on close,
and leaves the deterministic simulation and model interface untouched. Rapid
5 Hz raw telemetry is deliberately not an `aria-live` region; controller and
run-state changes use the lower-frequency status notice so assistive technology
is not flooded with continuously changing numbers.

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
package index plus browser wheels for NumPy and pygame-ce from the Pygbag/PyPI
package repositories.
Those package requests are distinct from the now-self-hosted core runtime and
must be measured in Chrome. As of the 2026-07 release audit, the NumPy and
pygame-ce wheel bodies total about 14.0 MB; their repository selection is not
part of Ghostline's static checksum lock. The 24.1 MB local bundle plus those
wheel bodies is about 38.1 MB of raw artifacts, so the original under-25-MB
total cold-transfer target is not established by this build and remains an
explicit release limitation.
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

The PEP 723 block in `web/main.py` intentionally uses the bare browser
repository name `numpy`. Pygbag 0.9.3's installer resolves that literal name and
does not parse desktop-style `==` constraints. Gymnasium is deliberately absent
from the browser package set: its unused vector imports pull in
`multiprocessing.sharedctypes`, which CPython/WASM does not provide. The web
policy adapter installs a narrow in-memory compatibility module containing only
the `Env.reset` seed contract and `Discrete`/`Box`/`Dict` space records used by
`GhostlineEnv`; desktop, tests, and training continue to use the real locked
Gymnasium package. This removes the failing import and keeps human startup free
of an unnecessary wheel without changing observations or simulation behavior.

## Chrome-only QA

Use Google Chrome for interactive release QA. Start the local build server in one terminal:

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
3. `AGENT TAKEOVER` continues an active human mission on its exact seed, updates backend/latency telemetry, and `TAKE CONTROL` returns the same mission to the player. From the menu or `autoplay=1`, a blank seed is pinned to the selected tier's disclosed passing validation showcase so the first Watch Agent view is representative rather than a random failure-tail sample.
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
