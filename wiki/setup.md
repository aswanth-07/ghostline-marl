---
title: Ghostline Setup and Release
updated: 2026-07-28
status: active
---

# Setup and release

Python 3.13 is the locked release baseline. CI also checks the base runtime on
Python 3.12 and 3.14. Use the repository `.venv` and `requirements.lock` for
every command. Other local virtual environments are unsupported.

## Install and play

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --constraint requirements.lock -e .
ghostline play
```

The normal campaign and published Agent Lab policy are public
`GhostlineEnv-v1`. The current multi-agent v2 game is opt-in:

```powershell
ghostline play --adaptive --tier 6 --directive ghost
```

The `--adaptive` CLI spelling is retained as a user-facing mode switch; it does
not create a third environment version. The selected simulation is
`GhostlineEnv-v2`.

The base install has no PyTorch, ONNX Runtime, recording codec, or packager
dependency. Both environments can run headlessly with deterministic scripted
controllers.

## Agent Lab and development

```powershell
# Lightweight published-v1 ONNX inference; no PyTorch.
python -m pip install --constraint requirements.lock -e ".[agent]"
ghostline lab --tier 6 --seed 2000000

# Tests, lock maintenance, and distributions.
python -m pip install --constraint requirements.lock -e ".[dev]"
python -m pytest -q
```

The published Agent Lab policy is v1-only. The desktop and web launchers must
refuse to pass its 36-action observation contract to a live v2 simulation.
Missing or incompatible policies fall back to human/scripted control.

## Contract smoke

```powershell
python -c "import gymnasium as gym, ghostline; a=gym.make('GhostlineEnv-v1'); b=gym.make('GhostlineEnv-v2'); print(a.action_space, b.action_space)"
```

Expected action spaces are `Discrete(36)` and `Discrete(288)`. There is no
third registered Ghostline environment.

## Correctness and generation

```powershell
# Published-v1 level contract.
python scripts/fuzz_ghostline_levels.py --seeds 10000

# Developmental v2 geometry, vents, panels, security doors, and readiness.
python scripts/fuzz_ghostline_levels.py --adaptive --seeds 10000

# Immutable public-v1 release throughput gate.
python scripts/benchmark_ghostline.py --decisions 10000 --tier 6 --workers 22 --minimum-decisions-per-second 3000 --output benchmarks/system/headless-throughput.json

# Developmental-v2 calibration; record the measured value without inventing a pass threshold.
python scripts/benchmark_ghostline.py --adaptive --decisions 1000 --tier 6 --workers 22 --minimum-decisions-per-second 0 --output artifacts/v2-readiness/headless-throughput.json
```

V2 fuzzing uses the v2 generator's own validator and readiness diagnostics,
not only the inherited base validator.

## Published v1 reproduction

```powershell
python -m pip install --constraint requirements.lock -e ".[train]"

ghostline imitate collect --output artifacts/teacher-data --episodes-per-tier 100 --overwrite
ghostline imitate bc --dataset artifacts/teacher-data --output artifacts/bc-current
ghostline imitate dagger --base-dataset artifacts/teacher-data --initial-checkpoint artifacts/bc-current/best.pt --output artifacts/dagger --beta-start 0
ghostline train --hours 24 --experiment ghostline-universal --init-checkpoint PATH_FROM_DAGGER_OUTPUT --initial-curriculum-tier 6
```

These commands reproduce the published runner lineage. Its checkpoint/evidence
metadata retains the historical internal label `GhostlineEnv-v2`; the public
environment id is v1.

## Developmental v2 runner training

Install the train extra, run a short CPU smoke with the standalone v2 runner
entry point, then launch the long CUDA campaign only after the preflight
manifest is frozen:

```powershell
python -m pip install --constraint requirements.lock -e ".[train]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

ghostline train-runner-v2 --output artifacts/runner-v2/preflight --published-v1-init models/ghostline-policy.pt --dry-run --cpu
ghostline train-runner-v2 --output artifacts/runner-v2/smoke --allow-scratch --envs 1 --rollout 2 --epochs 1 --minibatch-envs 1 --sync-envs --validation-interval 0 --max-updates 1 --cpu
ghostline train-runner-v2 --output artifacts/runner-v2/smoke --envs 1 --rollout 2 --epochs 1 --minibatch-envs 1 --sync-envs --validation-interval 0 --max-updates 2 --cpu --resume

ghostline train-runner-v2 --output artifacts/runner-v2/ppo --published-v1-init models/ghostline-policy.pt --envs 8 --rollout 128 --epochs 4 --minibatch-envs 2 --seconds 86400
```

The first command validates the complete manifest and initialization without
starting workers. The second exercises one complete recurrent PPO update and
checkpoint; the third proves that environment schedules, live episode state,
observations, GRU state, optimizer state, and every RNG stream restore exactly.
Do not start a long campaign until the 10,000-seed v2 audit and the security
optimizer smoke both pass.

`--published-v1-init` is an explicit weight transplant, not checkpoint
compatibility. It verifies the immutable v1 checkpoint, copies shape-compatible
perception/recurrent/value weights, expands each 36-action logit across its v2
semantic variants, leaves new field channels neutral, and records the source
hash. The orthogonal-scratch ablation must say `--allow-scratch`;
`--init-checkpoint` accepts only a current-fingerprint v2 policy.

## Developmental v2 security training

```powershell
python -m pip install --constraint requirements.lock -e ".[marl]"

# Pipeline smoke.
ghostline train-security --max-steps 20 --envs 1 --rollout 3 --epochs 1 --device cpu --tiers 3

# Long campaign after the preflight gate.
ghostline train-security --dry-run --hours 72 --envs 8 --rollout 64 --tiers 3,4,5,6 --runner-model models/ghostline-policy.pt
ghostline train-security --hours 72 --envs 8 --rollout 64 --tiers 3,4,5,6 --runner-model models/ghostline-policy.pt --runner-pool artifacts/runner-v2/ppo/best.pt
ghostline evaluate-security --model artifacts/security-mappo/champion.pt --episodes-per-tier 500 --seed-start 14000000 --slice-manifest benchmarks/security/v2-final-test-slices.json
```

The published v1 runner may be a frozen opponent only through its explicit
adapter. `--scripted-runner` is the easier baseline. Every security checkpoint
binds the observation, source, generation, reward, critic, runner-opponent, and
configuration fingerprints. A no-validation smoke ends at `last-policy.pt`;
only held-out selection writes `champion.pt`.

The pre-migration `models/ghostline-security.pt` is stale for v2. The launcher
must reject it and use the tactical fallback until a new compatible checkpoint
passes held-out evaluation.

## Recording, ONNX export, and Windows package

The current public package continues to ship the published v1 champion:

```powershell
python -m pip install --constraint requirements.lock -e ".[train,media,build]"
ghostline record --model models/ghostline-policy.pt --tier 6 --seed 2000000 --output videos/ghostline-demo.mp4
ghostline export --model models/ghostline-policy.pt --output models/ghostline-policy.fp32.onnx --quantize --deployment-output models/ghostline-policy.onnx --parity-samples 1000
Copy-Item models/ghostline-policy.fp32.parity.json benchmarks/neural/champion-onnx-parity.json
python scripts/verify_release_evidence.py
ghostline package --model models/ghostline-policy.onnx
.\dist\Ghostline.exe --release-smoke-test
```

The ONNX metadata must still say historical `GhostlineEnv-v2` because release
verification binds those exact bytes. Packaging maps that artifact to public
v1 in UI and documentation. It must not relabel graph metadata.

The FP32 export is canonical. Dynamic INT8 becomes the deployment graph only
after at least 1,000 recurrent transitions produce zero deterministic-action
mismatches; otherwise verified FP32 is deployed.

The player executable contains the game, declared runtime art, ONNX Runtime,
the verified policy, licenses, and notices. It excludes Torch, trainers,
TensorBoard, and recording codecs. `--human-only` is diagnostic and not a
portfolio release.

## Wheel and clean-install gate

```powershell
python -m build
python scripts/verify_source_archive.py
python scripts/verify_clean_install.py
```

The clean-install probe installs the base wheel in an isolated environment,
confirms deferred Pygame/Torch imports, steps public v1 and developmental v2,
checks `36` and `288` actions, verifies the v1 historical-contract annotation,
renders a headless frame, and exercises the v2 tactical fallback. Player wheels
deliberately omit the retired `models/ghostline-security.pt`; the source archive
retains it only as immutable historical evidence.

## Static web build

```powershell
# Human-only diagnostic.
python scripts/build_web.py --human-only

# Published v1 portfolio build.
python scripts/build_web.py --model models/ghostline-policy.onnx
```

The runtime stage includes the exact v1 wrapper plus v2 config, types,
generation, simulation, and environment modules. The browser shell labels the
contracts v1/v2 and blocks policy takeover on v2. Interactive QA uses Chrome
only.

## Dependency lock

After deliberately changing direct pins in `pyproject.toml`:

```powershell
python -m piptools compile --extra=agent --extra=build --extra=dev --extra=media --extra=train --extra=web --output-file=requirements.lock --strip-extras pyproject.toml
python -m pip check
```

Never change dependencies during a long run. Stop, update the lock, run every
smoke gate, and resume only from a checkpoint whose full contract still
matches.
