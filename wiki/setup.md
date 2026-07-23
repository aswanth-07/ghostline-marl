---
title: Ghostline Setup and Release
updated: 2026-07-23
status: active
---

# Setup and release

Python 3.13 is the locked release baseline. CI also checks the base runtime on
Python 3.12 and 3.14. Commands below use `requirements.lock` as a constraints
file so editable installs resolve to the reviewed direct and transitive pins.

## Human game

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --constraint requirements.lock -e .
ghostline play
```

The base install deliberately has no PyTorch, ONNX Runtime, recording codec, or
packager dependency. It supports human play and `GhostlineEnv-v2` headlessly.
It also supports rule-controlled Adaptive Contracts and `GhostlineEnv-v3`:

```powershell
ghostline play --adaptive --tier 6 --directive ghost
```

## Agent Lab and development

```powershell
# Lightweight ONNX inference; no PyTorch.
python -m pip install --constraint requirements.lock -e ".[agent]"
ghostline lab --tier 6 --seed 2000000

# Complete tests, lock maintenance, and Python distribution builds. This extra
# includes TensorBoard and recording codecs because the suite imports and
# exercises the training and recorder modules in a clean CI environment.
python -m pip install --constraint requirements.lock -e ".[dev]"
python -m pytest -q
```

Agent Lab falls back to human/scripted play when no compatible
`models/ghostline-policy.onnx` is present. A portfolio release does not use that
fallback: its packaging gate requires and smoke-tests the selected policy.

## Correctness and throughput

```powershell
python scripts/fuzz_ghostline_levels.py --seeds 10000
python scripts/benchmark_ghostline.py --decisions 10000 --tier 6 --workers 22 --minimum-decisions-per-second 3000 --output benchmarks/system/headless-throughput.json
```

Set `--workers` to the intended CPU worker count; one worker is useful for
profiling, while the aggregate command measures the training-throughput gate.

## Hybrid training on Windows/CUDA

WSL2 is optional, not required. The supported local path is the same Python
3.13 environment on Windows with the CUDA-enabled PyTorch 2.13 wheel:

```powershell
python -m pip install --constraint requirements.lock -e ".[train]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

ghostline imitate collect --output artifacts/teacher-data --episodes-per-tier 100 --overwrite
ghostline imitate bc --dataset artifacts/teacher-data --output artifacts/bc-current
ghostline imitate dagger --base-dataset artifacts/teacher-data --initial-checkpoint artifacts/bc-current/best.pt --output artifacts/dagger --beta-start 0
ghostline train --hours 24 --experiment ghostline-universal --init-checkpoint PATH_FROM_DAGGER_OUTPUT --initial-curriculum-tier 6 --initial-validation-cursor 3800
```

Checkpoint paths may change as experiments are selected; use `ghostline
imitate --help` and `ghostline train --help` for the current command contract.
Training and evaluation results must remain in this project directory.

## Adaptive-security training

```powershell
python -m pip install --constraint requirements.lock -e ".[marl]"
ghostline train-security --hours 72 --envs 8 --rollout 64 --tiers 3,4,5,6 --runner-model models/ghostline-policy.pt
ghostline evaluate-security --model artifacts/security-mappo/champion.pt --episodes-per-tier 100 --seed-start 12000000 --output benchmarks/security/final-test.json
```

Use `--max-steps 20 --envs 1 --rollout 3 --epochs 1 --device cpu` for a
pipeline smoke. The production campaign should use CUDA, retain the automatic
11M validation reports, and open the 12M final namespace only after selecting
and freezing a champion.

## Recording, ONNX export, and Windows package

```powershell
python -m pip install --constraint requirements.lock -e ".[train,media,build]"
# Run exactly once, only after validation has selected and frozen the champion.
ghostline evaluate --model models/ghostline-policy.pt --episodes 500 --seed-start 8000000 --slice-manifest benchmarks/final-test-slices.json --output benchmarks/neural/champion-final-8m-500.json
ghostline record --model models/ghostline-policy.pt --tier 6 --seed 2000000 --output videos/ghostline-demo.mp4
ghostline export --model models/ghostline-policy.pt --output models/ghostline-policy.fp32.onnx --quantize --deployment-output models/ghostline-policy.onnx --parity-samples 1000
Copy-Item models/ghostline-policy.fp32.parity.json benchmarks/neural/champion-onnx-parity.json
python scripts/verify_release_evidence.py
ghostline package --model models/ghostline-policy.onnx
.\dist\Ghostline.exe --release-smoke-test
```

The tracked slice manifest retains 2M-7M as historical, locked 8M before the
first current-fingerprint episode, and permanently marked it consumed.
Final JSON, aggregate CSV, and episode CSV evidence bind the checkpoint SHA-256,
environment fingerprint, exact seeds, and deterministic action-sequence hashes.
If a future champion misses acceptance, that slice remains retired; improvements must use
validation evidence and a newly declared untouched slice.

If the selected ONNX file already exists and no recording/export is needed,
installing `.[build]` alone is sufficient for `ghostline package`.

The FP32 file is the immutable canonical export. The default INT8 candidate is
`ghostline-policy.fp32.int8.onnx`; it is copied to the deployment path only if
all 1,000 recurrent deterministic actions match PyTorch. If quantization fails
or changes any action, the command copies verified FP32 instead and records the
rejection in `ghostline-policy.fp32.parity.json`.

The PyInstaller entry point is player-only. It embeds assets, ONNX Runtime, and
the selected policy while excluding Torch, TensorBoard, recording, and trainer
modules. Runtime art is selected exclusively from
`assets/licenses.json:runtime_distribution.files`; portfolio screenshots,
source drafts, web derivatives, and retired key art cannot enter the player by
recursive discovery. The asset manifest and `THIRD_PARTY_NOTICES.md` are
embedded with the MIT terms.

Before PyInstaller runs, portfolio packaging applies the same complete tensor,
dtype, output, `GhostlineEnv-v2`, recurrent-width, and frozen-fingerprint gate
as the web builder. It then requires a sibling export parity report whose
audited artifact SHA-256 is the exact ONNX file being packaged, whose source
checkpoint SHA-256 is present, and whose recurrent audit contains at least
1,000 transitions, all six tiers, 128-step sequences, and zero action
mismatches. Merely renaming an arbitrary ONNX file cannot pass this gate.

Packaging now fails closed in four stages: validate release inputs and the
declared runtime-asset set, inspect the completed PyInstaller archive for
training/media packages, launch the packaged executable's headless simulation
and policy smoke test, then emit the release documents. The schema-3
`dist/Ghostline.manifest.json` hashes the executable, bundled policy, asset
manifest, every runtime atlas, and every release document. It also inventories
the package versions/licenses actually discovered in the executable and records
the user-data/recording contract plus the source-checkpoint and recurrent-parity
evidence for the selected policy.

The release folder contains `LICENSE`, `ASSET-LICENSES.json`,
`THIRD_PARTY_NOTICES.md`, `Ghostline.policy-parity.json`, and exact installed dependency license texts under
`dist/licenses/`. Ghostline source and project-owned assets remain MIT; bundled
dependencies retain their respective licenses. The one-file player records run
telemetry to `%LOCALAPPDATA%\Ghostline\runs-v1.jsonl` and progression/settings
to `progression-v1.json`. MP4 capture intentionally remains a source-tooling
feature under the `media` extra and is not bundled into the player executable.

`ghostline package --human-only` is a diagnostic escape hatch and is not an
acceptable portfolio release artifact.

## Wheel and clean-install gate

```powershell
python -m build
python scripts/verify_source_archive.py
python scripts/verify_clean_install.py
```

The clean-install probe creates an isolated environment, installs only the
base wheel, runs `pip check`, imports simulation/generation without Pygame or
Torch side effects, steps `GhostlineEnv-v2`, and verifies that Ghostline is the
only public console script. It then resolves the MIT license, asset manifest,
third-party notice, environment, character/security, and diagonal-locomotion
runtime atlases from the installed package; validates the manifest's exact
three-file runtime declaration; verifies retired menu art is not a dependency;
renders a hidden gameplay frame; and proves the wheel is playable outside the
source checkout. CI repeats the same probe from the generated source archive.
The wheel remains runtime-only. The sdist is intentionally the reproducible
source archive: it includes the lock file, authored art and provenance,
screenshots, benchmark evidence, wiki, release scripts/workflows, static web
sources, and all runnable Ghostline tests. `verify_source_archive.py` audits
that contract without extraction; the tag workflow adds `--release` so the
final neural JSON/CSV, parity, and throughput evidence must also be present in
the archive. The three tests coupled to the retired
`neon_arena` implementation are excluded consistently with that package; the
legacy source remains repository-only engineering history.

## Release workflow

Ordinary pull requests run the complete suite on Windows and Linux, a practical
1,000-seed generator audit, both clean-install probes, and a human-only web
diagnostic. The tag/manual release workflow first runs the complete suite,
10,000-seed audit, benchmark-harness smoke, source-archive audit, and
`verify_release_evidence.py`. The evidence gate requires the exact selected
checkpoint and ONNX graph, 3,000 canonical 8M episode records, all thresholds,
1,000-transition recurrent parity, a recorded tier-6 throughput of at least
3,000 decisions/second, and the selected MP4 demo. The measured WSL2 result is
3,194 decisions/second; the earlier 5,000/s target remains a documented
optimization limitation. Windows and web builds cannot
start until that job passes. A `v*` tag creates a GitHub Release containing the
Windows and web archives, wheel, sdist, checkpoint, deployment ONNX, model card,
final JSON/CSV evidence, parity/throughput audits, and demo video; manual
dispatch builds the same artifacts without publishing.

## Refreshing the dependency lock

After deliberately updating direct pins in `pyproject.toml`, regenerate the
single reviewed constraints file under Python 3.13:

```powershell
python -m piptools compile --extra=agent --extra=build --extra=dev --extra=media --extra=train --extra=web --output-file=requirements.lock --strip-extras pyproject.toml
python -m pip check
```

Do not update the lock opportunistically during a long training run. Stop the
run, update and smoke-test the environment, then resume only from a compatible
checkpoint.
