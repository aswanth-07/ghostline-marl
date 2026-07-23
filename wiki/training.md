---
title: Ghostline Training
updated: 2026-07-24
status: active
---

# Training

## Adaptive-security MAPPO status

The Env-v3 adversarial-security track now has a measured release result.
`GhostlineSecurityParallel-v0` exposes simultaneous semantic
operative actions through PettingZoo. A parameter-shared local-grid/set/ego
encoder feeds a 256-unit GRU and separate intent, target, radio, and ability
heads. Execution is decentralized. A distinct MLP critic consumes the 64-value
global team state only during training.

`marl_train.py` implements recurrent rollout state, episode-boundary resets,
team GAE, factorized masked log probabilities, clipped MAPPO objectives,
central-value clipping, entropy regularization, gradient clipping, resumable
fingerprinted checkpoints, worst-tier validation selection, and disjoint 10M
training, 11M validation, and one-way 12M+ final-test slices. The frozen
published Env-v2 champion is the
default training opponent and its SHA-256 is recorded in every resume contract;
changing it fails closed. Worst-tier selection breaks ties by tier-six stop
rate, mean all-tier stop rate, damage, detections, and delay. Evaluation emits JSON plus aggregate and
per-episode CSV with Wilson intervals. A real CPU optimizer/checkpoint smoke run
is part of the test suite, and a CUDA calibration verified the optimizer and
checkpoint path.

Security optimization begins with online, observation-only imitation of the
same deterministic tactical controller shipped as the no-model fallback. The
warm-up is deliberately entropy-regularized so it supplies useful pursuit and
radio coordination without collapsing recovery exploration. MAPPO then uses
an exact, named reward ledger: terminal containment, damage, detection, denied
data, survival pressure, one-time radio assists, invalid actions, formation spacing, and
discount-matched potential shaping. The potential combines team proximity,
awareness, trace, mission progress, and damage and is zeroed at termination;
this preserves the terminal objective while making early pursuit learnable.
Every operative receives the same team reward and its `info` record exposes
components whose sum is tested against the emitted reward.

After each validation gate, the adaptive sampler assigns 70% of new episodes
to the weakest held-out tier (split across ties) and retains 30% uniform replay
across every selected tier. The probability vector is checkpointed and logged,
so resuming reproduces the curriculum instead of silently returning to uniform
sampling. Rollout diagnostics also include factorized action histograms and
mean reward components for detecting policy collapse or reward exploitation.
Radio shaping is capped after the first possible teammate broadcasts, so a
policy cannot earn unbounded reward by retransmitting unchanged information.
Every held-out gate writes an immutable step-numbered policy checkpoint in
addition to the mutable latest/champion pointers, preventing later regressions
or changed tie-break logic from destroying a useful earlier policy.
The tactical-teacher source is part of the security environment fingerprint;
changing its role logic invalidates older warm-ups and resume checkpoints even
when simulation and observation tensor shapes remain unchanged.
Evaluation and human Adaptive Contracts batch all active operative observations
into one recurrent actor forward pass per tactical decision. This preserves
decentralized inputs and per-agent recurrent state while avoiding five serial
neural calls per decision.

The selected 256-unit policy was initialized from the strategic teacher, then
trained with 50% easier scripted opponents while every selection window remained
exclusive to the frozen neural runner. Two disjoint validation measurements
were used for selection; neither solved every tier.
The untouched 13M final slice measured `4/0/8/16%` stop rates across tiers 3-6
(25 contracts per tier), versus the teacher's 2% mean and the learned policy's
7% mean. Tier 4 remained at zero stops and is explicitly unresolved. The full
report, Wilson intervals, failed 12M predecessor, and checkpoint hash are under
`benchmarks/security/`. Lightweight distributions without PyTorch continue to
use the deterministic observation-only fallback.

```text
ghostline train-security --hours 72 --envs 8 --rollout 64 --tiers 3,4,5,6 --runner-model models/ghostline-policy.pt --bc-warmup-steps 10000
ghostline train-security --init-model artifacts/security-bc/champion.pt --bc-warmup-steps 0 --no-resume --hours 72
ghostline train-security --init-model artifacts/security-bc/champion.pt --scripted-opponent-fraction 0.5 --bc-warmup-steps 0 --no-resume --hours 72
ghostline evaluate-security --model models/ghostline-security.pt --episodes-per-tier 25 --seed-start 13000000
python scripts/verify_security_release_evidence.py
```

`--init-model` starts a fresh optimizer from a fingerprint-compatible security
policy and records its path plus SHA-256. It is the explicit BC-to-MAPPO stage
boundary; it cannot be combined with an existing resume checkpoint.
`--scripted-opponent-fraction` supplies easier terminal-win episodes during
training while checkpoint selection continues to use only the frozen neural
runner. The fraction is recorded in the resume contract; changing it requires a
fresh run.

## Current neural release result

The selected 384-unit GRU BC+DAgger checkpoint passed the one-time 8M final
evaluation at `99.8/100.0/96.4/98.0/99.0/89.6%` across tiers 1-6 (500 unseen
seeds per tier). Its SHA-256 is
`76baa30af55cdaa2e71bb6ba06672bd9203455552358017505685827240b2e47`.
The immutable ledger is consumed and binds the JSON plus both CSV outputs to
fingerprint `521c449a...e129`.
FP32 ONNX achieved 1,000/1,000 deterministic recurrent-action matches; a 28.3%
smaller INT8 candidate changed five actions and was rejected. The historical
fingerprint PPO pilot also regressed the matched-seed DAgger rollback and is
retained as a negative result. See [`benchmarks/neural/README.md`](../benchmarks/neural/README.md)
for the full training lineage, selection gates, final metrics, and deployment
parity. No human-superiority claim is made without the planned human cohort.

Environment fingerprints canonicalize CRLF and LF source line endings before
hashing. This keeps the same semantic simulation/controller contract portable
across Windows and Linux checkouts while every other source-byte change still
invalidates datasets, checkpoints, exports, and release evidence.

## Model and observation contract

- Pure PyTorch 2.13 learning stack: behavior cloning, DAgger collection, recurrent clipped PPO, GAE, RND, and checkpointing are implemented in-project without Sample Factory, Stable-Baselines3, TorchRL, or another RL-framework runtime.
- `GhostlineEnv-v2` shared tactical inputs, including the explicit eight-value objective vector and twelve 13-feature live facility-security rows.
- Local-grid convolution, masked target/entity attention, ego/objective/ray encoders, and a fused 384-value representation.
- Configurable 256- or 384-unit GRU; 384 is the default and 512 is load-only legacy compatibility.
- Separate 256-unit policy/value paths, legal-action masking, and auxiliary goal-bearing and danger heads.
- Recurrent updates reset hidden state at episode boundaries. Imitation defaults
  to 64 supervised decisions preceded by as many as 32 real burn-in decisions;
  padded timesteps never contribute to loss.

## Hybrid pipeline

The historical pure-PPO attempt reached 2.98 million decisions with zero tier-one successes and remains negative evidence. After the 95/97/99% operative-speed curve and five-operative Ghostline threat pyramid were frozen, the recalibrated observation-only teacher passed two disjoint 100-seed-per-tier gates at `100/100/99/99/99/86%` and `100/100/99/99/99/89%`. The reports are [`teacher-fast-ops-validation-a-100.json`](../benchmarks/teacher/teacher-fast-ops-validation-a-100.json) and [`teacher-fast-ops-validation-b-100.json`](../benchmarks/teacher/teacher-fast-ops-validation-b-100.json), both bound to fingerprint `521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`. These are training-process gates, not a final-test or neural-policy claim. Every earlier corpus, checkpoint, and optimizer state remains historical-only.

1. The deterministic teacher consumes only the public v2 observation. It uses the sticky objective selected by the simulation, a visible six-tile navigation look-ahead, projected-footprint collision/ray clearance, live facility-security telemetry, explicit guard grade, cone exposure, action masks, and reserved pulse logic. It has no simulation object or renderer-only state.
   Tier 6 uses the validation-selected 1.8× directional-inertia scale. The setting changes only controller commitment, not observations or game rules.
2. Teacher and DAgger episodes are stored as compressed, independently recoverable trajectory files with a manifest declaring the observation contract and seed range.
   The manifest also records a SHA-256 fingerprint of simulation, generation,
   environment, controller, configuration, and entity-contract sources. The
   collector hashes them again at completion and fails closed if the game
   changed while trajectories were being produced.
   The sequence sampler validates every root's complete manifest and exact
   current fingerprint before reading a trajectory. Neural policy files and
   resumable BC/PPO optimizer states carry the same fingerprint and fail closed
   when missing or stale, so pre-freeze evidence cannot silently enter a run.
3. Behavior cloning keeps masked 36-way cross-entropy dominant and adds a 0.25-weight factorized auxiliary derived from the same legal joint logits reshaped as `[pulse=2, dash=2, move=9]`. Marginal move, dash, and pulse NLLs improve combinatorial and rare-action sample efficiency without changing `Discrete(36)`, inference, or checkpoint architecture. Goal-bearing, visible-danger, and a low-weight Huber value auxiliary remain secondary; value regression cannot dominate the action loss.
   Every root is split independently by actual tier using a portable SHA-256
   rank over split seed, root index, tier, and episode filename. The default
   90/10 split guarantees at least one train and one held-out episode for every
   root/tier group; duplicate episode identities and singleton production
   groups fail closed. The split digest and complete membership are written to
   `data-split.json` and must match on optimizer resume.
   Recurrent windows use the deterministic `ghostline-window-strata-v1` mix:
   50% uniform, 10% terminal, 15% teacher-action change, 5% teacher dash,
   10% teacher pulse, and 10% any joint teacher/behavior recovery. Eligible
   episodes are sampled before an eligible anchor, preventing event-heavy
   episodes from dominating. Missing categories fall back explicitly to
   uniform and are counted instead of being silently relabelled. This mix was
   chosen after the current corpus audit found dash in 30.10% of teacher steps
   but pulse in only 0.631%; DAgger joint recovery was 8.61%-13.34%.
   Merely placing a rare event inside a 64-step window would still dilute its
   loss. The anchored tick therefore receives a bounded priority: 2x for
   endpoint/action-change/dash, 6x for pulse, and 4x for recovery. Priority is
   combined with movement-recovery weighting by `max`, never multiplication,
   so no tick exceeds 6x. The dominant joint loss and factorized move/dash/pulse
   losses use this weight; auxiliary geometry/value losses remain unweighted.
   Positive dash/pulse precision, recall, and counts are reported separately
   from headline component accuracy.
4. DAgger executes a configurable mixture of teacher and neural actions but labels every visited recovery state with the teacher action. Clean round 1 uses `beta=0`: the neural policy executes every action, failures remain in the dataset, and the teacher supplies corrective labels on every visited state.
   A mixture below `beta=1` now fails before creating an output directory unless
   a behavior checkpoint is supplied; it can no longer silently record a
   pure-teacher corpus under a mixed-policy manifest. Rollout inference and
   optimization use separate devices: collection defaults to persistent CPU
   workers while behavior cloning defaults to CUDA when available. Each worker
   loads the checkpoint once, pins PyTorch model kernels to one CPU thread, and
   reuses that policy for all assigned episodes. The teacher/policy coin flips
   come from a seed-derived RNG owned by each episode, making the complete
   trajectory invariant to process count and completion order.
   The legacy Python `run_dagger(..., device=...)` keyword remains accepted as
   an alias for the training device only; new commands should use the explicit
   `--collection-device` and `--training-device` flags.
   After every selection, the teacher explicitly observes the action that was
   actually executed. Its next-step directional inertia and collision-stall
   memory therefore follow the policy-induced trajectory rather than a
   counterfactual teacher trajectory. Recovery data collected before this
   synchronization fix is excluded and regenerated from the frozen BC model.
5. Recurrent PPO fine-tunes the selected checkpoint with exact reward accounting, curriculum replay, and a decaying Random Network Distillation bonus computed from raw player-equivalent observations. Geodesic progress uses discount-consistent potential shaping, `0.35 * (0.995 * Phi(s') - Phi(s))`; unlike the former asymmetric clipping, a closed movement cycle cannot yield positive discounted shaping return.
   Vector workers use Gymnasium same-step autoreset. The learner receives the
   new initial observation on the terminal transition, clears the corresponding
   GRU state, and never trains on the ignored-action dummy transition produced
   by next-step autoreset.
   Same-step reset observations are explicitly masked out of RND reward, the
   default novelty coefficient is capped below task-return scale, and an
   independently loaded frozen imitation checkpoint supplies a decaying
   `KL(reference || PPO)` anchor. Clip fraction, approximate KL, and anchor KL
   are logged. Validation advances through disjoint held-out offsets instead
   of repeatedly selecting on the same twenty seeds. Resume restores the
   curriculum before worker creation plus CPU/CUDA/NumPy RNG state. A fresh
   PPO experiment writes `initial-rollback.pt` exactly once and binds it to the
   source-checkpoint SHA-256; this immutable copy is both the KL reference and
   the explicit rollback policy. `best.pt` is not created by partial curriculum
   validation: all six tiers must be present, and an equal success tuple keeps
   the earlier checkpoint because the internal gate has no damage/time
   tie-break telemetry. The initial validation cursor and curriculum tier are
   CLI-controlled, stored in a strict resume contract, and validation offsets
   advance without wraparound or seed reuse. Changing any optimization or
   validation contract setting requires a new experiment initialized from the
   selected checkpoint rather than a silent optimizer resume.

Useful commands:

```text
ghostline imitate collect --output artifacts/teacher-current --episodes-per-tier 100
ghostline imitate bc --dataset artifacts/teacher-current --output artifacts/bc-current --updates 20000 --sequence-length 64 --burn-in 32 --validation-windows 128
ghostline imitate dagger --base-dataset artifacts/teacher-current --initial-checkpoint artifacts/bc-current/best.pt --output artifacts/dagger-current
ghostline train --init-checkpoint artifacts/dagger-current/round-3/model/best.pt --hours 24 --initial-curriculum-tier 6 --initial-validation-cursor 3800
ghostline ablate --bc-checkpoint artifacts/bc-current/best.pt --dagger-checkpoint artifacts/dagger-current/round-3/model/best.pt
```

DAgger rounds are restartable without recollecting completed rounds. `--rounds`
is the final numbered round, while `--start-round` selects the first round to
execute. On resume, the command requires every earlier `round-N/data` directory
and automatically trains on the base dataset plus all earlier and newly
collected round datasets. `--initial-checkpoint` must be the policy that should
drive the resumed round, normally the preceding round's `model/best.pt`:

```text
ghostline imitate dagger --base-dataset artifacts/teacher-current --output artifacts/dagger-current --initial-checkpoint artifacts/bc-current/best.pt --start-round 1 --rounds 1 --episodes-per-tier 20 --updates-per-round 3000 --beta-start 0 --beta-decay 0.5 --recurrent-size 384 --collection-device cpu --training-device cuda --collection-workers 12
```

For the clean recovery run, `--beta-start 0` is explicitly supported and makes
round 1 fully policy-executed while retaining failures. Each imitation batch is
root-balanced: the current recovery root supplies 50% of sampled sequences and
the aggregate base/prior roots supply the other 50%. States where the teacher's
move differs from the executed behavior move receive 4x weight in the dominant
36-way cross-entropy and factorized move NLL; recovery-move accuracy is logged
separately. Round 2 is gated on external closed-loop evaluation of round 1.

The 12-worker rollout default was measured against 20 workers on the available
24-core Windows host using two repeated, identical 24-episode lesson-one
policy-driven collections. Twelve workers completed in 7.359 s and 7.262 s;
twenty completed in 8.896 s and 8.779 s. The lower-contention setting was about
17% faster by mean wall time. A separate 50/50 teacher-policy parity smoke
collected four episodes and 944 transitions with one and two workers; every NPZ
array, episode record, success result, and transition count matched exactly.
Multiworker behavior collection rejects non-CPU rollout devices so a DAgger
command cannot accidentally replicate a CUDA policy into many processes.

Checkpoint selection uses 128 deterministic stratified windows drawn only from
held-out episodes and evaluated every 100 updates, never the current training
minibatch. On the frozen 1,800-episode corpus this produces 1,620 train and 180
held-out episodes. The deterministic held-out composition is 64 uniform, 13 endpoint,
19 action-change, 6 dash, 13 pulse, and 13 recovery windows. Pure-teacher data
has no recovery disagreement, so those 13 requests are explicitly reported as
uniform fallbacks until DAgger roots are added.

At 64 supervised plus 32 burn-in decisions, the held-out batch contains exactly
104,853,504 bytes (99.996 MiB) of observation/label tensors. A no-gradient
GRU-384 CUDA forward on the 16 GB development GPU peaked 1,115,376,640 bytes
(1,063.706 MiB) above the loaded-policy baseline, leaving ample headroom for
the training minibatch. Held-out tensors remain on CPU between checks.
An actual GRU-384 CUDA backward, gradient-clip, and AdamW step on a 16-window
64+32 batch from the 1,620-episode train split completed in 0.756 s, peaked
301,048,320 bytes (287.102 MiB) above its loaded training baseline, saw 15
pulse-positive supervised ticks, and left every parameter finite.
`best.pt` records the lowest held-out episode-validation loss, while `latest.pt`
remains the resumable final optimizer state returned to the caller and used
for the external closed-loop gate. `training-state.json` is atomically updated
at every validation with train/held-out loss; joint, move, dash, pulse, and
recovery accuracies; positive dash/pulse recall; split sizes; and requested,
resolved, and fallback window strata. Resume validates the split, sampler,
batch/window contract and restores NumPy plus Torch CPU/CUDA RNG state.

DAgger reserves a disjoint 10,000-seed subrange per round inside each tier's
training namespace: round 1 uses offset 10,000, round 2 uses 20,000, and round 3
uses 30,000 before the collector adds `tier * 100,000`. Thus the first three
rounds remain below seed 700,000 and cannot cross into validation at 1,000,000.
The collector rejects per-tier round sizes above 10,000 and any schedule whose
largest seed would leave the training namespace.

`artifacts/dagger-256-v1` is retained only as contamination provenance: its
historical 500,000 offset placed tier-5 and tier-6 recovery trajectories in the
validation namespace. `artifacts/dagger-256-clean` also predates the current
environment fingerprint. Both are rejected from every current neural-model
training and selection claim; admissible recovery training starts from the
fresh `artifacts/dagger-current` lineage.

The first beta-zero clean recollection was stopped before BC after the genuinely
fresh 3M teacher gate missed the release thresholds (tiers 1–6:
100.0%, 100.0%, 97.0%, 94.8%, 96.8%, and 80.8%). Its 109 completed, pre-gate
episode files are preserved without a completion manifest or model checkpoint
under `artifacts/dagger-256-pregate-audit`; they are audit evidence only and are
excluded from training. After validation-only controller calibration and the
three-guard Countermeasure QoL curve, the teacher passed the then-untouched 6M
gate. The later final mechanics freeze changed route, security, terminal, and
patrol generation, so 6M is now historical evidence rather than a current
release claim. Frozen-distribution DAgger recollection restarts from scratch
with source fingerprints; no pre-freeze recovery file is eligible for training.

## Historical pre-freeze BC width candidates

The pre-final-mechanics teacher produced `artifacts/teacher-v2-final`: 600 successful
training episodes and 182,426 retained transitions from 626 attempts. Its
maximum scheduled seed is 601,999, so the dataset stayed inside the training
namespace, but its manifest fingerprint is stale. It and every checkpoint
trained from it are evidence only and are rejected as current initialization,
recovery, PPO, export, or release inputs.

Independent from-scratch 5,000-update CUDA runs on that exact dataset produced
two externally gated candidates:

| Candidate | Parameters | Final fixed-validation loss | Fixed-validation action accuracy | Time |
|---|---:|---:|---:|---:|
| GRU-256 `artifacts/bc-256-final/latest.pt` | 994,026 | 0.2261 | 95.12% | 543.2 s |
| GRU-384 `artifacts/bc-384-final/latest.pt` | 1,453,930 | 0.2182 | 95.70% | 548.6 s |

The 384-unit historical model had the lower supervised validation loss. Neither
candidate is eligible for current closed-loop selection or DAgger. The
machine-readable historical summary is `artifacts/bc-final-summary.json`.

## Curriculum, selection, and stopping

Seven reverse lessons progress from one-room linking/extraction through navigation, multiple targets, cameras, guards, countermeasures, and the unchanged six-tier distribution. The current tier receives 70% of samples and earlier tiers 30%; tier six mixes full contracts and replay equally.

Checkpoints are selected only at held-out validation windows by worst-tier success, then tier-six success. Training returns never compete with validation scores. A pre-validation checkpoint is retained only as a fallback. Acceptance must pass twice consecutively and the pass count is persisted across resume.

## Seeds and acceptance

- Training: `0-999,999`
- Validation: `1,000,000-1,049,999`
- Final test: `2,000,000+`. Audit slices become immutable when an attempt begins. The 2M-7M slices are historical; the current champion consumed the locked 8M slice exactly once.

Validation allocates six non-overlapping 8,000-seed tier blocks inside the reserved 50,000-seed range. Shared seed helpers enforce these bounds so curriculum validation and calibration cannot silently drift into another namespace.

Neural BC, DAgger, and PPO checkpoints are compared with
`python scripts/evaluate_validation_policy.py --checkpoint <policy.pt> --validation-offset <offset> --episodes <count> --output <report.json>`.
The evaluator always runs all six tiers, derives every seed through
`validation_seed(tier, offset + episode)`, and exposes no arbitrary or final-test
seed input. It rejects stale/missing checkpoint fingerprints and invalid or
overlapping windows before starting deterministic CPU workers (one Torch thread
each). The auditable JSON/CSV pair records the environment fingerprint,
checkpoint SHA-256, Wilson intervals, failure reasons, damage, detections,
duration, path efficiency, inference latency, exact episode seeds/action hashes,
and the lexicographic worst-tier-first selection tuple. The tuple ranks worst-tier
success, Tier-6 success, lower damage, higher path efficiency, lower duration,
then lower latency; final-test results never participate in checkpoint selection.

Final evaluation uses 500 unseen seeds per tier and reports deterministic JSON, aggregate CSV, and episode CSV evidence with Wilson 95% intervals, failure reasons, damage, detections, duration, trace, optional data, path efficiency, inference latency, exact episode seeds, action-sequence hashes, and every cumulative reward component plus its verified total. The evaluator reconstructs that component map from the real Gymnasium terminal keys and rejects a sum mismatch. The report binds the checkpoint SHA-256 and current environment fingerprint. The evaluator atomically changes a declared `reserved_unopened` slice to `opened_locked` before scheduling an episode, then to `consumed` or `aborted_retired`; there is no force/reopen path. Required neural success is at least 95% on tiers 1-5 and at least 85% on tier 6. The selected checkpoint consumed the reserved 8M slice exactly once and passed all six thresholds; those results are frozen release evidence and cannot be reopened for selection.

ONNX export performs the locked 1,000-transition recurrent-action comparison and writes a sibling `.parity.json` report. The requested `--output` is always the canonical FP32 model. Optional dynamic-INT8 quantization writes a separate candidate, repeats the same deterministic recurrent comparison independently, and selects it only with zero action mismatches. A rejected or failed quantization attempt leaves FP32 as the safe deployment fallback. The report records both artifacts' byte sizes and SHA-256 hashes, their parity results, the INT8 size reduction, and the precision copied to `--deployment-output`; FP32 parity failure still rejects the export entirely.

Before any portfolio package is built, `scripts/verify_release_evidence.py`
independently re-aggregates all 3,000 episode records, recomputes Wilson
intervals, validates the consumed one-way slice and its output hashes, and binds
the final report, checkpoint, selected ONNX bytes, recurrent parity report,
source-fingerprint throughput report, and demo video. This verifier is
read-only and has no code path that schedules or reopens final-test episodes.

## Current-fingerprint teacher qualification

The 2026-07-14 gameplay revision keeps every operative live in facility
telemetry shared by human and policy controllers, raises chase speeds to
95/97/99% of the runner, and uses a readable five-operative Tier-6 pyramid.
The resulting environment fingerprint is
`521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`.

The observation-only teacher was recalibrated exclusively in the validation
namespace. No teacher or environment source changed between these two
disjoint gates:

| Gate | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 |
|---|---:|---:|---:|---:|---:|---:|
| A | 100/100 | 100/100 | 99/100 | 99/100 | 99/100 | 86/100 |
| B | 100/100 | 100/100 | 99/100 | 99/100 | 99/100 | 89/100 |

Both gates exceed the teacher thresholds of 95% on tiers 1-5 and 85% on tier
6. Their machine-readable reports are
[`teacher-fast-ops-validation-a-100.json`](../benchmarks/teacher/teacher-fast-ops-validation-a-100.json)
and
[`teacher-fast-ops-validation-b-100.json`](../benchmarks/teacher/teacher-fast-ops-validation-b-100.json).
They qualify training data collection only; they are not neural-policy or
final-test claims. Collection produced 600 successful trajectories and 138,610
transitions after six failed attempts were discarded. The untouched 8M
final-test slice remains sealed until validation selects and freezes one neural
checkpoint.

## Historical teacher audit series

Every final-test slice is a one-way door: after its report is inspected, the
slice is immutable. A failure may motivate a correction, but the same slice
cannot judge or select that correction; only a later untouched slice can.
The complete audit history is tracked in
[`benchmarks/teacher/audit-history.json`](../benchmarks/teacher/audit-history.json).

| Slice | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| 3M | 100.0% | 100.0% | 97.0% | 94.8% | 96.8% | 80.8% | Retired - failed |
| 4M | 100.0% | 100.0% | 96.6% | 95.2% | 97.2% | 83.8% | Retired - failed |
| 5M | 100.0% | 100.0% | 97.6% | 93.6% | 97.4% | 85.6% | Retired - failed |
| 6M | 100.0% | 100.0% | 95.8% | 96.2% | 97.2% | 88.8% | Passed then-current curve; now historical |

The 3M and 4M audits exposed Tier-6 under-commitment. Directional inertia was
then calibrated exclusively on two disjoint 200-seed validation slices: the
selected 3.5x scale raised success from 87.5%/82.5% to 89.0%/90.0% and reduced
damage. The setting was not selected on a final-test slice. The 5M audit showed
that Tier 6 now passed but Countermeasure retained a non-monotonic integrity
spike. Its four-guard curve was reduced to three guards while keeping cameras,
networked doors, guarded terminals, and three pulse charges. No controller,
mechanics, or generation setting changed while the 6M gate was open. The later
portfolio mechanics pass deliberately created a new procedural distribution
and therefore retired this result from release-gate status.

The untouched 6M report covers 500 deterministic seeds per tier:

| Tier | Success | Wilson 95% interval | Median duration | Mean damage | Mean path efficiency |
|---|---:|---:|---:|---:|---:|
| 1 - Orientation | 500/500 (100.0%) | 99.24%-100.00% | 15.69 s | 0.000 | 0.721 |
| 2 - Surveillance | 500/500 (100.0%) | 99.24%-100.00% | 15.42 s | 0.000 | 0.747 |
| 3 - Patrol | 479/500 (95.8%) | 93.66%-97.24% | 30.42 s | 0.528 | 0.537 |
| 4 - Countermeasure | 481/500 (96.2%) | 94.14%-97.55% | 32.78 s | 0.568 | 0.655 |
| 5 - Lockdown | 486/500 (97.2%) | 95.36%-98.32% | 36.22 s | 0.418 | 0.716 |
| 6 - Ghostline | 444/500 (88.8%) | 85.73%-91.27% | 40.63 s | 0.948 | 0.581 |

The complete tracked report is
[`teacher-release-gate-6m-500.json`](../benchmarks/teacher/teacher-release-gate-6m-500.json)
with a [CSV export](../benchmarks/teacher/teacher-release-gate-6m-500.csv).
This historical result validated the original teacher trajectory corpus, but it
is not the final frozen-distribution or neural-policy result. The current
teacher separately passed the two validation gates above. Fresh corpus
collection, BC/DAgger selection, the locked 500-seed-per-tier final test, and
ONNX parity have since completed as documented at the top of this page. Only
the human comparison remains pending.

## Equal-budget ablations

- Random policy
- Fair observation-only teacher
- Pure recurrent PPO
- Pure feedforward PPO
- Behavior cloning only
- Behavior cloning plus PPO
- Behavior cloning plus DAgger plus PPO/RND

All trainable comparisons use equal environment-step budgets and multiple seeds.
