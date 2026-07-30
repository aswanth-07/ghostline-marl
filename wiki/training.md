---
title: Ghostline Training and Evaluation
updated: 2026-07-30
status: active
---

# Training

Ghostline separates the released single-agent benchmark from the developmental
multi-agent campaign:

- public v1 reproduces the shipped runner and its immutable evidence;
- v2 trains a new field-systems runner and adversarial security team under new
  fingerprints; the runner may use a declared overlap-only v1 weight
  transplant, while the required scratch run remains an ablation;
- no v1 checkpoint is relabeled or loaded as a v2 checkpoint.

## Published v1 evidence

The shipped 384-unit recurrent runner passed its one-time 8M audit over 500
unseen contracts per tier:

| Tier | Success |
|---|---:|
| 1 | 99.8% |
| 2 | 100.0% |
| 3 | 96.4% |
| 4 | 98.0% |
| 5 | 99.0% |
| 6 | 89.6% |

The checkpoint, ONNX graph, parity report, benchmark JSON/CSV, demo, and
throughput report are bound to environment fingerprint
`521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`.
Those records say `GhostlineEnv-v2` because that was the internal contract name
when the evidence was frozen. Publicly, the exact released environment is
`GhostlineEnv-v1`. Both labels are recorded; neither the bytes nor the evidence
chain is rewritten.

## V2 seed namespaces

Runner-v2 keeps the project-wide procedural schedule under a new environment
fingerprint:

- training: `0-999,999`;
- validation: `1,000,000-1,049,999`, with fixed per-tier strides;
- final test: `2,000,000+`, using a newly declared untouched slice.

The numeric ranges overlap the published runner's historical ranges, but the
environment fingerprint does not: v1 and v2 seeds identify different
distributions and their reports cannot be combined.

The tracked v2 runner ledger reserves `20,000,000` for the first selected
champion: 500 contracts per tier across Standard, Ghost, Speed, and Greed. The
evaluator locks that entry before the first episode and marks every attempted
run consumed or aborted-retired. It has no overwrite or reopen path.

Security-v2 uses its separate opponent-evaluation schedule:

- training: `10,000,000+`;
- validation: `11,000,000+`;
- final test: the tracked one-way `14,000,000` slice.

The retired security contract already consumed 12M and 13M. Current v2 uses a
new numeric window and a new fingerprint. Every opened final slice is retired,
including failed attempts. Exact ranges and consumed slices must be written to
each checkpoint and report. A long run does not begin until the current source
fingerprint, dependency lock, configuration, seed manifest, and smoke report
are archived together.

## V2 runner learner

The current runner environment fingerprint is
`b0b42206bef47cae86ff8a20d4967519e310d2d78d4e865dca3bb06785382a63`.
Every runner checkpoint under the preceding
`01d1b7835d17172edc8dda1158d93e5c24e9362cc3472722b4aee77452c75e8f`
fingerprint—including all `runner-v2-ghost-specialist-*`,
`runner-v2-ghost-curriculum-*`, and `runner-v2-ghost-sil-*` artifacts—is
diagnostic-only. None can be resumed, selected, migrated, or reported under
the current contract. The clean campaign initializes only through the
fingerprint-verified published-v1 overlap transplant.

The Ghost contract now evaluates a recoverable, player-readable outcome:
extract with zero damage while live trace is below `95%`. Lifetime maximum
trace remains telemetry, but no longer makes a recovered run fail forever.
This matches lockdown's design as escalating pressure rather than instant
failure. Dense reward still charges exposure, detections, loud dashes, rising
awareness, and damage; the mission potential now follows live trace so breaking
contact receives discount-correct credit.

The runner uses `RunnerPolicyV2`, a player-equivalent recurrent actor-critic
with:

- local-grid convolution;
- masked attention over terminals, perceived entities, and public field
  targets;
- ego, objective, ray, and field MLPs;
- directive FiLM;
- a GRU with separate policy/value decoders;
- a shared 36-action movement/dash/pulse head plus binary decoy, crouch, and
  interact heads, with a small 288-action residual for full expressivity;
- exact masking across all 288 semantic actions;
- objective-bearing and visible-danger auxiliary heads trained from the same
  recurrent latent using only public observation labels. Their weighted losses
  are enabled in the baseline and accounted separately.

The baseline optimizer is recurrent PPO/GAE with a 512-decision recurrent
window, `gamma = 0.999`, `lambda = 0.98`, and fixed `0.05` reward scaling.
Rollouts retain recurrent
sequence boundaries and reset masks; feedforward flattening is not an allowed
substitute. Checkpoints include optimizer, curriculum, all relevant random
states, live vector-environment episode state, decisions, update count, source
fingerprint, and the full training configuration. Resume fails closed if any
contract field differs.

The version-explicit entry point is:

```powershell
ghostline train-runner-v2 --help
ghostline train-runner-v2 --output artifacts/runner-v2/preflight --published-v1-init models/ghostline-policy.pt --dry-run --cpu
```

It uses asynchronous vector environments by default, samples disjoint
per-worker episode schedules, validates on held-out windows, promotes tiers
after consecutive passes, and can stop after two complete acceptance passes.
`--sync-envs` is reserved for deterministic debugging and smoke tests.
Every fresh campaign writes `experiment-manifest.json` with the dependency,
hardware, source, environment/model/trainer, initialization, opponent, budget,
and seed contracts. A dry run validates all checkpoint bytes without starting
an environment worker.

The recommended campaign initializes through `--published-v1-init`. This is a
provenance-recorded parameter transplant: it verifies the exact published-v1
checkpoint, copies only compatible encoders/recurrent/value parameters, expands
the 36 base action logits across the 288 semantic variants, and keeps new field
channels neutral. It never relaxes the v2 checkpoint fingerprint. The required
ablation trains from orthogonal initialization.

The runner reward stays in `env_v2.py`, never the simulation:

- dominant directive-complete extraction and bounded partial-extraction/data
  rewards;
- potential-based geodesic mission progress toward the directive's actual
  acquisition target;
- time and invalid-behavior costs;
- genuine exposure and first-detection objective costs;
- quiet-data bonus below the trace ceiling;
- exact named component accounting.

Exposure and detection are objective terms, not potentials: the design
intentionally changes the optimum away from a trace-saturated race. Route
progress remains potential-based because it should guide learning without
changing the successful route objective.

### Ghost recovery curriculum

`--ghost-training-stage` is an explicit training-only bridge for the first
human-security lesson. It never changes `GhostlineEnv-v2`, public play, or
release evaluation:

| Stage | Training roster |
|---:|---|
| `0` | full generated security; the only release setting |
| `1` | one guard, no cameras |
| `2` | one guard and one camera |
| `3` | two guards and one camera |

Stages are valid only with `--no-curriculum --directives ghost` and scripted
security. Each run uses a separate training seed range and validation cursor,
and the report/checkpoint records
`training_only_ghost_security_stage`. Partial-tier runs rank checkpoints only
over their declared tier, fixing the former sentinel behavior without allowing
partial evidence to satisfy the six-tier acceptance gate.

The trainer contract is `ghostline-runner-recurrent-ppo-v2.4`. The environment
fingerprint change makes every earlier v2 optimizer and policy snapshot stale,
regardless of trainer version.

Stage-1 diagnostics showed a hard-exploration plateau rather than numerical
instability: the selected policy extracted on 97% of 100 fresh contracts, but
only 35% met Ghost, and failed extractions averaged 96.8 maximum trace. With
security removed, the same policy reached 92% Ghost success. The missing skill
is therefore moving-guard avoidance, not navigation, hacking, or basic noise
budgeting.

V2.4 provides an optional recurrent self-imitation term to PPO. It selects only
complete self-generated successful episodes contained inside a rollout, then
imitates only their positive-advantage decisions. Advantage weights are
detached, normalized, and capped; partial episodes and every failed trajectory
are excluded. The default coefficient remains zero for ordinary baselines.
Under the retired maximum-trace contract, coefficient `0.2` briefly improved
two matched windows from `28/34%` to `34/36%` but then plateaued. That result
does not select a coefficient for the new environment.

Calibration on 100 disjoint Stage-1 validation contracts rejected a simple
maximum-trace threshold change: cutoffs from `75` through `95` moved success
only from `18%` to `30%`, while accepting saturated `100` trace made the rule
nearly unconditional. The recoverable rule produced `47%` for the retired
Stage-1 policy and `57%` for a clean published-v1 transplant. This supplies a
credible PPO starting distribution without making Ghost equivalent to ordinary
extraction.

The clean Stage-1 campaign is:

```powershell
ghostline train-runner-v2 `
  --output artifacts/runner-v2-recoverable-ghost-stage1-r1 `
  --published-v1-init models/ghostline-policy.pt `
  --tiers 3 --directives ghost --ghost-directive-fraction 1 `
  --ghost-training-stage 1 --no-curriculum `
  --training-seed-start 740000 --initial-validation-cursor 6700 `
  --envs 6 --rollout 512 --epochs 4 --minibatch-envs 3 `
  --learning-rate 5e-5 --entropy-coefficient 0.003 --gae-lambda 0.98 `
  --self-imitation-coefficient 0.05 `
  --validation-interval 100 --validation-episodes 100 `
  --validation-batch-size 10 --max-updates 1200
```

Stages 2, 3, and finally 0 initialize from the preceding stage's selected
checkpoint and use non-overlapping seed/cursor allocations. A stage advances
only after two deterministic validation windows reach its declared gate; the
full roster then has to pass the ordinary tier gate. A DAgger corpus is not
collected from the current v1 teacher because its v2 tier-3 qualification is
far below the policy threshold.

V2 success means both extraction and directive completion. Greed keeps the
objective, extraction gate, map cue, and shaping potential on unfinished
terminals after ordinary quota; ghost and speed may still extract after
missing their constraint, but receive only the bounded partial outcome and
cannot pass validation. This prevents reward, observation, mechanics, and
checkpoint selection from optimizing different tasks.

## V2 security learner

The shared dependency change binds the security environment to fingerprint
`1cc51952a7311dba146fe486b0a16babb48f77563fa93a06f5591d2e29a76b9d`.
Every earlier developmental security checkpoint, including those bound to
`97dfb60808aef79add8b9b67992daa03694d8a173c70d2fb1e20687e1d66c7c9`,
is diagnostic evidence only and cannot initialize, oppose, or be selected for
this contract.

The security benchmark uses parameter-shared recurrent MAPPO. One actor serves
all operative roles, but deployment remains decentralized: each agent receives
only its own observation and recurrent state.

The actor applies explicit role FiLM and uses an intent-conditioned target
pointer. It can therefore rank the same public doorway differently for SEAL,
PINCER, PURSUE, and INTERCEPT while retaining exact conditional masking.

The actor factorizes its policy over:

- ten semantic intents;
- ten public target rows;
- five radio messages;
- a binary role-gated ability.

All factors are masked before sampling and loss computation. A conditional
intent-by-target mask is applied during rollout, re-evaluation, and entropy
calculation, so PPO scores the same joint support it sampled. PINCER and SEAL
operate on public escape-route/door targets. The policy never receives the
runner's hidden objective, heading, or unseen exact position.

The centralized training critic is permutation-aware and agent-specific. It
encodes each operative block with shared weights, pools the active team with an
explicit presence mask, combines that with mission/facility context, and emits
one value for each operative slot. This matches the per-agent reward:

`shared team outcome + bounded attributed potential shaping`.

Inactive slots are removed from GAE, value loss, policy loss, entropy, and
normalization. This prevents padded agents and agents that disappear at an
episode boundary from contaminating gradients.

## Security reward ledger

Every security decision reports a bounded component ledger:

| Component | Purpose |
|---|---|
| `damage` | team credit for integrity damage |
| `contact_acquisition` | one first-contact credit per operative and episode |
| `runner_data` | cost when the runner secures data |
| `radio_assist` | capped first-information sharing credit |
| `invalid_action` | cost after mask/compatibility validation |
| `formation` | small bounded anti-stacking cost |
| `potential` | discount-matched team containment shaping |
| `terminal` | dominant stop/extraction outcome |
| per-agent shaping | discount-matched route coverage and awareness attribution |

The operative that establishes a new first contact receives a small one-time
attributed credit. Its per-agent component ledger is exact and the episode-level
contact set prevents reacquisition farming.

The team potential combines geodesic escape-route coverage, awareness, trace,
and the runner's partial mission progress. It is bounded before differencing.
Agent shaping is bounded separately. Reward components are summed exactly once;
tests assert the reported total and prevent repeated HOLD/detection farming.

The potential uses the same discount as GAE. Changing `gamma` without changing
the environment reward discount is a contract error.

## Opponent curriculum

Training one side forever against a fixed scripted opponent overfits the
opponent rather than the game. The first v2 campaigns establish compatible
runner and security snapshots. After those exist, the staged protocol is:

1. verify both deterministic tactical controllers;
2. warm the runner against scripted/frozen security;
3. freeze a runner snapshot and train security against a mixture of scripted,
   published-v1-compatible, and v2 snapshots where contract-compatible;
4. freeze the selected security snapshot and continue runner PPO;
5. alternate only after held-out regression gates pass;
6. preserve an earlier-opponent replay mixture to limit cycling.

`ghostline co-train-v2` automates this as concurrent frozen generations.
Opponent updates occur between generations, never mid-rollout. Every checkpoint records
the exact opponent hashes and mixture. Security accepts a repeatable
`--runner-pool` containing published-v1 adapters and/or native v2 runner
snapshots, with `--scripted-opponent-fraction` retaining the tactical baseline.

The coordinator also owns the host resource contract. Trainer subprocesses run
at below-normal Windows priority, numerical libraries are capped to one thread,
and the full process tree inherits a 50% logical-CPU affinity ceiling. The
long-run laptop baseline uses eight runner and four security environments; the
ceiling is enforced independently of worker behavior rather than inferred from
an average utilization sample.
Runner training accepts a repeatable `--security-opponent`; its actor receives
only the same local observations available during deployment, and exact resume
reconstructs its recurrent opponent state by action replay. Alternation begins
only after both first-stage validated champions exist. A result against one
training opponent is not a general security/runner claim.

## Curriculum and checkpoint selection

Tier promotion requires two consecutive held-out passes. After promotion,
sampling retains earlier tiers to prevent forgetting. At the full v2
distribution, half the budget targets the current difficult distribution and
half replays prior tiers and opponent snapshots.

Runner selection order:

1. worst-tier validation success;
2. tier-6 success;
3. damage;
4. trace;
5. path efficiency;
6. completion time and inference cost.

Security selection order:

1. worst-tier stop rate;
2. tier-6 stop rate;
3. all-tier stop rate;
4. damage and detections caused;
5. delay and path interception quality.

No final-test slice participates in architecture, reward, curriculum, or
checkpoint selection.

## Required baselines and ablations

Compare at equal environment-step budgets and multiple seeds:

- random legal runner/security;
- deterministic fair runner and tactical security;
- feedforward PPO/MAPPO;
- recurrent learner without adaptive curriculum;
- recurrent learner without per-agent critic/credit assignment;
- v2 runner without field-target encoder;
- final recurrent curriculum learner;
- frozen-opponent versus alternating frozen-opponent training.

Comparative claims require confidence intervals, not one run.

## Preflight before long training

All of the following must pass from a clean install:

1. full tests;
2. Gymnasium checker and PettingZoo parallel API;
3. exact 288-action round-trip and all factor masks;
4. 10,000-seed v2 readiness fuzz with zero failures;
5. deterministic replay for representative tiers/directives;
6. finite runner PPO smoke with a checkpoint/resume cycle;
7. finite MAPPO smoke with a checkpoint/resume cycle;
8. environment and optimizer throughput reports;
9. stale-checkpoint rejection for runner and security;
10. a frozen experiment manifest with dependencies, hardware, configuration,
    source commit, fingerprints, and seed namespaces.

Both trainers now implement this manifest and reserve the word `champion` for
a checkpoint selected by held-out validation. A budget-limited smoke without
validation produces `last-policy.pt`, never a misleading champion.

The 2026-07-28 preflight completed the 10,000-seed v2 audit with zero final
readiness failures, runner PPO fresh/resume, MAPPO fresh/resume against a mixed
frozen runner pool, exact terminal reward ledgers, a 1,000-transition v2
PyTorch/ONNX parity pass, isolated-wheel play, source-archive verification,
and the human-only web build. The measured Windows CPU throughput was 2,039
aggregate decisions per second (12,234 simulation ticks per second), below the
aspirational 5,000-decision target; worker count therefore must be calibrated
on the long-run host rather than assumed from core count.

The current pre-migration `models/ghostline-security.pt` is invalid for v2.
It may remain as historical evidence, but training, launcher selection, and
evaluation must reject it until replaced by a current-fingerprint checkpoint.

### What a finite smoke does not cover

The runner and security smokes prove that collection, loss, checkpointing, and
resume run and stay finite. They do not exercise learning dynamics, and the
preflight list must not be read as if they did. A smoke sized to a few hundred
decisions completes zero episodes, so no terminal reward ever reaches a value
target, and its explained variance describes only in-episode shaping.

Two consequences shape the first long campaign:

- Raw ledgers retain the `+20` directive-complete extraction and containment
  outcomes, while both critics train in fixed `0.05` scaled units. This keeps
  the conventional absolute value clip meaningful without a moving
  normalization statistic.
- The published-v1 value head is multiplied by the same fixed scale during
  overlap transplant. The full-horizon CUDA calibration then measured runner
  value loss `0.0045`, explained variance `0.62`, and finite gradients over
  8,192 samples.

## Final evaluation

V2 runner acceptance keeps the published standard:

- at least 95% success on tiers 1-5;
- at least 85% on tier 6;
- 500 unseen seeds per tier;
- Wilson 95% intervals;
- failure taxonomy, time, trace, damage, optional data, and path efficiency;
- Python/ONNX deterministic action parity on at least 1,000 recurrent
  transitions.

Version-explicit release commands are:

```powershell
ghostline evaluate-runner-v2 --model artifacts/runner-v2/ppo/best.pt --directives standard,ghost,speed,greed --episodes-per-tier 500 --seed-start 20000000 --slice-manifest benchmarks/runner-v2/final-test-slices.json --output benchmarks/runner-v2/final-test-20m.json
ghostline export-runner-v2 --model artifacts/runner-v2/ppo/best.pt --output models/ghostline-runner-v2.onnx --parity-samples 1000
```

Security evaluation reports stop rate with Wilson intervals, runner progress,
damage, detections, delay, team formation, reward components, and policy
latency against multiple frozen runner opponents. A security team is not called
strong based only on increasing episode duration or beating one scripted
runner.

No “better than a real player” claim is published until a matched-seed human
cohort is collected under a locked protocol.
