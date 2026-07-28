---
title: Ghostline Co-training Plan
updated: 2026-07-28
status: implemented
---

# Co-training plan

This page records the first campaign that trains the runner and the security
team in one session. The measurements are facts from the target host. The
implemented choices below are bound into manifests and checkpoints.

## What is already settled

Three things do not need building, and it is worth being explicit so effort
does not go into them:

- **Facilities are already fully randomized per episode.** Each environment
  advances its seed by the worker count on every reset, and the seed drives room
  geometry, vent pairs, device placement, patrol routes, and the door graph. A
  seed never repeats inside a run, and training, validation, and final-test
  namespaces are disjoint. A policy cannot memorize a layout, a vent mouth, or a
  camera position because it never sees the same facility twice.
- **Both learners exist** with strict checkpoint, resume, fingerprint, and
  manifest contracts, and each already accepts frozen opponents of the other
  side.
- **The reward ledgers, action masks, and observation contracts are audited.**
  The runner ledger sums to the episode return to within 1e-14; the security
  ledger is a shared subtotal plus an attributed per-operative term, and both
  halves are pinned by tests.

## Measured constraints on the current host

Every number below was measured on the training machine, not estimated.

| Quantity | Measurement |
|---|---|
| Environment step | 1.306 ms per decision, 766 decisions/s single-threaded |
| Policy forward, batch 8 | CPU 2.56 ms, GPU 2.30 ms |
| Policy forward, batch 1024 | CPU 40.66 ms, GPU 3.04 ms |
| Forward + backward, batch 256 | CPU 29.90 ms, GPU 8.24 ms |
| Forward + backward, batch 1024 | CPU 130.83 ms, GPU 8.53 ms |
| Hardware | 24 logical cores, 17.1 GB CUDA device |
| Aggregate throughput, 22 workers | 2,039 decisions/s, 12,234 simulation ticks/s |

Two conclusions follow directly.

**The environment is the throughput floor.** Stepping is Python and NumPy over a
60 Hz simulation, and no accelerator touches it. Collection scales only with
worker count, and the ceiling is roughly `766 x cores` before contention.

**The accelerator is worth exactly one thing: the update.** At collection batch
sizes it is a wash, 2.56 ms against 2.30 ms, and the host transfer can make it
worse. At an update minibatch of 1024 it is 130.83 ms against 8.53 ms, a factor
of 15.3. Put collection on cores and the update on the device; do not move
per-decision inference to the device expecting a gain.

## Horizon and credit assignment

The runner decides at 10 Hz. That gives three spans that should be compared
directly:

| Span | Decisions | Game time |
|---|---:|---:|
| Gradient reach through recurrent state, `rollout = 128` | 128 | 12.8 s |
| Value horizon, `gamma = 0.995` | 200 | 20 s |
| Observed episode length | 814 - 2250 | 81 - 225 s |

A short recurrent rollout limits the memory features that truncated
backpropagation can train. Direct reward propagation is separately controlled
by GAE through `gamma * lambda`, while the critic and potential-based progress
reward bridge consequences beyond one rollout. Treating the recurrent window
as the only credit horizon conflates those mechanisms.

The implemented baseline therefore changes the coupled set rather than only
one number:

- runner rollout `512`, `gamma = 0.999`, and `lambda = 0.98`;
- security rollout `192`, because contact, interception, and damage feedback
  arrive faster than runner mission completion;
- fixed reward scale `0.05` for both critics, with raw reward ledgers unchanged;
- the published runner value head is rescaled during transplant so it begins in
  the same units as the v2 critic.

Two interactions are worth planning around rather than discovering:

- **Raising `rollout` is what makes the accelerator pay.** It moves the update
  minibatch from 256 to 1024, where the device advantage goes from 3.6x to
  15.3x. The horizon fix and the hardware win are the same change.
- **Raising `gamma` raises return variance.** Expect noisier advantages and
  budget more samples per update, or the horizon gain is spent on variance.

`field_targets` holds 16 rows sorted by distance, so a device hacked two rooms
back can be evicted by nearer content. That eviction is the gap recurrent memory
has to bridge, and it is a fair reason to revisit capacity **after** the horizon
can carry a gradient far enough to train it.

## Concurrent training against a frozen opponent pool

The goal is one session in which both sides improve. The known failure mode of
naive simultaneous play is cycling: each side overfits the other's current
policy, gains evaporate when the opponent moves, and the metrics stay busy while
nothing improves. An opponent pool is the standard mitigation and preserves the
single session.

`ghostline co-train-v2` runs runner PPO and security MAPPO concurrently. Within
one generation, both opponents are immutable. After both processes finish,
only held-out-selected `best.pt` / `champion.pt` checkpoints enter the next
generation's uniform pool. The current generation can never see the other
current process's output. This is block-frozen league training: it retains
parallel wall-clock use without violating either learner's on-policy assumption
or chasing a policy that moves mid-rollout.

Generation zero transplants the immutable published-v1 runner and behavior
clones the tactical security controller. Later generations initialize from the
previous selected checkpoint and replay every admitted opponent plus the
scripted security fraction. Training and validation seed windows are disjoint
between generations. Every command, PID, opponent hash, metric tail, selected
checkpoint, and output hash is written to `session.json`.

Alternating frozen turns remain the reproducible fallback and the natural
control: if co-training does not beat it at equal environment steps, that is a
result worth having rather than a setback.

## Suggested instrumentation

The first campaign is the first time this distribution has been trained at all,
so the diagnostics matter as much as the result:

- Episode duration is the leading indicator of containment; stop rate is
  lagging and noisy at small validation counts.
- Report value loss and explained variance separately for updates that contain
  a terminal. Collapses on terminal batches are expected while returns are
  order 1 and no episode has extracted; they become meaningful once extraction
  is regular.
- Track the split between collection and update wall time. If the update is
  under a fifth of the total, further accelerator work is wasted and worker
  count is the lever.
- Log opponent identity, pool size, and the live fraction per update, so a
  regression can be traced to the opponent distribution rather than the policy.

## Verification

- Focused tests for changed areas plus the full unit suite.
- Fresh and resume smoke for the co-training loop, with the opponent pool state
  restored exactly.
- A stale-checkpoint rejection for any new fingerprinted artifact.
- Throughput measured on this host before and after, since worker count and
  device placement are both changing.
- Held-out evaluation from the disjoint namespaces; a co-trained result may not
  borrow acceptance from any earlier campaign.

The intended first campaign is 48 wall-clock hours over three 16-hour
generations. Interactive laptop runs use eight runner environments and four
security environments. Every trainer receives a one-thread numerical-library
cap, below-normal Windows priority, and an inherited 50% logical-CPU affinity
ceiling. The calibrated concurrent startup measured 23.1% mean, 35.9% p95, and
58.0% maximum total CPU while retaining about 794 runner decisions/s before
steady state. Monitoring uses an ordinary sleep/poll loop between one and sixty
seconds; no scheduled task is involved.

The algorithm choices follow PPO and GAE, the MAPPO parameter-sharing and CTDE
baseline, policy-invariant potential shaping, and frozen league practice:

- https://arxiv.org/abs/1707.06347
- https://arxiv.org/abs/1506.02438
- https://arxiv.org/abs/2103.01955
- https://ai.stanford.edu/~ang/papers/shaping-icml99.ps
- https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii/
