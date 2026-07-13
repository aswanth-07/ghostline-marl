# Neural benchmark evidence

The selected Ghostline checkpoint is the 384-unit recurrent policy with SHA-256
`458aa28d14b0829481a56c96dcc97a9ab9af2c463c15beef94d4c3e86ab59deb`.
It uses the frozen `GhostlineEnv-v2` fingerprint
`17d8617fd92015dc5a00b5314558fc7c0ff957685b12966efa5253806463739b`.

## Release result

The one-time 7M final test contains 500 never-before-opened seeds per tier. The
policy passed at `98.0/98.2/99.2/97.0/95.8/94.8%` for tiers 1-6. The JSON report
contains all 3,000 episode records, exact seeds, action-sequence hashes, reward
components, Wilson intervals, failure reasons, damage, detections, time, path
efficiency, optional-data rate, and inference latency. The aggregate and episode
CSV files are sibling artifacts. `benchmarks/final-test-slices.json` permanently
marks this slice consumed and binds hashes for all three outputs.

Before final evaluation, the same frozen checkpoint passed two disjoint
confirmation windows of 200 seeds per tier at
`99.5/98.5/98.5/97.5/97.0/91.5%` and
`98.0/100.0/99.0/97.5/96.5/94.5%`.

## Training lineage

The base corpus contains 1,800 successful teacher episodes and 467,853
transitions. Four recovery collections add 1,200 policy-induced episodes and
406,932 transitions, for 3,000 episodes in the complete lineage. The exact
per-round transition counts are 94,922, 95,297, 118,426, and 98,287; together
with the base corpus this is 874,785 transitions. The `training-lineage.csv`
file records every closed-loop selection gate and its checkpoint hash.

GRU-256 and GRU-384 behavior-cloning candidates used the same 10,000-update
budget and the exact same 600 validation contracts; GRU-384 won worst-tier
success 41% to 35%. Later DAgger points are sequential improvements measured on
disjoint validation windows, not equal-budget ablations.

The conservative PPO pilot was rejected. Its first six-tier gate reached only
`84/94/98/92/90/86%`, while the immutable DAgger rollback scored
`96/96/90/98/92/94%` on those exact seeds. `ppo-pilot-rejected.json` preserves
the failure analysis. No PPO result is hidden inside the selected checkpoint.

## Deployment parity

The canonical FP32 ONNX policy matched PyTorch on all 1,000 sampled recurrent
transitions. Dynamic INT8 reduced size by 28.3% but changed three actions, so it
was rejected and the verified 5.83 MB FP32 graph became the deployment model.

These results do not support a “superhuman” claim. A matched-seed cohort of at
least five unassisted players is still required before comparing the policy to
people.
