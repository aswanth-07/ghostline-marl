# Neural benchmark evidence

The selected Ghostline checkpoint is the 384-unit recurrent policy with
SHA-256
`76baa30af55cdaa2e71bb6ba06672bd9203455552358017505685827240b2e47`.
It is bound to `GhostlineEnv-v2` fingerprint
`521c449a8bd9a540977a918f5b094dd3aeff44cc579a55f75e22a74bab20e129`.

## Release result

The one-time 8M final test contains 500 never-before-opened seeds per tier. The
policy passed at `99.8/100.0/96.4/98.0/99.0/89.6%` for tiers 1-6. The
[JSON report](champion-final-8m-500.json) contains all 3,000 episode records,
exact seeds, action hashes, reward components, Wilson intervals, failures,
damage, detections, time, path efficiency, optional data, and latency. The
[aggregate CSV](champion-final-8m-500.csv) and
[episode CSV](champion-final-8m-500.episodes.csv) are sibling artifacts.
`benchmarks/final-test-slices.json` permanently marks this slice consumed and
binds hashes for all three outputs.

Before final evaluation, the same checkpoint passed two disjoint 100-seed-per-
tier confirmation windows at `99/99/96/99/99/85%` and
`100/100/96/99/99/86%`.

## Training lineage

The current base corpus contains 1,800 successful teacher episodes and 412,483
transitions. Four recovery collections add 2,100 policy-induced episodes and
529,401 transitions, for 3,900 episodes and 941,884 transitions overall. The
per-round recovery counts are 90,042, 146,817, 152,082, and 140,460
transitions. A historical checkpoint supplied declared initialization weights;
all labels, policy-induced recovery data, selection windows, and optimizer
state are current-fingerprint.

The final model received a 5,000-update initial clone, 6,000 DAgger updates,
and a 2,000-update low-rate consolidation pass. Intermediate reports are kept
as selection diagnostics; failed checkpoints remain labelled and were not used
to open the final slice.

The historical pure-PPO and conservative PPO pilots are retained as negative
evidence. Neither is hidden inside the selected checkpoint or presented as a
current improvement.

## Deployment parity

The canonical FP32 ONNX policy matched PyTorch on all 1,000 sampled recurrent
transitions. Dynamic INT8 reduced size by 28.3% but changed five actions, so it
was rejected and the verified 5.83 MB FP32 graph became the deployment model.

These results do not support a “superhuman” claim. A matched-seed cohort of at
least five unassisted players is still required before comparing the policy to
people.
