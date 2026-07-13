import assert from "node:assert/strict";
import test from "node:test";

import { matchedRunSnapshot } from "../static/matched-runs.mjs";


const run = (controller, tier, seed) => ({
  mode: controller,
  tier,
  seed,
  status: "success",
  data: 2,
  quota: 2,
  time: 18.2,
  trace: 7,
  damage: 0,
});


test("matched-run snapshot accepts only identical tier and seed", () => {
  const snapshot = matchedRunSnapshot({
    human: run("human", 4, 81234),
    agent: run("agent", 4, 81234),
  });
  assert.equal(snapshot.state, "matched");
  assert.equal(snapshot.matched, true);
  assert.match(snapshot.message, /T4 \/ seed 81234/);
});


test("matched-run snapshot clearly refuses different contracts", () => {
  const differentSeed = matchedRunSnapshot({
    human: run("human", 4, 81234),
    agent: run("agent", 4, 81235),
  });
  assert.equal(differentSeed.state, "refused");
  assert.equal(differentSeed.matched, false);
  assert.match(differentSeed.message, /Comparison refused/);

  const differentTier = matchedRunSnapshot({
    human: run("human", 3, 81234),
    agent: run("agent", 4, 81234),
  });
  assert.equal(differentTier.state, "refused");
  assert.equal(differentTier.matched, false);
});
