import assert from "node:assert/strict";
import test from "node:test";

import { GhostlinePolicyBridge } from "../static/policy-bridge.mjs";


class Tensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}


globalThis.CustomEvent = class {
  constructor(type, options) {
    this.type = type;
    this.detail = options.detail;
  }
};
globalThis.dispatchEvent = () => true;


test("masked action selection and recurrent state persist", async () => {
  const bridge = new GhostlinePolicyBridge();
  bridge.manifest = {
    hidden_size: 4,
    inputs: {
      ego: [1, 2],
      action_mask: [1, 4],
      hidden: [1, 1, 4],
    },
  };
  bridge.ort = { Tensor };
  bridge.session = {
    inputNames: ["ego", "action_mask", "hidden"],
    inputMetadata: {
      ego: { shape: [1, 2] },
      action_mask: { shape: [1, 4] },
      hidden: { shape: [1, 1, 4] },
    },
    run: async (feeds) => {
      assert.equal(feeds.action_mask.type, "int8");
      assert.deepEqual(Array.from(feeds.hidden.data), [0, 0, 0, 0]);
      return {
        logits: new Tensor("float32", Float32Array.from([1, 99, 3, 2]), [1, 4]),
        next_hidden: new Tensor("float32", Float32Array.from([1, 2, 3, 4]), [1, 1, 4]),
      };
    },
  };
  bridge.hidden = new Tensor("float32", new Float32Array(4), [1, 1, 4]);
  bridge.state = "ready";

  assert.equal(bridge.step(JSON.stringify({ ego: [0, 0], action_mask: [1, 0, 1, 1] })), 0);
  while (bridge.busy) await new Promise((resolve) => setTimeout(resolve, 1));
  assert.equal(bridge.lastAction, 2, "illegal logit 1 must be ignored");
  assert.equal(bridge.currentAction(), 2);
  assert.deepEqual(Array.from(bridge.hidden.data), [1, 2, 3, 4]);
});


test("reset clears queued action and recurrent memory", () => {
  const bridge = new GhostlinePolicyBridge();
  bridge.manifest = { hidden_size: 3, inputs: { hidden: [1, 1, 3] } };
  bridge.ort = { Tensor };
  bridge.session = { inputNames: [], inputMetadata: { hidden: { shape: [1, 1, 3] } } };
  bridge.lastAction = 17;
  bridge.pendingObservation = { stale: true };
  bridge.reset();
  assert.equal(bridge.lastAction, 0);
  assert.equal(bridge.pendingObservation, null);
  assert.deepEqual(Array.from(bridge.hidden.data), [0, 0, 0]);
});


test("inference failure clears stale action and recurrent state", async () => {
  const emitted = [];
  globalThis.dispatchEvent = (event) => {
    emitted.push(event);
    return true;
  };
  const bridge = new GhostlinePolicyBridge();
  bridge.manifest = {
    hidden_size: 2,
    inputs: { ego: [1, 2], action_mask: [1, 4], hidden: [1, 1, 2] },
  };
  bridge.ort = { Tensor };
  bridge.session = {
    inputNames: ["ego", "action_mask", "hidden"],
    inputMetadata: {
      ego: { shape: [1, 2] },
      action_mask: { shape: [1, 4] },
      hidden: { shape: [1, 1, 2] },
    },
    run: async () => { throw new Error("synthetic backend failure"); },
  };
  bridge.hidden = new Tensor("float32", new Float32Array(2), [1, 1, 2]);
  bridge.state = "ready";
  bridge.lastAction = 23;

  const originalConsoleError = console.error;
  console.error = () => {};
  try {
    const immediate = bridge.step(JSON.stringify({ ego: [0, 0], action_mask: [1, 1, 1, 1] }));
    assert.equal(immediate, 0, "an unresolved decision must not replay action 23");
    while (bridge.busy) await new Promise((resolve) => setTimeout(resolve, 1));
  } finally {
    console.error = originalConsoleError;
  }

  assert.equal(bridge.state, "unavailable");
  assert.equal(bridge.lastAction, 0);
  assert.equal(bridge.currentAction(), 0);
  assert.equal(bridge.session, null);
  assert.equal(bridge.hidden, null);
  assert.equal(bridge.pendingObservation, null);
  assert.equal(bridge.step("{}"), 0, "failed bridge must not replay action 23");
  assert.ok(emitted.some((event) => event.type === "ghostline:policy-state"));
});
