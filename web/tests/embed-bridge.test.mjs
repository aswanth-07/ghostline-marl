import assert from "node:assert/strict";
import test from "node:test";

import {
  GhostlineEmbedBridge,
  resolveParentOrigin,
  runCompleteMessage,
} from "../static/embed-bridge.mjs";


function framedBrowser({
  search = "?embed=1&autoplay=0",
  parentOrigin = "https://portfolio.example",
  referrer = "https://portfolio.example/projects/ghostline",
} = {}) {
  const posts = [];
  const parent = {
    postMessage: (message, targetOrigin) => posts.push({ message, targetOrigin }),
  };
  const windowObject = {
    location: { search, ancestorOrigins: [parentOrigin] },
    parent,
  };
  const documentObject = {
    referrer,
    documentElement: { dataset: {} },
    body: { dataset: {} },
  };
  return { documentObject, posts, windowObject };
}


test("embed mode publishes one origin-scoped ready event with policy availability", () => {
  const browser = framedBrowser();
  const bridge = new GhostlineEmbedBridge(browser);

  assert.equal(bridge.presentationMode, true);
  assert.equal(browser.documentObject.documentElement.dataset.embed, "true");
  assert.equal(browser.documentObject.body.dataset.embed, "true");
  bridge.setModelAvailable(true);
  assert.equal(bridge.markReady(), true);
  assert.equal(bridge.markReady(), false, "ready must be emitted once per page load");
  assert.deepEqual(browser.posts, [{
    targetOrigin: "https://portfolio.example",
    message: {
      source: "ghostline",
      version: 1,
      type: "ready",
      modelAvailable: true,
    },
  }]);
});


test("human-only builds report modelAvailable false without disabling play", () => {
  const browser = framedBrowser();
  const bridge = new GhostlineEmbedBridge(browser);

  assert.equal(bridge.markReady(false), true);
  assert.equal(browser.posts[0].message.modelAvailable, false);
});


test("completed contracts emit the stable portfolio telemetry contract", () => {
  const browser = framedBrowser();
  const bridge = new GhostlineEmbedBridge(browser);
  const metrics = {
    mode: "agent",
    status: "success",
    tier: 6,
    seed: 2_000_071,
    time: 41.25,
  };

  assert.equal(bridge.publishRunComplete(metrics), true);
  assert.deepEqual(browser.posts[0], {
    targetOrigin: "https://portfolio.example",
    message: {
      source: "ghostline",
      version: 1,
      type: "run-complete",
      controller: "agent",
      tier: 6,
      seed: 2_000_071,
      success: true,
      duration: 41.25,
    },
  });
});


test("bridge refuses malformed telemetry and untrusted or non-embedded parents", () => {
  assert.equal(runCompleteMessage({ mode: "agent", status: "success", tier: 7, seed: 1, time: 1 }), null);
  assert.equal(runCompleteMessage({ mode: "unknown", status: "success", tier: 1, seed: 1, time: 1 }), null);

  const mismatch = framedBrowser({ referrer: "https://unexpected.example/project" });
  assert.equal(resolveParentOrigin(mismatch.windowObject, mismatch.documentObject), null);
  assert.equal(new GhostlineEmbedBridge(mismatch).markReady(true), false);
  assert.equal(mismatch.posts.length, 0);

  const standalone = framedBrowser({ search: "?autoplay=0" });
  assert.equal(new GhostlineEmbedBridge(standalone).markReady(true), false);
  assert.equal(standalone.posts.length, 0);
});
