import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";


const root = path.resolve(process.argv[2] || ".web-build/ghostline/build/web");
const manifest = JSON.parse(fs.readFileSync(path.join(root, "policy-manifest.json"), "utf8"));
if (!manifest.available || !manifest.model_url) throw new Error("Build has no policy; pass --model --strict-model first");

const runtimeRoot = path.join(root, "vendor", "onnxruntime-web-1.27.0");
const module = await import(pathToFileURL(path.join(runtimeRoot, "ort.all.min.mjs")).href);
const ort = module.default;
ort.env.wasm.wasmPaths = pathToFileURL(`${runtimeRoot}${path.sep}`).href;
ort.env.wasm.numThreads = 1;

const model = fs.readFileSync(path.join(root, manifest.model_url));
const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
const masks = new Set(["target_mask", "entity_mask", "action_mask"]);

function inputShape(name) {
  const metadata = Array.isArray(session.inputMetadata)
    ? session.inputMetadata[session.inputNames.indexOf(name)]
    : session.inputMetadata[name];
  const fallback = manifest.inputs[name];
  return (metadata?.shape || metadata?.dimensions || fallback).map((value, index) => {
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback[index];
  });
}

function zeroInput(name) {
  const shape = inputShape(name);
  const length = shape.reduce((left, right) => left * right, 1);
  if (masks.has(name)) return new ort.Tensor("int8", new Int8Array(length).fill(1), shape);
  return new ort.Tensor("float32", new Float32Array(length), shape);
}

const feeds = Object.fromEntries(session.inputNames.map((name) => [name, zeroInput(name)]));
const start = performance.now();
const first = await session.run(feeds);
const firstLatency = performance.now() - start;
assert.deepEqual(first.logits.dims, [1, 36]);
assert.ok(first.next_hidden, "recurrent export must return next_hidden");

feeds.hidden = first.next_hidden;
const second = await session.run(feeds);
assert.deepEqual(second.logits.dims, [1, 36]);
assert.deepEqual(second.next_hidden.dims, first.next_hidden.dims);

const warmLatencies = [];
feeds.hidden = second.next_hidden;
for (let index = 0; index < 20; index += 1) {
  const warmStart = performance.now();
  const output = await session.run(feeds);
  warmLatencies.push(performance.now() - warmStart);
  feeds.hidden = output.next_hidden;
}
warmLatencies.sort((left, right) => left - right);
const median = warmLatencies[Math.floor(warmLatencies.length / 2)];
const p95 = warmLatencies[Math.floor(warmLatencies.length * 0.95)];

console.log(JSON.stringify({
  backend: "wasm",
  inputs: session.inputNames,
  outputs: Object.keys(first),
  hidden_shape: first.next_hidden.dims,
  cold_inference_ms: Number(firstLatency.toFixed(3)),
  warm_median_ms: Number(median.toFixed(3)),
  warm_p95_ms: Number(p95.toFixed(3)),
}));
