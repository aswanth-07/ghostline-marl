const FLOAT_INPUTS = new Set(["ego", "objective", "local_grid", "targets", "entities", "rays", "hidden"]);
const MASK_INPUTS = new Set(["target_mask", "entity_mask", "action_mask"]);

export function requestedExecutionProvider(search = globalThis.location?.search ?? "") {
  return new URLSearchParams(search).get("backend") === "webgpu" ? "webgpu" : "wasm";
}

function flatten(values, output = []) {
  if (Array.isArray(values)) {
    for (const value of values) flatten(value, output);
  } else {
    output.push(Number(values));
  }
  return output;
}

function metadataShape(session, inputName, fallback) {
  const metadata = session.inputMetadata;
  let record = null;
  if (Array.isArray(metadata)) {
    const index = session.inputNames.indexOf(inputName);
    record = index >= 0 ? metadata[index] : null;
  } else if (metadata) {
    record = metadata[inputName];
  }
  const shape = record?.shape ?? record?.dimensions ?? record?.dims;
  if (!Array.isArray(fallback) || fallback.length === 0) {
    throw new Error(`Policy manifest has no static shape for ${inputName}`);
  }
  if (!Array.isArray(shape)) return fallback;
  return shape.map((dimension, index) => {
    const numeric = Number(dimension);
    return Number.isFinite(numeric) && numeric > 0 ? numeric : fallback[index] ?? 1;
  });
}

async function fetchBinary(url, progress) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Model request failed (${response.status})`);
  const expected = Number(response.headers.get("content-length")) || 0;
  if (!response.body?.getReader) return response.arrayBuffer();

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    progress(expected ? received / expected : 0, received, expected);
  }
  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return merged.buffer;
}

export class GhostlinePolicyBridge {
  constructor({ manifestUrl = "./policy-manifest.json", runtimeUrl = "./vendor/onnxruntime-web-1.27.0/ort.all.min.mjs" } = {}) {
    this.manifestUrl = manifestUrl;
    this.runtimeUrl = runtimeUrl;
    this.manifest = null;
    this.ort = null;
    this.session = null;
    this.hidden = null;
    this.backend = "none";
    this.state = "idle";
    this.error = null;
    this.lastAction = 0;
    this.lastLatencyMs = 0;
    this.averageLatencyMs = 0;
    this.inferenceCount = 0;
    this.pendingObservation = null;
    this.busy = false;
    this.loadPromise = null;
  }

  async probe() {
    try {
      const response = await fetch(this.manifestUrl, { cache: "no-cache" });
      if (!response.ok) throw new Error(`Policy manifest request failed (${response.status})`);
      this.manifest = await response.json();
      this._emit("manifest", { available: Boolean(this.manifest.available), manifest: this.manifest });
      return Boolean(this.manifest.available);
    } catch (error) {
      this.error = String(error?.message ?? error);
      this._emit("manifest", { available: false, error: this.error });
      return false;
    }
  }

  load() {
    if (this.session) return Promise.resolve(true);
    if (this.loadPromise) return this.loadPromise;
    this.loadPromise = this._load().finally(() => {
      this.loadPromise = null;
    });
    return this.loadPromise;
  }

  async _load() {
    this.state = "loading";
    this.error = null;
    this._emit("state", { state: this.state, progress: 0 });
    try {
      if (!this.manifest && !(await this.probe())) throw new Error("No release policy is bundled yet");
      if (!this.manifest?.available || !this.manifest.model_url) throw new Error("No release policy is bundled yet");

      const imported = await import(this.runtimeUrl);
      this.ort = imported.default ?? imported;
      const runtimeRoot = new URL("./vendor/onnxruntime-web-1.27.0/", document.baseURI).href;
      this.ort.env.wasm.wasmPaths = runtimeRoot;
      this.ort.env.wasm.numThreads = globalThis.crossOriginIsolated
        ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2))
        : 1;
      this.ort.env.wasm.proxy = false;

      const modelUrl = new URL(this.manifest.model_url, document.baseURI).href;
      const model = await fetchBinary(modelUrl, (fraction, received, expected) => {
        this._emit("state", { state: "loading", progress: fraction * 0.78, received, expected });
      });
      this._emit("state", { state: "loading", progress: 0.82 });

      const requestedProvider = requestedExecutionProvider();
      const webgpuAvailable = Boolean(navigator.gpu);
      if (requestedProvider === "webgpu" && webgpuAvailable) {
        try {
          this.session = await this.ort.InferenceSession.create(model, {
            executionProviders: ["webgpu", "wasm"],
            graphOptimizationLevel: "all",
          });
          this.backend = "webgpu";
        } catch (webgpuError) {
          console.warn("Ghostline WebGPU comparison path failed; using WASM.", webgpuError);
        }
      }
      if (!this.session) {
        this.session = await this.ort.InferenceSession.create(model, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });
        this.backend = "wasm";
      }

      const fallback = this.manifest.inputs?.hidden;
      const hiddenShape = metadataShape(this.session, "hidden", fallback);
      if (hiddenShape.at(-1) !== Number(this.manifest.hidden_size)) {
        throw new Error("Policy manifest recurrent width does not match ONNX session metadata");
      }
      this.hidden = new this.ort.Tensor("float32", new Float32Array(hiddenShape.reduce((a, b) => a * b, 1)), hiddenShape);
      this.state = "ready";
      this._emit("state", { state: this.state, progress: 1, backend: this.backend });
      return true;
    } catch (error) {
      this._failClosed(error);
      console.error("Ghostline policy unavailable:", error);
      return false;
    }
  }

  reset() {
    this.pendingObservation = null;
    this.lastAction = 0;
    if (this.session && this.ort) {
      const fallback = this.manifest?.inputs?.hidden;
      const shape = metadataShape(this.session, "hidden", fallback);
      this.hidden = new this.ort.Tensor("float32", new Float32Array(shape.reduce((a, b) => a * b, 1)), shape);
    }
  }

  step(serializedObservation) {
    if (!this.session || this.state !== "ready") {
      this.lastAction = 0;
      return 0;
    }
    try {
      this.pendingObservation = typeof serializedObservation === "string"
        ? JSON.parse(serializedObservation)
        : serializedObservation;
      if (!this.busy) void this._drain();
    } catch (error) {
      this._failClosed(error);
      console.error("Ghostline rejected a malformed policy observation.", error);
    }
    return this.currentAction();
  }

  currentAction() {
    // While a new recurrent decision is unresolved, neutral is safer than
    // replaying an action computed for an older observation.
    return this.session && this.state === "ready" && !this.busy && !this.pendingObservation
      ? this.lastAction
      : 0;
  }

  hasCompletedAction(afterInferenceCount = -1) {
    return Boolean(
      this.session &&
      this.state === "ready" &&
      !this.busy &&
      !this.pendingObservation &&
      this.inferenceCount > Number(afterInferenceCount)
    );
  }

  async _drain() {
    this.busy = true;
    try {
      while (this.pendingObservation && this.session) {
        const observation = this.pendingObservation;
        this.pendingObservation = null;
        await this._infer(observation);
      }
    } catch (error) {
      this._failClosed(error);
      console.error("Ghostline policy inference failed:", error);
    } finally {
      this.busy = false;
    }
  }

  _tensor(name, values) {
    const fallback = this.manifest.inputs[name];
    const shape = metadataShape(this.session, name, fallback);
    const flat = flatten(values);
    const expected = shape.reduce((a, b) => a * b, 1);
    if (flat.length !== expected) throw new Error(`${name} has ${flat.length} values; expected ${expected}`);
    if (MASK_INPUTS.has(name)) return new this.ort.Tensor("int8", Int8Array.from(flat), shape);
    if (FLOAT_INPUTS.has(name)) return new this.ort.Tensor("float32", Float32Array.from(flat), shape);
    throw new Error(`Unsupported policy input: ${name}`);
  }

  async _infer(observation) {
    const feeds = {};
    for (const inputName of this.session.inputNames) {
      if (inputName === "hidden") feeds.hidden = this.hidden;
      else feeds[inputName] = this._tensor(inputName, observation[inputName]);
    }
    const start = performance.now();
    const results = await this.session.run(feeds);
    this.lastLatencyMs = performance.now() - start;
    this.inferenceCount += 1;
    this.averageLatencyMs += (this.lastLatencyMs - this.averageLatencyMs) / Math.min(this.inferenceCount, 100);
    if (results.next_hidden) this.hidden = results.next_hidden;

    const logits = results.logits?.data;
    if (!logits) throw new Error("Policy output does not contain logits");
    const legal = flatten(observation.action_mask);
    let best = 0;
    let score = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < logits.length; index += 1) {
      if (legal[index] > 0 && Number(logits[index]) > score) {
        best = index;
        score = Number(logits[index]);
      }
    }
    this.lastAction = best;
    this._emit("inference", {
      action: best,
      latency_ms: this.lastLatencyMs,
      average_latency_ms: this.averageLatencyMs,
      backend: this.backend,
    });
  }

  _failClosed(error) {
    // A failed recurrent step invalidates both memory and the previous action.
    // Returning action zero until Python restores human control guarantees that
    // no stale movement/dash/pulse command is replayed after the failure.
    this.pendingObservation = null;
    this.session = null;
    this.hidden = null;
    this.backend = "none";
    this.state = "unavailable";
    this.lastAction = 0;
    this.error = String(error?.message ?? error);
    this._emit("state", { state: this.state, error: this.error });
  }

  _emit(kind, detail) {
    globalThis.dispatchEvent(new CustomEvent(`ghostline:policy-${kind}`, { detail }));
  }
}

export const ghostlinePolicy = new GhostlinePolicyBridge();
