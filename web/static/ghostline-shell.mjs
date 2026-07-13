import { GhostlineEmbedBridge } from "./embed-bridge.mjs";
import { ghostlinePolicy } from "./policy-bridge.mjs";
import { matchedRunSnapshot } from "./matched-runs.mjs";

const embedBridge = new GhostlineEmbedBridge();
const commands = [];
const runHistory = { human: null, agent: null };
let gameReady = false;
let embedInteractive = false;
let lastStatus = "active";
let autoplayQueued = false;
let policyFailureQueued = false;
let policyAvailability = null;

const $ = (id) => document.getElementById(id);
const tier = () => Number($("tier-select")?.value || 1);
const seed = () => {
  const raw = $("seed-input")?.value?.trim();
  return raw ? Math.max(0, Math.min(2147483647, Number(raw) || 0)) : null;
};

function queue(type) {
  commands.push({ type, tier: tier(), seed: seed() });
}

function formatMetric(value, suffix = "") {
  return value === undefined || value === null ? "—" : `${value}${suffix}`;
}

function renderComparison() {
  const snapshot = matchedRunSnapshot(runHistory);
  const status = $("match-status");
  if (status) {
    status.dataset.state = snapshot.state;
    status.textContent = snapshot.message;
  }
  for (const mode of ["human", "agent"]) {
    const record = runHistory[mode];
    const root = $(`${mode}-metrics`);
    if (!root) continue;
    if (!record) {
      root.innerHTML = '<span class="empty-metric">No completed run yet</span>';
      continue;
    }
    if (!snapshot.matched && runHistory.human && runHistory.agent) {
      root.innerHTML = `
        <strong class="metric-result refused">NOT COMPARED</strong>
        <span>T${record.tier} / seed ${record.seed}</span>
        <span>Run the identical contract to unlock metrics.</span>`;
      continue;
    }
    root.innerHTML = `
      <strong class="metric-result ${record.status}">${record.status === "success" ? "CLEARED" : "FAILED"}</strong>
      <span>T${record.tier} / seed ${record.seed}</span>
      <span>${record.data}/${record.quota} data</span>
      <span>${record.time.toFixed(1)} s</span>
      <span>${record.trace.toFixed(0)}% trace</span>
      <span>${record.damage} damage</span>`;
  }
}

function setBootState(state, message = "") {
  const overlay = $("launch-gate");
  const title = $("launch-title");
  const copy = $("launch-copy");
  const button = $("focus-game");
  document.body.dataset.boot = state;
  if (!overlay) return;
  if (state === "ready") {
    gameReady = true;
    title.textContent = "THE LINE IS OPEN";
    copy.textContent = "Click to unlock audio and route keyboard input to the game.";
    button.hidden = false;
    overlay.hidden = false;
    maybeAutoplay();
  } else if (state === "engage") {
    // The browser runtime is now mounted and waiting only for the visitor's
    // required audio/user-activation click.  Treat this as embed readiness so
    // a parent does not time out while the game correctly waits at its own
    // focus gate.  ``gameReady`` remains false until Python starts the game.
    embedInteractive = true;
    title.textContent = "AUTHORIZE AUDIO";
    copy.textContent = message || "One click lets the browser start audio and the secure simulation.";
    button.textContent = "INITIALIZE GHOSTLINE";
    button.hidden = false;
    overlay.hidden = false;
    maybePublishEmbedReady();
  } else if (state === "running") {
    overlay.hidden = true;
    $("canvas")?.focus();
  } else {
    title.textContent = "CONNECTING TO GHOSTLINE";
    copy.textContent = message || "Preparing the secure facility simulation…";
    button.hidden = true;
    overlay.hidden = false;
  }
}

function showNotice(message, kind = "info") {
  const notice = $("notice");
  if (!notice) return;
  notice.textContent = message;
  notice.dataset.kind = kind;
  notice.hidden = false;
  clearTimeout(showNotice.timeout);
  showNotice.timeout = setTimeout(() => { notice.hidden = true; }, 4800);
}

function setPolicyState(state, message) {
  const chip = $("policy-chip");
  if (chip) {
    chip.dataset.state = state;
    chip.textContent = message || state.toUpperCase();
  }
  const agentButton = $("agent-control");
  if (agentButton) agentButton.ariaBusy = state === "loading" ? "true" : "false";
}

function setControlMode(mode) {
  document.body.dataset.control = mode;
  const chip = $("control-chip");
  if (chip) chip.textContent = mode === "agent" ? "AGENT CONTROL" : "HUMAN CONTROL";
  const takeover = $("agent-control");
  const manual = $("human-control");
  if (takeover) takeover.hidden = mode === "agent";
  if (manual) manual.hidden = mode !== "agent";
}

function updateMetrics(serialized) {
  let metrics;
  try {
    metrics = typeof serialized === "string" ? JSON.parse(serialized) : serialized;
  } catch {
    return;
  }
  $("live-tier").textContent = `T${metrics.tier}`;
  $("live-seed").textContent = formatMetric(metrics.seed);
  $("live-data").textContent = `${metrics.data}/${metrics.quota}`;
  $("live-time").textContent = `${Number(metrics.time).toFixed(1)}s`;
  $("live-trace").textContent = `${Number(metrics.trace).toFixed(0)}%`;
  $("live-damage").textContent = formatMetric(metrics.damage);
  if (lastStatus === "active" && metrics.status !== "active") {
    embedBridge.publishRunComplete(metrics);
    // Pin the completed contract into the launcher so the other controller's
    // next run is matched by default instead of silently drawing a new seed.
    if ($("tier-select")) $("tier-select").value = String(metrics.tier);
    if ($("seed-input")) $("seed-input").value = String(metrics.seed);
    if (Object.hasOwn(runHistory, metrics.mode)) {
      runHistory[metrics.mode] = metrics;
      renderComparison();
    } else if (metrics.mode === "hybrid") {
      showNotice("Mixed-control run complete. It is excluded from human-versus-agent benchmark cards.", "info");
    }
  }
  lastStatus = metrics.status;
}

function maybeAutoplay() {
  if (autoplayQueued || !gameReady) return;
  const query = new URLSearchParams(location.search);
  if (query.get("autoplay") !== "1") return;
  autoplayQueued = true;
  setBootState("running");
  setTimeout(() => queue("agent"), 1400);
}

function maybePublishEmbedReady() {
  if ((!gameReady && !embedInteractive) || policyAvailability === null) return;
  embedBridge.markReady(policyAvailability);
}

async function toggleFullscreen() {
  const frame = $("game-frame");
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await frame.requestFullscreen({ navigationUI: "hide" });
    $("canvas")?.focus();
  } catch (error) {
    showNotice(`Fullscreen is unavailable: ${error.message}`, "error");
  }
}

globalThis.ghostlinePolicy = ghostlinePolicy;
globalThis.ghostlineShell = {
  consumeCommand: () => commands.length ? JSON.stringify(commands.shift()) : null,
  markGameReady: () => {
    setBootState("ready");
    maybePublishEmbedReady();
  },
  setBootState,
  setPolicyState,
  setControlMode,
  showNotice,
  updateMetrics,
};

$("play-selected")?.addEventListener("click", () => queue("launch-human"));
$("agent-control")?.addEventListener("click", () => queue("agent"));
$("human-control")?.addEventListener("click", () => queue("human"));
$("fullscreen-control")?.addEventListener("click", toggleFullscreen);
$("focus-control")?.addEventListener("click", () => $("canvas")?.focus());
$("focus-game")?.addEventListener("click", () => {
  if (gameReady) setBootState("running");
  else setBootState("booting", "Loading the facility and browser runtime…");
  queue("focus");
});
$("game-frame")?.addEventListener("pointerdown", () => $("canvas")?.focus());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) queue("pause-hidden");
});
globalThis.addEventListener("blur", () => queue("pause-focus"));

globalThis.addEventListener("ghostline:policy-manifest", (event) => {
  policyAvailability = Boolean(event.detail.available);
  embedBridge.setModelAvailable(policyAvailability);
  if (event.detail.available) {
    setPolicyState("available", "AGENT READY TO LOAD");
    maybeAutoplay();
  } else {
    setPolicyState("unavailable", "HUMAN-ONLY BUILD");
  }
  maybePublishEmbedReady();
});
globalThis.addEventListener("ghostline:policy-state", (event) => {
  const { state, progress = 0, backend, error } = event.detail;
  const percentage = Math.round(progress * 100);
  if (state === "loading") setPolicyState(state, `LOADING AGENT ${percentage}%`);
  else if (state === "ready") {
    policyFailureQueued = false;
    setPolicyState(state, `AGENT ONLINE // ${String(backend).toUpperCase()}`);
  } else if (state === "unavailable") {
    setPolicyState(state, `AGENT UNAVAILABLE${error ? " // RETRY" : ""}`);
    if (document.body.dataset.control === "agent" && !policyFailureQueued) {
      policyFailureQueued = true;
      queue("policy-failed");
    }
  }
});
globalThis.addEventListener("ghostline:policy-inference", (event) => {
  const latency = Number(event.detail.average_latency_ms || 0);
  $("policy-latency").textContent = `${latency.toFixed(1)} ms`;
  $("policy-backend").textContent = String(event.detail.backend || "—").toUpperCase();
});

renderComparison();
setControlMode("human");
setBootState("booting");
void ghostlinePolicy.probe();
