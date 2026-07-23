import { GhostlineEmbedBridge } from "./embed-bridge.mjs";
import { ghostlinePolicy } from "./policy-bridge.mjs";
import { matchedRunSnapshot } from "./matched-runs.mjs";

const embedBridge = new GhostlineEmbedBridge();
const commands = [];
const runHistory = { human: null, agent: null };
const agentShowcaseSeeds = {
  1: 1007004,
  2: 1015005,
  3: 1023012,
  4: 1031013,
  5: 1039028,
  6: 1047023,
};
const portfolioDemo = { tier: 6, seed: 2000000 };
let gameReady = false;
let lastStatus = "active";
let currentMetrics = null;
let autoplayQueued = false;
let agentActivationPending = false;
let policyFailureQueued = false;
let policyAvailability = null;

const $ = (id) => document.getElementById(id);
const tier = () => Number($("tier-select")?.value || 1);
const contractMode = () => $("mode-select")?.value === "adaptive" ? "adaptive" : "classic";
const directive = () => $("directive-select")?.value || "standard";
const seed = () => {
  const raw = $("seed-input")?.value?.trim();
  return raw ? Math.max(0, Math.min(2147483647, Number(raw) || 0)) : null;
};

function queue(type, extra = {}) {
  commands.push({ type, tier: tier(), seed: seed(), contractMode: contractMode(), directive: directive(), ...extra });
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
    button.textContent = "ENTER FACILITY";
    button.hidden = false;
    overlay.hidden = false;
    maybeAutoplay();
  } else if (state === "engage") {
    title.textContent = "AUTHORIZE AUDIO";
    copy.textContent = message || "One click lets the browser start audio and the secure simulation.";
    button.textContent = "INITIALIZE GHOSTLINE";
    button.hidden = false;
    overlay.hidden = false;
  } else if (state === "running") {
    overlay.hidden = true;
    $("canvas")?.focus();
  } else {
    title.innerHTML = "PLAY THE CONTRACT.<br><em>OR WATCH THE POLICY.</em>";
    copy.textContent = message || "Preparing the deterministic game and recurrent policy runtime.";
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

function usesCompactLabLayout() {
  return Boolean(
    globalThis.matchMedia?.("(max-width: 980px), (pointer: coarse) and (max-height: 600px)").matches
  );
}

function setIntelPanel(open, { focus = false, closeTarget = "trigger" } = {}) {
  const active = Boolean(open);
  const compact = usesCompactLabLayout();
  const panel = $("agent-lab-panel");
  document.body.classList.toggle("mobile-lab-open", compact && active);
  document.body.classList.toggle("lab-panel-collapsed", !compact && !active);
  for (const element of document.querySelectorAll(".site-header, .game-column, .site-legal")) {
    element.inert = compact && active;
  }
  if (panel) {
    if (compact && active) {
      panel.setAttribute("role", "dialog");
      panel.setAttribute("aria-modal", "true");
    } else {
      panel.removeAttribute("role");
      panel.removeAttribute("aria-modal");
    }
  }
  const trigger = $("intel-panel-control");
  if (trigger) {
    trigger.setAttribute("aria-expanded", String(active));
    trigger.textContent = compact ? "RUN SETUP" : active ? "HIDE INTEL" : "SHOW INTEL";
  }
  if (active && focus) $("tier-select")?.focus();
  else if (!active && focus) {
    (closeTarget === "canvas" ? $("canvas") : trigger)?.focus();
  }
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
  if (chip) {
    chip.textContent = mode === "agent"
      ? "AGENT CONTROL"
      : mode === "handoff"
        ? "AGENT HANDOFF"
        : "HUMAN CONTROL";
  }
  const takeover = $("agent-control");
  const manual = $("human-control");
  if (takeover) takeover.hidden = mode !== "human";
  if (manual) {
    manual.hidden = mode === "human";
    manual.textContent = mode === "handoff" ? "CANCEL HANDOFF" : "TAKE CONTROL";
  }
}

function updateMetrics(serialized) {
  let metrics;
  try {
    metrics = typeof serialized === "string" ? JSON.parse(serialized) : serialized;
  } catch {
    return;
  }
  currentMetrics = metrics;
  $("live-tier").textContent = `T${metrics.tier}`;
  $("live-seed").textContent = formatMetric(metrics.seed);
  $("live-data").textContent = `${metrics.data}/${metrics.quota}`;
  $("live-time").textContent = `${Number(metrics.time).toFixed(1)}s`;
  $("live-trace").textContent = `${Number(metrics.trace).toFixed(0)}%`;
  $("live-damage").textContent = formatMetric(metrics.damage);
  $("live-contract").textContent = metrics.contract === "GhostlineEnv-v3"
    ? `ADAPTIVE // ${String(metrics.directive || "standard").toUpperCase()}`
    : "CLASSIC // ENV-v2";
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
  if (seed() === null && $("seed-input")) {
    $("seed-input").value = String(agentShowcaseSeeds[tier()]);
  }
  setBootState("running");
  setTimeout(() => { void requestAgentControl(); }, 1400);
}

async function requestAgentControl({ fresh = false } = {}) {
  if (agentActivationPending) return;
  // A Watch Agent request from the menu should demonstrate the selected
  // recurrent policy, not roll directly into its known procedural failure
  // tail. Live mid-contract handoffs still preserve the active human seed.
  const activeContract = currentMetrics?.status === "active" && Number(currentMetrics?.time || 0) > 0;
  if (activeContract && currentMetrics?.contract === "GhostlineEnv-v3") {
    showNotice("The published runner policy uses the frozen Classic contract. Finish this Adaptive run or launch Classic before takeover.", "info");
    return;
  }
  if (!activeContract && seed() === null && $("seed-input")) {
    $("seed-input").value = String(agentShowcaseSeeds[tier()]);
  }
  agentActivationPending = true;
  if (gameReady) setBootState("running");
  else setBootState("booting", "Initializing the game before agent handoff…");
  queue("focus");
  setControlMode("handoff");
  setPolicyState("loading", "LOADING AGENT 0%");
  showNotice("Loading the recurrent policy. You can cancel the handoff at any time.", "info");
  try {
    const loaded = await ghostlinePolicy.load();
    if (!agentActivationPending) return;
    if (loaded) {
      setPolicyState("loading", "CONNECTING AGENT");
      queue("agent-ready", { fresh });
    } else {
      agentActivationPending = false;
      setControlMode("human");
      showNotice("The policy could not load. Continuing in human mode.", "error");
    }
  } catch (error) {
    agentActivationPending = false;
    setControlMode("human");
    showNotice(`The policy could not load: ${error.message}`, "error");
  }
}

async function replayPortfolioAgentRun() {
  if ($("mode-select")) $("mode-select").value = "classic";
  if ($("directive-select")) $("directive-select").value = "standard";
  if ($("tier-select")) $("tier-select").value = String(portfolioDemo.tier);
  if ($("seed-input")) $("seed-input").value = String(portfolioDemo.seed);
  showNotice("Loading the exact tier-six contract and checkpoint used by the portfolio recording.", "info");
  setIntelPanel(false, { focus: true, closeTarget: "canvas" });
  await requestAgentControl({ fresh: true });
}

function restoreHumanControl() {
  const cancelledHandoff = agentActivationPending || document.body.dataset.control === "handoff";
  agentActivationPending = false;
  setControlMode("human");
  queue("human");
  if (cancelledHandoff) showNotice("Agent handoff cancelled. Manual control remains active.", "info");
}

function maybePublishEmbedReady() {
  // Portfolio readiness describes the secure, origin-scoped web shell.  It is
  // deliberately independent from Python's user-activation gate: once the
  // policy manifest is known, the parent can reveal Ghostline's own accurate
  // download/audio progress instead of obscuring it with a competing loader.
  if (policyAvailability === null) return;
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
    gameReady = true;
    if (agentActivationPending) setBootState("running");
    else setBootState("ready");
    maybePublishEmbedReady();
  },
  setBootState,
  setPolicyState,
  setControlMode,
  showNotice,
  updateMetrics,
};

$("play-selected")?.addEventListener("click", () => {
  queue("launch-human");
  setIntelPanel(false, { focus: true, closeTarget: "canvas" });
});
$("mode-select")?.addEventListener("change", () => {
  const adaptive = contractMode() === "adaptive";
  if ($("directive-select")) $("directive-select").disabled = !adaptive;
});
if ($("directive-select")) $("directive-select").disabled = contractMode() !== "adaptive";
$("agent-control")?.addEventListener("click", () => { void requestAgentControl(); });
$("portfolio-agent-control")?.addEventListener("click", () => { void replayPortfolioAgentRun(); });
$("human-control")?.addEventListener("click", restoreHumanControl);
$("fullscreen-control")?.addEventListener("click", toggleFullscreen);
$("focus-control")?.addEventListener("click", () => $("canvas")?.focus());
$("intel-panel-control")?.addEventListener("click", () => {
  const trigger = $("intel-panel-control");
  setIntelPanel(trigger?.getAttribute("aria-expanded") !== "true", { focus: true });
});
$("mobile-lab-close")?.addEventListener("click", () => setIntelPanel(false, { focus: true }));
$("focus-game")?.addEventListener("click", () => {
  if (gameReady) {
    setIntelPanel(false, { focus: true, closeTarget: "canvas" });
    setBootState("running");
  }
  else setBootState("booting", "Loading the facility and browser runtime…");
  queue("focus");
  if (
    !document.fullscreenElement
    && globalThis.matchMedia?.("(pointer: coarse) and (orientation: landscape)").matches
  ) {
    void toggleFullscreen();
  }
});
$("game-frame")?.addEventListener("pointerdown", () => {
  $("canvas")?.focus();
  queue("focus");
});
globalThis.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && document.body.classList.contains("mobile-lab-open")) {
    setIntelPanel(false, { focus: true });
  }
});
let compactLabLayout = usesCompactLabLayout();
globalThis.addEventListener("resize", () => {
  const nextCompact = usesCompactLabLayout();
  if (nextCompact !== compactLabLayout) {
    compactLabLayout = nextCompact;
    setIntelPanel(!nextCompact);
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) queue("pause-hidden");
});
globalThis.addEventListener("blur", () => queue("pause-focus"));
globalThis.addEventListener("focus", () => queue("focus"));

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
    if (document.body.dataset.control === "handoff") setPolicyState("loading", "CONNECTING AGENT");
    else setPolicyState(state, `AGENT ONLINE // ${String(backend).toUpperCase()}`);
  } else if (state === "unavailable") {
    setPolicyState(state, `AGENT UNAVAILABLE${error ? " // RETRY" : ""}`);
    if (["agent", "handoff"].includes(document.body.dataset.control) && !policyFailureQueued) {
      policyFailureQueued = true;
      agentActivationPending = false;
      setControlMode("human");
      queue("policy-failed");
    }
  }
});
globalThis.addEventListener("ghostline:policy-inference", (event) => {
  const latency = Number(event.detail.average_latency_ms || 0);
  $("policy-latency").textContent = `${latency.toFixed(1)} ms`;
  $("policy-backend").textContent = String(event.detail.backend || "—").toUpperCase();
  if (document.body.dataset.control === "handoff") {
    agentActivationPending = false;
    setControlMode("agent");
    setPolicyState("ready", `AGENT ONLINE // ${String(event.detail.backend || "—").toUpperCase()}`);
    showNotice("Agent takeover engaged. Use TAKE CONTROL at any time.", "success");
  }
});

renderComparison();
setControlMode("human");
setBootState("booting");
setIntelPanel(!compactLabLayout);
void ghostlinePolicy.probe();
