const MESSAGE_SOURCE = "ghostline";
const MESSAGE_VERSION = 1;
const CONTROLLERS = new Set(["human", "agent", "hybrid"]);
const RESULTS = new Set(["success", "failed"]);


function httpOrigin(value) {
  if (!value) return null;
  try {
    const url = new URL(String(value));
    return url.protocol === "https:" || url.protocol === "http:" ? url.origin : null;
  } catch {
    return null;
  }
}


export function resolveParentOrigin(windowObject, documentObject) {
  const referrerOrigin = httpOrigin(documentObject?.referrer);
  const ancestorValue = windowObject?.location?.ancestorOrigins?.[0];
  const ancestorOrigin = httpOrigin(ancestorValue);
  if (referrerOrigin && ancestorOrigin && referrerOrigin !== ancestorOrigin) return null;
  return referrerOrigin ?? ancestorOrigin;
}


function integer(value, minimum, maximum) {
  return Number.isInteger(value) && value >= minimum && value <= maximum ? value : null;
}


function finite(value, minimum, maximum) {
  return Number.isFinite(value) && value >= minimum && value <= maximum ? value : null;
}


export function runCompleteMessage(metrics) {
  if (!metrics || typeof metrics !== "object") return null;
  const controller = String(metrics.mode ?? "");
  const status = String(metrics.status ?? "");
  const tier = integer(metrics.tier, 1, 6);
  const seed = integer(metrics.seed, 0, 2_147_483_647);
  const duration = finite(metrics.time, 0, 86_400);
  if (
    !CONTROLLERS.has(controller)
    || !RESULTS.has(status)
    || tier === null
    || seed === null
    || duration === null
  ) {
    return null;
  }
  return Object.freeze({
    source: MESSAGE_SOURCE,
    version: MESSAGE_VERSION,
    type: "run-complete",
    controller,
    success: status === "success",
    tier,
    seed,
    duration,
  });
}


export class GhostlineEmbedBridge {
  constructor({ windowObject = globalThis.window, documentObject = globalThis.document } = {}) {
    this.window = windowObject;
    this.document = documentObject;
    const query = new URLSearchParams(String(windowObject?.location?.search ?? ""));
    this.presentationMode = query.get("embed") === "1";
    this.framed = Boolean(windowObject?.parent && windowObject.parent !== windowObject);
    this.targetOrigin = this.framed
      ? resolveParentOrigin(windowObject, documentObject)
      : null;
    this.readySent = false;
    this.modelAvailable = false;
    const mode = this.presentationMode ? "true" : "false";
    if (documentObject?.documentElement?.dataset) documentObject.documentElement.dataset.embed = mode;
    if (documentObject?.body?.dataset) documentObject.body.dataset.embed = mode;
  }

  setModelAvailable(available) {
    this.modelAvailable = Boolean(available);
  }

  markReady(modelAvailable = this.modelAvailable) {
    if (this.readySent) return false;
    const posted = this._post(Object.freeze({
      source: MESSAGE_SOURCE,
      version: MESSAGE_VERSION,
      type: "ready",
      modelAvailable: Boolean(modelAvailable),
    }));
    this.readySent = posted;
    return posted;
  }

  publishRunComplete(metrics) {
    const message = runCompleteMessage(metrics);
    return message ? this._post(message) : false;
  }

  _post(message) {
    if (!this.presentationMode || !this.framed || !this.targetOrigin) return false;
    try {
      this.window.parent.postMessage(message, this.targetOrigin);
      return true;
    } catch {
      return false;
    }
  }
}
