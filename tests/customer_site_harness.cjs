const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const scenario = process.argv[2];
const appPath = path.join(__dirname, "..", "site", "app.js");
const appSource = fs.readFileSync(appPath, "utf8");

function job(runId, status) {
  const terminal = status === "rendered";
  const failed = status === "failed";
  return {
    summary: {
      run_id: runId,
      status,
      stages: {
        ingest: terminal || failed ? "completed" : "queued",
        transcribe: terminal ? "completed" : "pending",
        segment: terminal ? "completed" : "pending",
        localize: terminal ? "completed" : "pending",
        synthesize: terminal ? "completed" : "pending",
        render: terminal ? "completed" : "pending",
      },
      outputs: {},
    },
    errors: failed ? ["provider rejected the request"] : [],
  };
}

const cases = {
  rendered: {
    href: "http://127.0.0.1:8787/",
    stored: "run-rendered",
    responses: [[200, job("run-rendered", "rendered")]],
  },
  active_then_rendered: {
    href: "http://127.0.0.1:8787/",
    stored: "run-active",
    responses: [
      [200, job("run-active", "queued")],
      [200, job("run-active", "rendered")],
    ],
  },
  url_precedence: {
    href: "http://127.0.0.1:8787/?campaign=demo&job=url-run",
    stored: "stored-run",
    responses: [[200, job("url-run", "rendered")]],
  },
  missing_url: {
    href: "http://127.0.0.1:8787/?campaign=demo&job=missing-run",
    stored: "stored-run",
    responses: [[404, { error: "Unknown run" }]],
  },
  reset: {
    href: "http://127.0.0.1:8787/?campaign=demo&job=run-active",
    stored: "run-active",
    responses: [[200, job("run-active", "queued")]],
  },
  failed: {
    href: "http://127.0.0.1:8787/?job=run-failed",
    stored: null,
    responses: [[200, job("run-failed", "failed")]],
  },
  cancelled: {
    href: "http://127.0.0.1:8787/?job=run-cancelled",
    stored: null,
    responses: [[200, job("run-cancelled", "cancelled")]],
  },
  reset_while_fetching: {
    href: "http://127.0.0.1:8787/?campaign=demo&job=run-active",
    stored: "run-active",
    responses: [[200, job("run-active", "queued")]],
  },
  non_json_submit: {
    href: "http://127.0.0.1:8787/",
    stored: null,
    responses: [[501, null, true]],
  },
};

if (!(scenario in cases)) throw new Error(`Unknown scenario: ${scenario}`);
const selected = cases[scenario];
const storage = new Map();
if (selected.stored) storage.set("video-translator-job-id", selected.stored);
const requestedIds = new Set();
const fetches = [];
const responses = [...selected.responses];
const timers = new Map();
let nextTimerId = 1;
let releaseFetch = null;

function element(id) {
  return {
    id,
    files: [],
    disabled: false,
    hidden: false,
    textContent: "",
    innerHTML: "",
    value: "hi",
    selectedIndex: 0,
    options: [{ text: "Hindi" }],
    dataset: {},
    style: {},
    classList: {
      add() {},
      remove() {},
      toggle() {},
    },
    addEventListener() {},
    removeAttribute(name) {
      delete this[name];
    },
    setAttribute(name, value) {
      this[name] = value;
    },
    load() {},
    appendChild() {},
    cloneNode() {
      return element(`${id}-clone`);
    },
  };
}

const elements = new Map();
function getElement(id) {
  requestedIds.add(id);
  if (!elements.has(id)) elements.set(id, element(id));
  return elements.get(id);
}

const canvas = getElement("posterCanvas");
canvas.width = 1000;
canvas.height = 500;
canvas.getContext = () => ({
  clearRect() {},
  fillRect() {},
  fillText() {},
  fillStyle: "",
  font: "",
});

const location = {
  href: selected.href,
  protocol: new URL(selected.href).protocol,
};
const windowObject = {
  location,
  localStorage: {
    getItem(key) {
      return storage.get(key) ?? null;
    },
    setItem(key, value) {
      storage.set(key, String(value));
    },
    removeItem(key) {
      storage.delete(key);
    },
  },
  history: {
    replaceState(_state, _unused, url) {
      location.href = String(url);
      location.protocol = new URL(location.href).protocol;
    },
  },
  setTimeout(callback) {
    const id = nextTimerId;
    nextTimerId += 1;
    timers.set(id, callback);
    return id;
  },
  clearTimeout(id) {
    timers.delete(id);
  },
  setInterval() {
    return 999;
  },
  clearInterval() {},
};

const context = vm.createContext({
  console,
  URL,
  encodeURIComponent,
  FormData: class {
    append() {}
  },
  document: {
    getElementById: getElement,
    querySelector(selector) {
      return getElement(selector);
    },
  },
  window: windowObject,
  fetch: async (url) => {
    fetches.push(url);
    const [status, payload, invalidJson] = (
      responses.shift() ?? [500, { error: "extra poll" }, false]
    );
    const response = {
      ok: status >= 200 && status < 300,
      status,
      async json() {
        if (invalidJson) throw new SyntaxError("Unexpected token '<'");
        return payload;
      },
    };
    if (scenario !== "reset_while_fetching") return response;
    return new Promise((resolve) => {
      releaseFetch = () => resolve(response);
    });
  },
});

function flush() {
  return new Promise((resolve) => setImmediate(resolve));
}

async function runNextTimer() {
  const entry = timers.entries().next().value;
  if (!entry) throw new Error("Expected a scheduled poll");
  const [id, callback] = entry;
  timers.delete(id);
  await callback();
  await flush();
}

(async () => {
  vm.runInContext(appSource, context, { filename: appPath });
  if (scenario === "reset_while_fetching") {
    vm.runInContext("reset()", context);
    releaseFetch();
  }
  await flush();
  await flush();
  const initialTimerCount = timers.size;
  if (scenario === "active_then_rendered") await runNextTimer();
  if (scenario === "reset") {
    vm.runInContext("reset()", context);
    await flush();
  }
  if (scenario === "non_json_submit") {
    vm.runInContext("state.file = { name: 'demo.mp4' }", context);
    await vm.runInContext(
      "postForm('/api/jobs', {}).catch((error) => "
      + "showFailure(error.message, { allowResubmit: true }))",
      context,
    );
    await flush();
  }
  const state = vm.runInContext("state", context);
  process.stdout.write(JSON.stringify({
    fetches,
    initial_timer_count: initialTimerCount,
    timer_count: timers.size,
    stored_job: storage.get("video-translator-job-id") ?? null,
    href: location.href,
    state_job_id: state.jobId,
    run_status: getElement("runStatus").textContent,
    upload_status: getElement("uploadStatus").textContent,
    start_disabled: getElement("startButton").disabled,
    retry_element_requested: requestedIds.has("retryButton"),
    error_message: getElement("errorMessage").textContent,
    error_visible: getElement("errorBanner").hidden === false,
  }));
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
