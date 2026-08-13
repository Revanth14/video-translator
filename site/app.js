const state = {
  file: null,
  objectUrl: null,
  progress: 0,
  timer: null,
  jobId: null,
  poller: null,
  stageStatuses: {},
};

const jobStorageKey = "video-translator-job-id";
let polling = false;
const pollDelayMs = 1000;

const stages = ["upload", "transcribe", "translate", "voice", "render"];

const els = {
  dropzone: document.getElementById("dropzone"),
  videoInput: document.getElementById("videoInput"),
  sourceVideo: document.getElementById("sourceVideo"),
  uploadPanel: document.querySelector(".source-panel"),
  uploadStatus: document.getElementById("uploadStatus"),
  errorBanner: document.getElementById("errorBanner"),
  errorTitle: document.getElementById("errorTitle"),
  errorMessage: document.getElementById("errorMessage"),
  startButton: document.getElementById("startButton"),
  resetButton: document.getElementById("resetButton"),
  progressBar: document.getElementById("progressBar"),
  stageProgress: document.getElementById("stageProgress"),
  utteranceProgress: document.getElementById("utteranceProgress"),
  attemptProgress: document.getElementById("attemptProgress"),
  progressValue: document.getElementById("progressValue"),
  runStatus: document.getElementById("runStatus"),
  resultPreview: document.getElementById("resultPreview"),
  languageSelect: document.getElementById("languageSelect"),
  glossaryInput: document.getElementById("glossaryInput"),
  voiceReferenceInput: document.getElementById("voiceReferenceInput"),
  videoDownload: document.getElementById("videoDownload"),
  videoDownloadLabel: document.getElementById("videoDownloadLabel"),
  subtitleDownload: document.getElementById("subtitleDownload"),
  audioDownload: document.getElementById("audioDownload"),
};

drawPoster();
restoreJob();

els.videoInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) setVideo(file);
});

["dragenter", "dragover"].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("dragging");
  });
});

["dragleave", "drop"].forEach((name) => {
  els.dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("dragging");
  });
});

els.dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file && file.type.startsWith("video/")) setVideo(file);
});

els.startButton.addEventListener("click", startRun);
els.resetButton.addEventListener("click", reset);

function setVideo(file) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  els.sourceVideo.src = state.objectUrl;
  els.uploadPanel.classList.add("has-video");
  hideError();
  els.startButton.disabled = false;
  els.uploadStatus.textContent = file.name;
  completeStages(1);
  setProgress(12);
  els.resultPreview.innerHTML = pendingResultMarkup();
  disableDownloads();
}

function startRun() {
  if (!state.file || state.timer) return;
  if (canUseBackend()) {
    startBackendRun();
    return;
  }
  startDemoRun();
}

function startDemoRun() {
  const language = els.languageSelect.options[els.languageSelect.selectedIndex].text;
  els.startButton.disabled = true;
  els.runStatus.textContent = "Running";
  els.runStatus.dataset.state = "running";
  els.uploadStatus.textContent = "Uploaded";
  setProgress(18);
  completeStages(1);

  let tick = 0;
  state.timer = window.setInterval(() => {
    tick += 1;
    const progress = [32, 48, 66, 84, 100][tick - 1] || 100;
    setProgress(progress);
    completeStages(Math.min(tick + 1, stages.length));
    if (progress >= 100) finishRun(language);
  }, 850);
}

async function startBackendRun() {
  const language = els.languageSelect.value;
  const form = new FormData();
  form.append("video", state.file);
  form.append("language", language);
  form.append("start", "0");
  const glossary = els.glossaryInput.files[0];
  const voiceReference = els.voiceReferenceInput.files[0];
  if (glossary) form.append("glossary", glossary);
  if (voiceReference) form.append("voice_reference", voiceReference);
  setRunning();
  try {
    const payload = await postForm("/api/jobs", form);
    state.jobId = payload.summary.run_id;
    rememberJob(state.jobId);
    if (renderJob(payload)) startPolling();
  } catch (error) {
    showFailure(error.message, { allowResubmit: true });
  }
}

async function pollJob() {
  const jobId = state.jobId;
  if (!jobId || polling) return Boolean(jobId);
  polling = true;
  try {
    const payload = await getJson(`/api/jobs/${encodeURIComponent(jobId)}`);
    // Reset or a newly submitted run may replace the observed job while this
    // request is in flight. Never repaint the page with that stale response.
    if (state.jobId !== jobId) return false;
    return renderJob(payload);
  } catch (error) {
    if (state.jobId !== jobId) return false;
    return handlePollFailure(error);
  } finally {
    polling = false;
  }
}

function handlePollFailure(error) {
  // A missing job is permanent: the run was removed, or this browser
  // remembered a run that belongs to another machine. Release it so the page
  // is usable again instead of polling a job that will never appear.
  if (error.status === 400 || error.status === 404) {
    discardMissingJob();
    return false;
  }
  // Anything else is treated as transient, so polling continues.
  els.uploadStatus.textContent = "Reconnecting";
  showError(error.message, "Connection interrupted");
  return true;
}

function discardMissingJob() {
  stopPolling();
  forgetJob();
  state.jobId = null;
  state.stageStatuses = {};
  els.runStatus.textContent = "Ready";
  delete els.runStatus.dataset.state;
  els.uploadStatus.textContent = state.file
    ? state.file.name
    : "That job is no longer available";
  els.startButton.disabled = !state.file;
  els.resultPreview.innerHTML = emptyResultMarkup();
  disableDownloads();
  setProgress(state.file ? 12 : 0);
  completeStages(state.file ? 1 : 0);
}

function renderJob(payload) {
  hideError();
  const summary = payload.summary;
  const stageStatuses = summary.stages || {};
  state.stageStatuses = stageStatuses;
  renderDurableProgress(summary.progress || {});
  const completed = [
    stageStatuses.ingest === "completed",
    stageStatuses.transcribe === "completed"
      && stageStatuses.segment === "completed",
    stageStatuses.localize === "completed",
    stageStatuses.synthesize === "completed",
    stageStatuses.render === "completed",
  ].filter(Boolean).length;
  completeStages(completed);
  setProgress(Math.max(12, completed * 20));
  els.runStatus.textContent = customerStatus(summary);
  els.runStatus.dataset.state = summary.status === "failed" ? "failed" : "running";
  els.uploadStatus.textContent = summary.run_id;
  const outputs = summary.outputs || {};
  const videoPath = outputs.dubbed_video;
  if (videoPath) {
    const videoUrl = mediaUrl(videoPath, summary.run_id);
    els.resultPreview.innerHTML = (
      `<video controls playsinline src="${videoUrl}"></video>`
    );
    setDownload(els.videoDownload, videoPath, summary.run_id);
    setDownload(els.subtitleDownload, outputs.hindi_srt, summary.run_id);
    setDownload(els.audioDownload, outputs.dubbed_audio, summary.run_id);
  }
  if (summary.status === "failed") {
    stopPolling();
    showFailure(payload.errors?.at(-1) || "Translation failed.");
    return false;
  }
  if (summary.status === "cancelled") {
    stopPolling();
    els.runStatus.textContent = "Cancelled";
    els.runStatus.dataset.state = "failed";
    els.uploadStatus.textContent = "Translation cancelled.";
    els.startButton.disabled = true;
    return false;
  }
  if (summary.status === "rendered") {
    stopPolling();
    setProgress(100);
    completeStages(stages.length);
    els.runStatus.textContent = "Ready";
    els.runStatus.dataset.state = "done";
    els.uploadStatus.textContent = "Done";
    return false;
  }
  return true;
}

function renderDurableProgress(progress) {
  const stageProgress = progress.stages || {};
  const utterances = progress.utterances || {};
  const attempts = progress.attempts || {};
  els.stageProgress.textContent = (
    `${stageProgress.completed || 0} of ${stageProgress.total || 6} `
    + "stages complete"
  );
  els.utteranceProgress.textContent = (
    `${utterances.synthesized || 0} of ${utterances.total || 0} `
    + "utterances voiced"
  );
  els.attemptProgress.textContent = `${attempts.total || 0} attempts`;
}

function setRunning() {
  hideError();
  els.startButton.disabled = true;
  els.runStatus.textContent = "Running";
  els.runStatus.dataset.state = "running";
  els.uploadStatus.textContent = state.jobId ? "Working" : "Uploading";
  setProgress(Math.max(state.progress, 18));
}

function finishRun(language) {
  window.clearInterval(state.timer);
  state.timer = null;
  els.runStatus.textContent = "Ready";
  els.runStatus.dataset.state = "done";
  els.resultPreview.innerHTML = "";
  const preview = els.sourceVideo.cloneNode(true);
  preview.controls = true;
  preview.muted = false;
  preview.currentTime = 0;
  els.resultPreview.appendChild(preview);
  setDownload(els.videoDownload, state.objectUrl, "");
  els.videoDownloadLabel.textContent = `${language} video`;
}

function reset() {
  if (state.timer) window.clearInterval(state.timer);
  stopPolling();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = null;
  state.objectUrl = null;
  state.timer = null;
  state.jobId = null;
  state.stageStatuses = {};
  forgetJob();
  els.videoInput.value = "";
  els.sourceVideo.removeAttribute("src");
  els.sourceVideo.load();
  els.uploadPanel.classList.remove("has-video");
  hideError();
  els.startButton.disabled = true;
  els.uploadStatus.textContent = "No file";
  els.runStatus.textContent = "Ready";
  delete els.runStatus.dataset.state;
  els.resultPreview.innerHTML = emptyResultMarkup();
  disableDownloads();
  setProgress(0);
  completeStages(0);
  renderDurableProgress({});
  drawPoster();
}

function stopPolling() {
  if (state.poller) window.clearTimeout(state.poller);
  state.poller = null;
}

function scheduleNextPoll() {
  if (!state.jobId || state.poller) return;
  state.poller = window.setTimeout(async () => {
    state.poller = null;
    if (await pollJob()) scheduleNextPoll();
  }, pollDelayMs);
}

function startPolling({ immediate = false } = {}) {
  stopPolling();
  if (immediate) {
    void (async () => {
      if (await pollJob()) scheduleNextPoll();
    })();
    return;
  }
  scheduleNextPoll();
}

function rememberJob(runId) {
  if (!canUseBackend()) return;
  try {
    window.localStorage.setItem(jobStorageKey, runId);
  } catch (_error) {
    // URL persistence remains available when storage is blocked.
  }
  const url = new URL(window.location.href);
  url.searchParams.set("job", runId);
  window.history.replaceState({}, "", url);
}

function forgetJob() {
  if (!canUseBackend()) return;
  try {
    window.localStorage.removeItem(jobStorageKey);
  } catch (_error) {
    // Continue clearing the URL even when storage is blocked.
  }
  const url = new URL(window.location.href);
  url.searchParams.delete("job");
  window.history.replaceState({}, "", url);
}

function restoreJob() {
  if (!canUseBackend()) return;
  const url = new URL(window.location.href);
  let storedRunId = null;
  try {
    storedRunId = window.localStorage.getItem(jobStorageKey);
  } catch (_error) {
    storedRunId = null;
  }
  const runId = url.searchParams.get("job") || storedRunId;
  if (!runId) return;
  state.jobId = runId;
  rememberJob(runId);
  setRunning();
  startPolling({ immediate: true });
}

function showFailure(message, { allowResubmit = false } = {}) {
  els.runStatus.textContent = "Failed";
  els.runStatus.dataset.state = "failed";
  els.uploadStatus.textContent = allowResubmit ? "Not submitted" : "Stopped";
  showError(message);
  els.startButton.disabled = !(allowResubmit && state.file);
}

function showError(message, title = "Couldn’t start translation") {
  els.errorTitle.textContent = title;
  els.errorMessage.textContent = message;
  els.errorBanner.hidden = false;
}

function hideError() {
  els.errorBanner.hidden = true;
  els.errorMessage.textContent = "";
}

function setProgress(value) {
  state.progress = value;
  els.progressBar.style.width = `${value}%`;
  els.progressValue.textContent = `${value}%`;
  const progress = els.progressBar.parentElement;
  if (progress) progress.setAttribute("aria-valuenow", String(value));
}

function completeStages(count) {
  stages.forEach((stage, index) => {
    const card = document.querySelector(`[data-stage="${stage}"]`);
    card.classList.toggle("complete", index < count);
  });
}

function drawPoster() {
  const canvas = document.getElementById("posterCanvas");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#dfe9ff";
  context.fillRect(0, 0, width, height);

  for (let index = 0; index < 14; index += 1) {
    const heightScale = 48 + ((index * 31) % 140);
    context.fillStyle = index % 3 === 0
      ? "rgba(102,92,220,0.18)"
      : "rgba(42,116,238,0.14)";
    context.fillRect(74 + index * 60, 270 - heightScale / 2, 18, heightScale);
  }

  context.fillStyle = "rgba(255,255,255,0.7)";
  context.fillRect(90, 418, 780, 2);
}

function emptyResultMarkup() {
  return `
    <span class="empty-preview">
      <span class="result-orb" aria-hidden="true">
        <span></span>
        <svg viewBox="0 0 28 28"><path d="m11 9 8 5-8 5V9Z"/></svg>
      </span>
      <strong>Your Hindi video will appear here</strong>
      <small>You can leave this page. Progress is saved automatically.</small>
    </span>`;
}

function pendingResultMarkup() {
  return `
    <span class="empty-preview">
      <span class="result-orb" aria-hidden="true">
        <span></span>
        <svg viewBox="0 0 28 28"><path d="m11 9 8 5-8 5V9Z"/></svg>
      </span>
      <strong>Ready when you are</strong>
      <small>Start translation to create the Hindi version.</small>
    </span>`;
}

function canUseBackend() {
  return window.location.protocol === "http:" || window.location.protocol === "https:";
}

function customerStatus(summary) {
  const current = Object.entries(summary.stages || {})
    .find(([, status]) => status === "running")?.[0];
  if (current === "ingest") return "Uploading";
  if (current === "transcribe") return "Transcribing";
  if (current === "segment") return "Preparing speech";
  if (current === "localize") return "Translating";
  if (current === "synthesize") return "Generating voice";
  if (current === "render") return "Rendering";
  const queued = Object.values(summary.stages || {})
    .some((status) => status === "queued");
  if (summary.status === "queued" || queued) return "Queued";
  const labels = {
    created: "Queued",
    queued: "Queued",
    ingested: "Transcribing",
    transcribed: "Preparing speech",
    segmented: "Translating",
    localized: "Generating voice",
    synthesized: "Rendering",
    rendered: "Ready",
    failed: "Failed",
  };
  return labels[summary.status] || "Running";
}

async function getJson(url) {
  const response = await fetch(url);
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    const error = new Error(payload.error || "Request failed.");
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function postForm(url, form) {
  const response = await fetch(url, {
    method: "POST",
    body: form,
  });
  const payload = await readJsonResponse(response);
  if (!response.ok) {
    const error = new Error(payload.error || "Request failed.");
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function readJsonResponse(response) {
  try {
    return await response.json();
  } catch (_error) {
    const unavailable = [404, 405, 501].includes(response.status);
    const message = unavailable
      ? (
        "The translation API is unavailable on this server. "
        + "Start it with: uv run dub-mvp web --no-open --port 8787"
      )
      : `The server returned an invalid response (HTTP ${response.status}).`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
}

function mediaUrl(path, runId) {
  if (!path) return "";
  if (path.startsWith("blob:")) return path;
  return `/media/${mediaPath(path, runId)}`;
}

function mediaPath(path, runId) {
  const marker = `/${runId}/`;
  const index = path.indexOf(marker);
  const relative = index >= 0 ? `${runId}/${path.slice(index + marker.length)}` : path;
  return encodeURIComponent(relative);
}

function setDownload(link, path, runId) {
  if (!path) return;
  link.href = mediaUrl(path, runId);
  link.classList.remove("disabled");
  link.setAttribute("download", "");
}

function disableDownloads() {
  [els.videoDownload, els.subtitleDownload, els.audioDownload].forEach((link) => {
    link.removeAttribute("href");
    link.removeAttribute("download");
    link.classList.add("disabled");
  });
  els.videoDownloadLabel.textContent = "Video";
}
