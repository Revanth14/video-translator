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

const stages = ["upload", "transcribe", "translate", "voice", "render"];

const els = {
  dropzone: document.getElementById("dropzone"),
  videoInput: document.getElementById("videoInput"),
  sourceVideo: document.getElementById("sourceVideo"),
  uploadPanel: document.querySelector(".source-panel"),
  uploadStatus: document.getElementById("uploadStatus"),
  startButton: document.getElementById("startButton"),
  retryButton: document.getElementById("retryButton"),
  resetButton: document.getElementById("resetButton"),
  progressBar: document.getElementById("progressBar"),
  runStatus: document.getElementById("runStatus"),
  resultPreview: document.getElementById("resultPreview"),
  languageSelect: document.getElementById("languageSelect"),
  glossaryInput: document.getElementById("glossaryInput"),
  voiceReferenceInput: document.getElementById("voiceReferenceInput"),
  videoDownload: document.getElementById("videoDownload"),
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
els.retryButton.addEventListener("click", retryRun);
els.resetButton.addEventListener("click", reset);

function setVideo(file) {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  els.sourceVideo.src = state.objectUrl;
  els.uploadPanel.classList.add("has-video");
  els.startButton.disabled = false;
  els.retryButton.hidden = true;
  els.uploadStatus.textContent = file.name;
  completeStages(1);
  setProgress(12);
  els.resultPreview.innerHTML = "<span>Start translation</span>";
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
  els.retryButton.hidden = true;
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
  form.append("end", "90");
  const glossary = els.glossaryInput.files[0];
  const voiceReference = els.voiceReferenceInput.files[0];
  if (glossary) form.append("glossary", glossary);
  if (voiceReference) form.append("voice_reference", voiceReference);
  setRunning();
  try {
    const payload = await postForm("/api/jobs", form);
    state.jobId = payload.summary.run_id;
    rememberJob(state.jobId);
    renderJob(payload);
    startPolling();
  } catch (error) {
    showFailure(error.message);
    els.startButton.disabled = false;
  }
}

async function pollJob() {
  if (!state.jobId || polling) return;
  polling = true;
  try {
    renderJob(await getJson(`/api/jobs/${encodeURIComponent(state.jobId)}`));
  } catch (error) {
    handlePollFailure(error);
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
    return;
  }
  // Anything else is treated as transient, so polling continues.
  els.uploadStatus.textContent = error.message;
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
  els.retryButton.hidden = true;
  els.resultPreview.innerHTML = "<span>Result appears here</span>";
  disableDownloads();
  setProgress(state.file ? 12 : 0);
  completeStages(state.file ? 1 : 0);
}

function renderJob(payload) {
  const summary = payload.summary;
  const stageStatuses = summary.stages || {};
  state.stageStatuses = stageStatuses;
  const completed = [
    stageStatuses.ingest,
    stageStatuses.transcribe,
    stageStatuses.localize,
    stageStatuses.synthesize,
    stageStatuses.render,
  ].filter((status) => status === "completed").length;
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
  if (summary.status === "failed" || summary.status === "cancelled") {
    stopPolling();
    showFailure(payload.errors?.at(-1) || "Translation failed.");
    return;
  }
  if (summary.status === "rendered") {
    stopPolling();
    setProgress(100);
    completeStages(stages.length);
    els.runStatus.textContent = "Ready";
    els.runStatus.dataset.state = "done";
    els.uploadStatus.textContent = "Done";
    return;
  }
}

function setRunning() {
  els.startButton.disabled = true;
  els.retryButton.hidden = true;
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
  els.videoDownload.textContent = `${language} video`;
}

function retryRun() {
  if (!state.file) return;
  stopPolling();
  state.jobId = null;
  state.stageStatuses = {};
  forgetJob();
  completeStages(1);
  setProgress(12);
  disableDownloads();
  startRun();
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
  els.startButton.disabled = true;
  els.retryButton.hidden = true;
  els.uploadStatus.textContent = "No file";
  els.runStatus.textContent = "Ready";
  delete els.runStatus.dataset.state;
  els.resultPreview.innerHTML = "<span>Result appears here</span>";
  disableDownloads();
  setProgress(0);
  completeStages(0);
  drawPoster();
}

function stopPolling() {
  if (state.poller) window.clearInterval(state.poller);
  state.poller = null;
}

function startPolling() {
  stopPolling();
  state.poller = window.setInterval(pollJob, 1000);
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
  pollJob();
  startPolling();
}

function showFailure(message) {
  els.runStatus.textContent = "Failed";
  els.runStatus.dataset.state = "failed";
  els.uploadStatus.textContent = message;
  els.startButton.disabled = false;
  els.retryButton.hidden = false;
}

function setProgress(value) {
  state.progress = value;
  els.progressBar.style.width = `${value}%`;
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
  context.fillStyle = "#111816";
  context.fillRect(0, 0, width, height);

  for (let row = 0; row < 9; row += 1) {
    for (let col = 0; col < 14; col += 1) {
      const x = 42 + col * 66;
      const y = 38 + row * 54;
      const alpha = 0.08 + ((row + col) % 4) * 0.025;
      context.fillStyle = `rgba(255,255,255,${alpha})`;
      context.fillRect(x, y, 44, 28);
    }
  }

  context.fillStyle = "#0b7f68";
  context.fillRect(72, 380, 560, 12);
  context.fillStyle = "#9db6ad";
  context.fillRect(72, 410, 360, 12);
  context.fillStyle = "rgba(255,255,255,0.86)";
  context.font = "700 44px system-ui, sans-serif";
  context.fillText("Upload video", 72, 180);
  context.font = "500 24px system-ui, sans-serif";
  context.fillText("Localize voice, subtitles, and timing", 72, 220);
}

function canUseBackend() {
  return window.location.protocol === "http:" || window.location.protocol === "https:";
}

function customerStatus(summary) {
  const current = Object.entries(summary.stages || {})
    .find(([, status]) => status === "running")?.[0];
  if (current === "ingest") return "Uploading";
  if (current === "transcribe") return "Transcribing";
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
    transcribed: "Translating",
    localized: "Generating voice",
    synthesized: "Rendering",
    rendered: "Ready",
    failed: "Failed",
  };
  return labels[summary.status] || "Running";
}

async function getJson(url) {
  const response = await fetch(url);
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }
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
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Request failed.");
  return payload;
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
  els.videoDownload.textContent = "Video";
}
