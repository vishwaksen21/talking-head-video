/* Voice Studio frontend. Plain JS, no build step. Every value shown comes from the API. */
"use strict";

const $app = document.getElementById("app");
const S = {
  GENERATING_VIDEO: "GENERATING_VIDEO", EXTRACTING_AUDIO: "EXTRACTING_AUDIO", TRANSCRIBING: "TRANSCRIBING",
  AWAITING: "AWAITING_TRANSCRIPT_REVIEW", SYNTHESIZING: "SYNTHESIZING_VOICE", SYNCHRONIZING: "SYNCHRONIZING",
  COMPOSING: "COMPOSING", COMPLETED: "COMPLETED", FAILED: "FAILED", CANCELLED: "CANCELLED", IDLE: "IDLE",
};
const STATE_LABEL = {
  IDLE: "Queued", GENERATING_VIDEO: "Generating video", EXTRACTING_AUDIO: "Extracting audio",
  TRANSCRIBING: "Transcribing", AWAITING_TRANSCRIPT_REVIEW: "Review transcript", SYNTHESIZING_VOICE: "Creating voice",
  SYNCHRONIZING: "Synchronising", COMPOSING: "Building video", COMPLETED: "Completed", FAILED: "Failed", CANCELLED: "Cancelled",
  VALIDATING_INPUT: "Validating input", PREPARING_REFERENCE: "Preparing photo", GENERATING_SPEECH: "Generating speech",
  VALIDATING_DIALOGUE: "Checking speech", AWAITING_AUDIO_APPROVAL: "Approve audio", VALIDATING_LIPSYNC: "Checking lip-sync",
  VALIDATING_IDENTITY: "Checking identity", VALIDATING_OUTPUT: "Checking output", NEEDS_REVIEW: "Needs review",
  LIKENESS_ACCEPTED: "Likeness accepted", LONG_READY: "In progress", LONG_GENERATING: "Generating clips",
  LONG_STITCHING: "Cutting together", LONG_MATCHING: "Matching voices",
};
const ACTIVE = new Set([S.IDLE, S.GENERATING_VIDEO, S.EXTRACTING_AUDIO, S.TRANSCRIBING, S.SYNTHESIZING, S.SYNCHRONIZING, S.COMPOSING,
  "VALIDATING_INPUT", "PREPARING_REFERENCE", "GENERATING_SPEECH", "VALIDATING_DIALOGUE", "VALIDATING_LIPSYNC", "VALIDATING_IDENTITY", "VALIDATING_OUTPUT",
  "LONG_GENERATING", "LONG_STITCHING", "LONG_MATCHING"]);

let pollTimer = null;
let currentProject = null; // { id, sig }

// ------------------------------------------------------------------ helpers
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const secs = (v) => (v === null || v === undefined ? "—" : `${Number(v).toFixed(2)} s`);
const when = (t) => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

async function api(path, options = {}) {
  const init = { headers: { "Content-Type": "application/json" }, ...options };
  if (init.body && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  let res;
  try {
    res = await fetch(path, init);
  } catch {
    throw new Error("The Voice Studio server is not responding. Is it still running?");
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
  if (!res.ok) {
    const d = data && data.detail;
    throw new Error(Array.isArray(d) ? d.map((x) => x.msg).join("; ") : d && typeof d === "object" ? d.message || JSON.stringify(d)
      : d || `Request failed (${res.status})`);
  }
  return data;
}

function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast" + (isError ? " error" : "");
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (el.hidden = true), isError ? 7000 : 3500);
}

async function busy(button, fn) {
  if (button.disabled) return;
  button.disabled = true;
  button.classList.add("loading");
  try { await fn(); } catch (e) { toast(e.message, true); } finally {
    if (button.isConnected) { button.disabled = false; button.classList.remove("loading"); }
  }
}

function pill(state) {
  const cls = state === S.COMPLETED || state === "LIKENESS_ACCEPTED" ? "pill-done" : state === S.FAILED ? "pill-failed" :
    [S.AWAITING, "AWAITING_AUDIO_APPROVAL", "NEEDS_REVIEW"].includes(state) ? "pill-review" : ACTIVE.has(state) ? "pill-active" : "";
  return `<span class="pill ${cls}">${esc(STATE_LABEL[state] || state)}</span>`;
}

function stopPolling() { clearTimeout(pollTimer); pollTimer = null; }

// ------------------------------------------------------------------ router
async function route() {
  stopPolling();
  stopRecorders();
  currentProject = null;
  const hash = location.hash || "#/create";
  const [, name, rawId] = hash.split("?")[0].split("/");
  const id = rawId;
  document.querySelectorAll(".nav-links a").forEach((a) =>
    a.toggleAttribute("aria-current", a.dataset.route === (name === "project" ? "projects" : name)));
  if (a11yFirst) a11yFirst = false; else $app.focus({ preventScroll: true });
  window.scrollTo(0, 0);
  try {
    if (name === "settings") await renderSettings();
    else if (name === "projects") await renderProjects();
    else if (name === "chatterbox") await renderChatterbox();
    else if (name === "project" && id) await renderProject(id);
    else await renderCreate();
  } catch (e) {
    $app.innerHTML = `<div class="notice notice-error"><strong>Something went wrong</strong>${esc(e.message)}</div>`;
  }
}
let a11yFirst = true;
window.addEventListener("hashchange", route);

// ------------------------------------------------------------------ settings
async function renderSettings() {
  const [settings, system] = await Promise.all([api("/api/settings"), api("/api/system")]);
  const voices = system.voices;
  $app.innerHTML = `
    <div class="page-head"><h1>Settings</h1>
      <p class="lead">Connect the Utilities API and check that the voice tools on this Mac are ready.</p></div>
    <div class="stack">
      <section class="card" aria-labelledby="api-h">
        <div class="card-head"><div><h2 id="api-h">Utilities API</h2>
          <p>The key is stored on this computer by the Voice Studio server. It is never sent back to the browser.</p></div></div>
        <form id="settings-form" autocomplete="off">
          <div class="field"><label for="base_url">Base URL</label>
            <input id="base_url" type="url" required value="${esc(settings.base_url)}" placeholder="https://your-gateway.example.com">
            <p class="hint">You can paste the address of the API's /docs page. It is trimmed to the API root.</p></div>
          <div class="field"><label for="api_key">API key</label>
            <input id="api_key" type="password" placeholder="${settings.api_key_set ? "A key is saved. Leave blank to keep it" : "Paste your API key"}">
          </div>
          <div class="row" style="margin-top:24px">
            <button class="btn btn-primary" type="submit" id="save-btn">Save</button>
            <button class="btn btn-secondary" type="button" id="test-btn">Test connection</button>
            <span id="conn-status" class="muted small" role="status">${settings.api_key_set ? "Key saved" : "No key saved yet"}</span>
          </div>
        </form>
        <div id="models-area" style="margin-top:24px"></div>
      </section>

      <section class="card" aria-labelledby="sys-h">
        <div class="card-head"><div><h2 id="sys-h">Voice tools on this Mac</h2>
          <p>Chatterbox creates the cloned voice locally. FFmpeg handles the audio and video files.</p></div></div>
        <dl class="facts">
          <dt>Chatterbox</dt><dd>${system.chatterbox.installed ? "Installed" : "Not available: " + esc(system.chatterbox.detail)}</dd>
          <dt>Model</dt><dd>${system.chatterbox.loaded ? "Loaded" : "Loads on first use (about 20 s)"}${system.chatterbox.load_error ? " — last error: " + esc(system.chatterbox.load_error) : ""}</dd>
          <dt>Runs on</dt><dd>${system.chatterbox.device === "mps" ? "Apple GPU (MPS)" : esc(system.chatterbox.device)}</dd>
          <dt>FFmpeg</dt><dd>${system.ffmpeg ? "Installed" : "Missing. Install it with: brew install ffmpeg"}</dd>
        </dl>
        <h3 style="margin-top:24px">Target voices</h3>
        <div class="row" style="margin-bottom:12px"><label for="voice-file" class="btn btn-secondary" style="margin:0">Add a voice file</label>
          <input id="voice-file" type="file" accept=".wav,.mp3,.flac,.m4a,audio/*" class="sr-only">
          <span class="muted small">6 to 10 seconds of clean speech works best. Only the first 10 seconds are used.</span></div>
        <div class="table-wrap"><table>
          <thead><tr><th>Voice</th><th class="num">Clip length</th><th>Listen</th></tr></thead>
          <tbody>${voices.map((v) => `<tr><td>${esc(v.label)}${v.id === system.default_voice ? ' <span class="pill">Default</span>' : ""}</td>
            <td class="num">${v.duration ? secs(v.duration) + (v.duration > 10 ? " (first 10 s used)" : "") : "—"}</td>
            <td>${v.id === "builtin" ? '<span class="muted small">Chatterbox’s own voice</span>' : `<audio controls preload="none" src="/api/voices/${encodeURIComponent(v.id)}/audio"></audio>`}</td></tr>`).join("")}</tbody>
        </table></div>
      </section>
    </div>`;

  document.getElementById("voice-file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    try {
      const res = await fetch("/api/voices", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      toast(`Voice added: ${data.label}`);
      renderSettings();
    } catch (e) { toast(e.message, true); }
  });
  const status = document.getElementById("conn-status");
  document.getElementById("settings-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("save-btn"), async () => {
      const keyEl = document.getElementById("api_key");
      const saved = await api("/api/settings", { method: "PUT", body: { base_url: document.getElementById("base_url").value, api_key: keyEl.value || null } });
      keyEl.value = "";
      keyEl.placeholder = "A key is saved. Leave blank to keep it";
      document.getElementById("base_url").value = saved.base_url;
      status.textContent = "Saved";
      toast("Settings saved");
    });
  });
  document.getElementById("test-btn").addEventListener("click", (ev) => busy(ev.currentTarget, async () => {
    status.textContent = "Testing…";
    try {
      const r = await api("/api/settings/test", { method: "POST" });
      status.textContent = `Connected. ${r.video_models} video model${r.video_models === 1 ? "" : "s"} available.`;
      await showModels(document.getElementById("models-area"));
    } catch (e) {
      status.textContent = "Not connected";
      document.getElementById("models-area").innerHTML = `<div class="notice notice-error"><strong>Connection failed</strong>${esc(e.message)}</div>`;
    }
  }));
}

async function showModels(el) {
  const data = await api("/api/models");
  el.innerHTML = `
    <h3>Video models from the API</h3>
    <div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>Lengths</th><th>Shapes</th><th>Quality</th><th>Sound</th></tr></thead>
      <tbody>${data.models.map((m) => `<tr><td><strong>${esc(m.label)}</strong><div class="muted small">${esc(m.id)}</div>
        ${m.granted_note ? `<div class="small" style="color:var(--red-dark)">${esc(m.granted_note)}</div>` : ""}</td>
        <td>${m.durations.join(", ")} s</td><td>${m.aspect_ratios.join(", ")}</td><td>${m.resolutions.join(", ") || "Standard"}</td>
        <td>${m.supports_audio ? "Yes" : "No"}${m.capabilities_known ? "" : '<div class="muted small">Options not confirmed</div>'}</td></tr>`).join("")}</tbody>
    </table></div>
    <p class="hint">${esc(data.source)}</p>`;
}

// ------------------------------------------------------------------ create
async function renderCreate() {
  const m = new URLSearchParams(location.hash.split("?")[1] || "").get("mode");
  const mode = ["creative", "long"].includes(m) ? m : "strict";
  $app.innerHTML = `
    <div class="page-head"><h1>Create a video</h1>
      <p class="lead">Choose how the video is made. Strict Talking Head keeps one uploaded person speaking your exact words.</p></div>
    <div class="mode-switch" role="radiogroup" aria-label="Generation mode">
      <a role="radio" aria-checked="${mode === "strict"}" class="mode ${mode === "strict" ? "on" : ""}" href="#/create?mode=strict">
        <strong>Strict Talking Head</strong><span>One 8-second clip of the person in your photo saying your words.</span></a>
      <a role="radio" aria-checked="${mode === "long"}" class="mode ${mode === "long" ? "on" : ""}" href="#/create?mode=long">
        <strong>Long talking head</strong><span>1 to 1.5 minutes: several clips of the same person, cut together.</span></a>
      <a role="radio" aria-checked="${mode === "creative"}" class="mode ${mode === "creative" ? "on" : ""}" href="#/create?mode=creative">
        <strong>Veo Creative Video</strong><span>Free-form scenes from prompts. Speech is re-voiced afterwards. No identity guarantee.</span></a>
    </div>
    <section class="card" style="margin-top:24px"><div id="create-body"><div class="status-line"><span class="pill pill-active">Loading</span></div></div></section>`;
  return mode === "strict" ? renderStrict() : mode === "long" ? renderLongCreate() : renderCreative();
}

async function renderCreative() {
  let data;
  try {
    data = await api("/api/models");
  } catch (e) {
    document.getElementById("create-body").innerHTML = `<div class="notice notice-error"><strong>Video models could not be loaded</strong>${esc(e.message)}</div>
      <p style="margin-top:16px"><a class="btn btn-secondary" href="#/settings">Open Settings</a></p>`;
    return;
  }
  const models = data.models;
  const preferred = models.find((m) => m.id === "veo-3.1-fast-generate-001") || models.find((m) => !m.granted_note) || models[0];
  document.getElementById("create-body").innerHTML = `
    <form id="create-form">
      <span class="label">Scenes</span>
      <p class="hint" style="margin-top:0">Each scene is one generation from your quota. Prompts are sent exactly as written.</p>
      <div id="scenes"></div>
      <button class="btn btn-quiet" type="button" id="add-scene">+ Add a scene</button>
      <div class="field" style="margin-top:16px"><label for="model">Model</label>
        <select id="model">${models.map((m) => `<option value="${esc(m.id)}" ${m.id === preferred.id ? "selected" : ""}>${esc(m.label)} (${esc(m.id)})</option>`).join("")}</select>
        <p class="hint" id="model-note"></p></div>
      <div class="options-grid field" id="model-options"></div>
      <div class="field"><label for="ref-file">Reference image (optional)</label>
        <input id="ref-file" type="file" accept="image/png,image/jpeg,image/webp">
        <div id="ref-preview" class="row" style="margin-top:8px"></div>
        <p class="hint" id="ref-note">A photo of the person or product to keep consistent across scenes. Veo makes 8-second clips when an image is attached.</p></div>
      <div class="field"><label for="script">Approved script (optional)</label>
        <textarea id="script" rows="4" maxlength="2000" placeholder="One sentence per line. These exact words are spoken in your voice, whatever the video says."></textarea>
        <p class="hint">Leave empty to re-voice whatever is said in the generated video.</p></div>
      <details class="more"><summary>More options</summary><div>
        <div class="field"><label for="pname">Project name (optional)</label><input id="pname" type="text" placeholder="e.g. prudential-test-001" pattern="[a-z0-9]([a-z0-9]|-){2,63}"></div>
        <div class="field"><label for="export-name">Export file name (optional)</label><input id="export-name" type="text" placeholder="final_video.mp4"></div>
        <div class="field"><label class="check"><input type="checkbox" id="endcard"> End with a white card showing the Prudential logo (the real logo file, never drawn by the model)</label></div>
        <div class="field"><label for="negative">Things to avoid (optional)</label><input id="negative" type="text" placeholder="e.g. text on screen, logos"></div>
        <div class="field"><label for="seed">Seed (optional)</label><input id="seed" type="number" min="0" max="4294967295" step="1" placeholder="Random"></div>
      </div></details>
      <div class="row" style="margin-top:24px">
        <button class="btn btn-primary" type="submit" id="generate-btn">Generate video</button>
        <span class="muted small" id="quota-note">Uses one video generation from your API quota.</span>
      </div>
    </form>`;

  const modelEl = document.getElementById("model");
  const scenesEl = document.getElementById("scenes");
  let reference = null;
  const sceneHTML = (n) => `<div class="field" data-scene><label>Scene ${n}</label>
    <div class="row" style="align-items:flex-start;flex-wrap:nowrap"><textarea required maxlength="4000" rows="4" style="flex:1"
      placeholder='${n === 1 ? 'A woman at a kitchen table looks at the camera and says: "Our family plan covers all of us."' : "What happens next"}'></textarea>
      ${n > 1 ? `<button class="btn btn-quiet" type="button" data-remove-scene>Remove</button>` : ""}</div>
    <label class="check small" data-ref-toggle hidden style="margin-top:8px"><input type="checkbox" data-use-ref checked> Use the reference image in this scene</label></div>`;
  const renumber = () => {
    const rows = [...scenesEl.querySelectorAll("[data-scene]")];
    rows.forEach((r, i) => (r.querySelector("label").textContent = `Scene ${i + 1}`));
    document.getElementById("add-scene").hidden = rows.length >= 4;
    document.getElementById("quota-note").textContent = rows.length === 1 ? "Uses one video generation from your API quota."
      : `Uses ${rows.length} video generations from your API quota, one per scene. Scenes are joined with a short crossfade.`;
  };
  scenesEl.innerHTML = sceneHTML(1);
  document.getElementById("add-scene").addEventListener("click", () => {
    scenesEl.insertAdjacentHTML("beforeend", sceneHTML(scenesEl.children.length + 1));
    renumber();
    syncRefToggles();
    scenesEl.lastElementChild.querySelector("textarea").focus();
  });
  scenesEl.addEventListener("click", (ev) => { if (ev.target.matches("[data-remove-scene]")) { ev.target.closest("[data-scene]").remove(); renumber(); } });
  renumber();

  const renderOptions = () => {
    const m = models.find((x) => x.id === modelEl.value);
    const durations = reference ? [m.reference_duration || 8] : m.durations;
    const sel = (id, label, values, def, fmt = (v) => v) => values.length ? `<div><label for="${id}">${label}</label>
      <select id="${id}" ${values.length === 1 ? "disabled" : ""}>${values.map((v) => `<option value="${esc(v)}" ${v === def ? "selected" : ""}>${esc(fmt(v))}</option>`).join("")}</select></div>` : "";
    document.getElementById("model-options").innerHTML =
      sel("duration", "Length per scene", durations, reference ? durations[0] : m.default_duration, (v) => `${v} seconds`) +
      sel("aspect", "Shape", m.aspect_ratios, m.aspect_ratios[0], (v) => (v === "16:9" ? "Landscape 16:9" : v === "9:16" ? "Portrait 9:16" : v)) +
      sel("resolution", "Quality", m.resolutions, m.resolutions[0]) +
      (m.supports_audio ? `<div><span class="label">Sound</span><label class="check"><input type="checkbox" id="audio" checked> Generate speech and sound</label></div>` : "");
    const notes = [];
    if (!m.supports_audio) notes.push("This model makes silent videos, so you will type the words to speak after it finishes.");
    if (reference && !m.supports_reference_images) notes.push("This model cannot take a reference image through the API. Choose a Veo 3.1 model.");
    if (m.granted_note) notes.push(m.granted_note);
    if (!m.capabilities_known) notes.push("Its options are not confirmed by the API.");
    document.getElementById("model-note").textContent = notes.join(" ");
  };
  modelEl.addEventListener("change", renderOptions);
  renderOptions();

  const fileEl = document.getElementById("ref-file");
  const showRef = () => {
    document.getElementById("ref-preview").innerHTML = reference ? `<img src="${esc(reference.preview)}" alt="Reference image" style="height:80px;border-radius:var(--r-sm);border:1px solid var(--gray-border)">
      <span class="small muted">${esc(reference.name)} · ${reference.width}×${reference.height}</span>
      <button class="btn btn-quiet" type="button" id="ref-clear">Remove</button>` : "";
    const clear = document.getElementById("ref-clear");
    if (clear) clear.addEventListener("click", () => { reference = null; fileEl.value = ""; showRef(); renderOptions(); syncRefToggles(); });
  };
  fileEl.addEventListener("change", async () => {
    const f = fileEl.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    document.getElementById("ref-note").textContent = "Uploading…";
    try {
      const res = await fetch("/api/uploads/image", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      reference = { ...data, preview: URL.createObjectURL(f) };
      document.getElementById("ref-note").textContent = "Attached. Veo makes 8-second clips when an image is attached.";
    } catch (e) {
      reference = null; fileEl.value = "";
      document.getElementById("ref-note").textContent = e.message;
      toast(e.message, true);
    }
    showRef();
    renderOptions();
    syncRefToggles();
  });
  const syncRefToggles = () => scenesEl.querySelectorAll("[data-ref-toggle]").forEach((el) => (el.hidden = !reference));

  document.getElementById("create-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("generate-btn"), async () => {
      const val = (id) => document.getElementById(id)?.value || null;
      const seed = val("seed");
      const scenes = [...scenesEl.querySelectorAll("[data-scene]")].map((row) => ({
        prompt: row.querySelector("textarea").value, use_references: row.querySelector("[data-use-ref]").checked }));
      const p = await api("/api/projects", { method: "POST", body: {
        scenes, model: modelEl.value, name: val("pname"),
        duration: val("duration") ? Number(val("duration")) : null,
        aspect_ratio: val("aspect"), resolution: val("resolution"),
        generate_audio: document.getElementById("audio") ? document.getElementById("audio").checked : false,
        negative_prompt: val("negative"), seed: seed === null ? null : Number(seed),
        reference_ids: reference ? [reference.id] : [], script: val("script"),
        end_card: document.getElementById("endcard").checked, export_name: val("export-name"),
      } });
      location.hash = `#/project/${p.id}`;
    });
  });
}

// ------------------------------------------------------------------ projects
async function renderProjects() {
  const list = await api("/api/projects");
  $app.innerHTML = `
    <div class="page-head row"><div><h1>Projects</h1><p class="lead">Every video you have generated. Open one to watch, edit the words or download.</p></div>
      <span class="spacer"></span><a class="btn btn-primary" href="#/create">New video</a></div>
    ${list.length ? `<div class="table-wrap"><table>
      <thead><tr><th>Created</th><th>Prompt</th><th>Model</th><th>Status</th><th></th></tr></thead>
      <tbody>${list.map((p) => `<tr><td>${esc(when(p.created_at))}</td><td><div class="prompt-cell">${esc(p.prompt)}</div>
        ${p.error ? `<div class="small" style="color:var(--red-dark)">${esc(p.error)}</div>` : ""}</td>
        <td class="small">${esc(p.model)}</td><td>${pill(p.state)}</td>
        <td><a class="btn btn-quiet" href="#/project/${esc(p.id)}">Open</a></td></tr>`).join("")}</tbody></table></div>`
      : `<div class="card empty"><h2>No projects yet</h2><p>Generate your first video to see it here.</p><a class="btn btn-primary" href="#/create">Create a video</a></div>`}`;
}

// ------------------------------------------------------------------ project
function stepOf(p) {
  const s = [S.FAILED, S.CANCELLED].includes(p.state) ? p.failed_stage || S.GENERATING_VIDEO : p.state;
  if ([S.IDLE, S.GENERATING_VIDEO].includes(s)) return 0;
  if ([S.EXTRACTING_AUDIO, S.TRANSCRIBING, S.AWAITING].includes(s)) return 1;
  if (s === S.SYNTHESIZING) return 2;
  if ([S.SYNCHRONIZING, S.COMPOSING].includes(s)) return 3;
  return 4;
}

function signature(p) {
  return JSON.stringify([p.state, Object.keys(p.files), p.warnings, p.error, p.transcript && p.transcript.edited,
    p.params && p.params.strict && [p.params.strict.take, p.params.strict.validation && p.params.strict.validation.run_id]]);
}

async function renderProject(id, keepScroll = false) {
  const [p, system] = await Promise.all([api(`/api/projects/${encodeURIComponent(id)}`), renderProject.system || api("/api/system")]);
  renderProject.system = system;
  currentProject = { id, sig: signature(p) };
  const y = window.scrollY;
  if (p.params && p.params.mode === "long") {
    currentProject = null;
    return renderLong(id, keepScroll);
  }
  if (p.params && p.params.mode === "strict") {
    $app.innerHTML = strictHTML(p, system);
    bindStrict(p);
  } else {
    $app.innerHTML = projectHTML(p, system);
    bindProject(p, system);
  }
  if (keepScroll) window.scrollTo(0, y);
  schedulePoll(p);
}

function schedulePoll(p) {
  stopPolling();
  if (!p.active && p.state !== S.IDLE) return;
  pollTimer = setTimeout(async () => {
    if (!currentProject || currentProject.id !== p.id) return;
    let fresh;
    try { fresh = await api(`/api/projects/${encodeURIComponent(p.id)}`); } catch { return schedulePoll(p); }
    if (!currentProject || currentProject.id !== p.id) return;
    if (signature(fresh) !== currentProject.sig) {
      renderProject.system = null; // refresh model-loaded status too
      return renderProject(p.id, true);
    }
    document.querySelectorAll("[data-live-message]").forEach((el) => (el.textContent = fresh.message || ""));
    schedulePoll(fresh);
  }, 2000);
}

function card(n, status, title, sub, body) {
  return `<section class="card" data-status="${status}" aria-labelledby="step-${n}">
    <div class="card-head"><span class="step-num" aria-hidden="true">${status === "done" ? "✓" : n}</span>
      <div><h2 id="step-${n}">${title}</h2><p>${sub}</p></div></div>${body}</section>`;
}

function live(p) {
  return `<div class="status-line">${pill(p.state)}<span data-live-message>${esc(p.message || "")}</span></div><div class="progress" aria-hidden="true"><span></span></div>`;
}

function projectHTML(p, system) {
  const step = stepOf(p);
  const st = (i) => (p.state === S.COMPLETED || i < step ? "done" : i === step ? "current" : "waiting");
  const failed = [S.FAILED, S.CANCELLED].includes(p.state);
  const m = p.media || {};
  const t = p.transcript || {};
  const cfg = (p.params && p.params.video_config) || {};
  const f = p.files;

  const head = `<div class="page-head"><div class="row"><a class="btn btn-quiet" href="#/projects">← Projects</a></div>
    <div class="row" style="margin-top:8px"><h1 style="margin:0">Project</h1>${pill(p.state)}<span class="spacer"></span>
      ${p.active ? `<button class="btn btn-gray" id="cancel-btn">Cancel</button>` : ""}
      ${failed ? `<button class="btn btn-primary" id="retry-btn">Retry</button>` : ""}
      ${failed && /Generate again/.test(p.error || "") ? `<button class="btn btn-secondary" id="regen-btn">Generate again (uses 1 generation)</button>` : ""}</div>
    <p class="muted small" style="margin:8px 0 0">Created ${esc(when(p.created_at))} · ${esc(p.model)}</p></div>`;

  const banners = [
    p.state === S.FAILED ? `<div class="notice notice-error"><strong>This step failed</strong>${esc(p.error)}</div>` : "",
    p.state === S.CANCELLED ? `<div class="notice"><strong>Cancelled</strong>Press Retry to continue from where it stopped.</div>` : "",
    ...(p.warnings || []).map((w) => `<div class="notice notice-warn">${esc(w)}</div>`),
  ].join("");

  // 1. video
  const scenePrompts = (p.params && p.params.scenes) || null;
  const refs = (p.params && p.params.reference_files) || [];
  const sceneFiles = Object.keys(f).filter((k) => k.startsWith("scene_")).sort();
  const videoBody = `
    ${scenePrompts ? scenePrompts.map((sc, i) => `<div class="field"><span class="label">Scene ${i + 1} prompt${refs.length ? (sc.use_references ? " · with reference image" : " · text only") : ""}</span><p style="margin:0;white-space:pre-wrap">${esc(sc.prompt)}</p></div>`).join("")
      : `<div class="field"><span class="label">Prompt</span><p style="margin:0;white-space:pre-wrap">${esc(p.prompt)}</p></div>`}
    ${refs.length ? `<div class="field"><span class="label">Reference image sent to the model</span>${refs.map((r) => `<img src="/api/projects/${esc(p.id)}/reference/${encodeURIComponent(r.path.split("/").pop())}" alt="Reference image" style="height:96px;border-radius:var(--r-sm);border:1px solid var(--gray-border)">`).join(" ")}</div>` : ""}
    <dl class="facts" style="margin-top:16px">
      <dt>Length</dt><dd>${cfg.duration_seconds ? cfg.duration_seconds + " s" : "Model default"}</dd>
      <dt>Shape</dt><dd>${esc(cfg.aspect_ratio || "Model default")}</dd>
      ${cfg.resolution ? `<dt>Quality</dt><dd>${esc(cfg.resolution)}</dd>` : ""}
      <dt>Sound</dt><dd>${cfg.generate_audio ? "Requested" : "Not requested"}</dd>
    </dl>
    ${step === 0 && p.active ? `<div style="margin-top:16px">${live(p)}</div>` : ""}
    ${f["generated.mp4"] ? `<div class="media-box" style="margin-top:16px"><video controls preload="metadata" src="${f["generated.mp4"]}"></video></div>
      ${sceneFiles.length > 1 ? `<details class="more"><summary>Watch each scene clip</summary><div class="grid-2">${sceneFiles.map((k, i) => `<div><div class="media-box"><video controls preload="metadata" src="${f[k]}"></video></div><p class="media-label">Scene ${i + 1}${m.generated && m.generated.scenes ? ` · starts at ${secs(m.generated.scenes[i].start)}` : ""}</p></div>`).join("")}</div></details>` : ""}
      <p class="media-label">${sceneFiles.length > 1 ? `${sceneFiles.length} scenes joined` : "Generated video"} · ${secs(m.generated && m.generated.duration)} · ${m.generated ? `${m.generated.width}×${m.generated.height}` : ""} · ${m.generated && m.generated.has_audio ? "has sound" : "no sound"}</p>` : ""}`;

  // 2. audio + transcript
  const oa = m.original_audio;
  const canEdit = f["generated.mp4"] && [S.AWAITING, S.COMPLETED, S.FAILED, S.CANCELLED].includes(p.state) && step >= 1;
  const audioBody = `
    ${step === 1 && p.active ? live(p) : ""}
    ${f["original_audio.wav"] ? `<span class="label">Original soundtrack</span><audio controls preload="metadata" src="${f["original_audio.wav"]}"></audio>
      <p class="media-label">${secs(oa && oa.duration)} · ${oa ? `${oa.sample_rate} Hz · ${oa.channels === 1 ? "mono" : oa.channels + " channels"} · WAV (from ${esc(m.generated.audio_codec || "")})` : ""}</p>` : ""}
    ${canEdit ? (p.state === S.COMPLETED ? `<details class="more"><summary>Change the words and create the voice again</summary><div>${editorHTML(p, system)}</div></details>` : editorHTML(p, system))
      : (t.original && t.original.length ? transcriptReadOnly(t.edited || t.original) : step < 1 ? `<p class="muted">Starts when the video is ready.</p>` : "")}`;

  // 3. voice
  const lines = Object.keys(f).filter((k) => k.startsWith("line_"));
  const voiceName = (system.voices.find((v) => v.id === (p.params && p.params.voice)) || {}).label;
  const voiceBody = `
    <dl class="facts"><dt>Target voice</dt><dd>${esc(voiceName || "Chosen when you continue")}</dd>
      <dt>Voice model</dt><dd>Chatterbox ${system.chatterbox.installed ? "(ready" + (system.chatterbox.loaded ? ", loaded)" : ", loads on first use)") : "— not available"}</dd></dl>
    ${step === 2 && p.active ? `<div style="margin-top:16px">${live(p)}</div>` : ""}
    ${f["cloned_voice.wav"] ? `<div style="margin-top:16px"><span class="label">Cloned voice track</span><audio controls preload="metadata" src="${f["cloned_voice.wav"]}"></audio></div>` : ""}
    ${lines.length ? `<details class="more"><summary>Listen to each line</summary><div>${lines.map((k, i) => `<p class="media-label">Line ${i + 1}</p><audio controls preload="none" src="${f[k]}"></audio>`).join("")}</div></details>` : ""}`;

  // 4. final
  const sync = p.sync;
  const fin = m.final;
  const finalBody = `
    ${step === 3 && p.active ? live(p) : ""}
    ${f["final.mp4"] ? `<div class="grid-2">
        <div><div class="media-box"><video controls preload="metadata" src="${f["generated.mp4"]}"></video></div><p class="media-label">Original</p></div>
        <div><div class="media-box"><video controls preload="metadata" src="${f["final.mp4"]}"></video></div><p class="media-label">With the cloned voice</p></div>
      </div>
      <div class="grid-2" style="margin-top:24px">
        <dl class="facts"><dt>Original video</dt><dd>${secs(sync && sync.video_duration)}</dd>
          <dt>Cloned speech ends at</dt><dd>${secs(sync && sync.speech_end)}</dd>
          <dt>Final video</dt><dd>${secs(fin && fin.duration)}</dd>
          <dt>Last frame held</dt><dd>${sync && sync.freeze_seconds ? secs(sync.freeze_seconds) : "No"}</dd>
          <dt>Sync</dt><dd>${sync ? (sync.status === "in_sync" ? "Each line starts where the original line started" : sync.status === "extended" ? "Speech is longer than the video. Last frame held" : "Adjusted. See the notes above") : "—"}</dd></dl>
        <div><a class="btn btn-primary" href="${f["final.mp4"]}?download=true" download>Download MP4</a>
          <p class="hint">${fin ? `${fin.width}×${fin.height} · H.264 video · AAC audio · ${(fin.bytes / 1048576).toFixed(1)} MB` : ""}</p></div>
      </div>
      ${sync ? `<details class="more"><summary>Technical details</summary><div><div class="table-wrap"><table>
        <thead><tr><th>Line</th><th class="num">Original start</th><th class="num">Placed at</th><th class="num">Length</th><th class="num">Speed</th></tr></thead>
        <tbody>${sync.placements.map((x) => `<tr><td>${x.index + 1}</td><td class="num">${secs(x.original_start)}</td><td class="num">${secs(x.start)}</td>
          <td class="num">${secs(x.end - x.start)}</td><td class="num">${x.tempo > 1 ? "+" + Math.round((x.tempo - 1) * 100) + "%" : "Natural"}</td></tr>`).join("")}</tbody></table></div>
        <p class="hint">Transcription: ${esc(t.model || "—")}, timing ${t.timing === "measured" ? "measured from the audio" : "estimated by the model"}.</p></div></details>` : ""}`
      : `<p class="muted">Appears when the voice is ready.</p>`}`;

  return head + `<div class="stack">${banners ? `<div>${banners}</div>` : ""}
    ${card(1, st(0), "Video", "Generated from your prompt by the selected model.", videoBody)}
    ${card(2, st(1), "Soundtrack and words", "The audio is taken from the video and the speech is written out for you to check.", audioBody)}
    ${card(3, st(2), "Cloned voice", "Chatterbox speaks the checked words in the target voice.", voiceBody)}
    ${card(4, st(3), "Final video", "The original pictures with the cloned voice, lined up line by line.", finalBody)}</div>`;
}

function transcriptReadOnly(lines) {
  return `<span class="label" style="margin-top:16px">Transcript</span>` + lines.map((l) => `<p style="margin:4px 0"><span class="muted small">${secs(l.start)}</span> ${esc(l.text)}</p>`).join("");
}

function editorHTML(p, system) {
  const t = p.transcript || {};
  const params = p.params || {};
  const lines = (t.edited && t.edited.length ? t.edited : t.proposed && t.proposed.length ? t.proposed
    : t.original && t.original.length ? t.original : [{ start: 0.3, text: "" }]);
  const scripted = Boolean(t.approved_script);
  const originals = scripted ? [] : t.original || [];
  const cmp = t.comparison;
  const low = new Set(t.low_confidence || []);
  const voice = params.voice || system.default_voice;
  return `<form id="transcript-form" style="margin-top:16px">
    ${scripted ? `<span class="label">Approved script</span>
      <p class="hint" style="margin-top:0">These lines are spoken in your voice. Each is placed in its scene; adjust start times if needed.</p>
      <div class="notice ${cmp && cmp.status === "differs" ? "notice-warn" : ""}" style="margin-bottom:16px"><strong>What the generated video said</strong>
        ${cmp && cmp.status !== "no_speech" ? `“${esc(cmp.heard)}”<br><span class="small">${cmp.status === "matches" ? "Matches the approved script." : `Differs from the approved script (${Math.round(cmp.similarity * 100)}% word match). The approved script is used.`}${cmp.placement === "aligned_to_speech" ? " Lines are timed to where the presenter says them." : ""}</span>` : "No speech was recognised in the generated audio. The approved script is used."}</div>`
    : `<span class="label">${t.has_speech ? "Transcript: check and correct it" : "What should the voice say?"}</span>
    <p class="hint" style="margin-top:0">${t.has_speech ? "Each line keeps its start time so the new voice speaks when the original did. Edit any words that are wrong." : esc(t.reason || "") + " Type each line and when it should start."}</p>`}
    <div id="lines">${lines.map((l, i) => lineHTML(l, originals[i], low.has(i), p.media.generated.duration)).join("")}</div>
    <button class="btn btn-quiet" type="button" id="add-line">+ Add a line</button>
    <div class="grid-2" style="margin-top:24px">
      <div class="field"><label for="voice">Target voice</label>
        <select id="voice">${system.voices.map((v) => `<option value="${esc(v.id)}" ${v.id === voice ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></div>
      <div class="field" style="margin-top:0"><span class="label">If the new speech is longer than the video</span>
        <label class="check"><input type="radio" name="overflow" value="freeze" ${params.overflow !== "stretch" ? "checked" : ""}> Hold the last frame until it finishes</label>
        <label class="check"><input type="radio" name="overflow" value="stretch" ${params.overflow === "stretch" ? "checked" : ""}> Speak up to 30% faster first</label></div>
    </div>
    <details class="more"><summary>More options</summary><div class="options-grid">
      ${slider("max_speedup", "Max speed-up per line", 1, 1.3, 0.01, params.max_speedup ?? 1.15, (v) => `+${Math.round((v - 1) * 100)}%`)}
      ${slider("exaggeration", "Expressiveness", 0, 1, 0.05, params.exaggeration ?? 0.5)}
      ${slider("cfg_weight", "Closeness to the voice", 0, 1, 0.05, params.cfg_weight ?? 0.5)}
      ${slider("temperature", "Variation", 0.1, 1.5, 0.05, params.temperature ?? 0.8)}
    </div></details>
    <div class="row" style="margin-top:24px"><button class="btn btn-primary" type="submit" id="voice-btn">Create cloned voice</button>
      <span class="muted small">Runs on this Mac. About 10 to 20 seconds per line.</span></div>
  </form>`;
}

function slider(id, label, min, max, step, value, fmt = (v) => Number(v).toFixed(2)) {
  return `<div><label for="${id}">${label}: <span data-out="${id}">${esc(fmt(value))}</span></label>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" data-fmt="${id === "max_speedup" ? "pct" : "num"}"></div>`;
}

function lineHTML(line, original, isLow, duration) {
  const sceneTag = line.scene ? `<div class="orig">Scene ${line.scene}</div>` : "";
  const changed = original && original.text !== line.text;
  const target = line.target_length ? `<div class="orig">On screen: ${Number(line.target_length).toFixed(2)} s</div>` : "";
  return `<div class="line" data-line data-target="${line.target_length || ""}">
    <div><label class="small">Starts at (s)</label><input type="number" step="0.01" min="0" max="${duration}" value="${Number(line.start).toFixed(2)}" data-start aria-label="Start time in seconds">${sceneTag}${target}</div>
    <div><label class="small">Words</label><textarea data-text rows="2" aria-label="Words to speak">${esc(line.text)}</textarea>
      ${original ? `<div class="orig">Heard: ${changed ? `<mark>${esc(original.text)}</mark>` : esc(original.text)}</div>` : ""}
      ${isLow ? `<div class="flag">Low confidence. Please listen and check.</div>` : ""}</div>
    <div><span class="label small" aria-hidden="true">&nbsp;</span><button class="btn btn-quiet" type="button" data-remove aria-label="Remove line">Remove</button></div></div>`;
}

function bindProject(p, system) {
  const cancel = document.getElementById("cancel-btn");
  if (cancel) cancel.addEventListener("click", () => busy(cancel, async () => {
    await api(`/api/projects/${p.id}/cancel`, { method: "POST" });
    toast("Cancelling…");
    renderProject(p.id, true);
  }));
  const retry = document.getElementById("retry-btn");
  if (retry) retry.addEventListener("click", () => busy(retry, async () => {
    await api(`/api/projects/${p.id}/retry`, { method: "POST" });
    renderProject(p.id, true);
  }));
  const regen = document.getElementById("regen-btn");
  if (regen) regen.addEventListener("click", () => busy(regen, async () => {
    await api(`/api/projects/${p.id}/retry?approve_new_generation=true`, { method: "POST" });
    renderProject(p.id, true);
  }));
  const form = document.getElementById("transcript-form");
  if (!form) return;
  const linesEl = document.getElementById("lines");
  const duration = p.media.generated.duration;
  linesEl.addEventListener("click", (ev) => {
    if (ev.target.matches("[data-remove]")) {
      if (linesEl.querySelectorAll("[data-line]").length > 1) ev.target.closest("[data-line]").remove();
      else toast("Keep at least one line.", true);
    }
  });
  document.getElementById("add-line").addEventListener("click", () => {
    const starts = [...linesEl.querySelectorAll("[data-start]")].map((i) => Number(i.value) || 0);
    const next = Math.min(starts.length ? Math.max(...starts) + 1 : 0.3, Math.max(0, duration - 0.5));
    linesEl.insertAdjacentHTML("beforeend", lineHTML({ start: next, text: "" }, null, false, duration));
    linesEl.lastElementChild.querySelector("textarea").focus();
  });
  form.querySelectorAll('input[type="range"]').forEach((r) => r.addEventListener("input", () => {
    const v = Number(r.value);
    form.querySelector(`[data-out="${r.id}"]`).textContent = r.dataset.fmt === "pct" ? `+${Math.round((v - 1) * 100)}%` : v.toFixed(2);
  }));
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("voice-btn"), async () => {
      const segments = [...linesEl.querySelectorAll("[data-line]")].map((row) => ({
        start: Number(row.querySelector("[data-start]").value), text: row.querySelector("[data-text]").value,
        target_length: row.dataset.target ? Number(row.dataset.target) : null,
      }));
      const num = (id) => Number(document.getElementById(id).value);
      await api(`/api/projects/${p.id}/transcript`, { method: "POST", body: {
        segments, voice: document.getElementById("voice").value,
        overflow: form.querySelector('input[name="overflow"]:checked').value,
        max_speedup: num("max_speedup"), exaggeration: num("exaggeration"), cfg_weight: num("cfg_weight"), temperature: num("temperature"),
      } });
      renderProject(p.id, true);
    });
  });
}

// ------------------------------------------------------------------ microphone recorder
// Captures raw PCM from the microphone and hands back a 16-bit mono WAV, so the server gets
// exactly what was heard (no lossy browser codec). Echo cancellation, noise suppression and
// automatic gain are off: they change the timbre that Chatterbox would copy.
const recorders = new Set();
function stopRecorders() { recorders.forEach((r) => r.cancel()); recorders.clear(); }

const TAP_WORKLET = `class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; }
  process(inputs) {
    const chans = inputs[0];
    if (chans && chans.length) {
      const out = new Float32Array(chans[0].length);
      for (const c of chans) for (let i = 0; i < c.length; i++) out[i] += c[i] / chans.length;
      this.buf.push(out); this.n += out.length;
      if (this.n >= 2048) { this.port.postMessage(this.buf); this.buf = []; this.n = 0; }
    }
    return true;
  }
}
registerProcessor("tap", Tap);`;

function micError(e) {
  if (e && (e.name === "NotAllowedError" || e.name === "SecurityError"))
    return "Microphone access is blocked. Allow it in the address bar and in System Settings → Privacy & Security → Microphone.";
  if (e && e.name === "NotFoundError") return "No microphone was found. Connect one, or choose an input in System Settings → Sound.";
  if (e && e.name === "NotReadableError") return "The microphone is being used by another app. Close it and try again.";
  return `The microphone could not be started: ${e && e.message ? e.message : e}`;
}

function wavBlob(chunks, rate) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const v = new DataView(new ArrayBuffer(44 + n * 2));
  const tag = (o, s) => { for (let i = 0; i < 4; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  tag(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); tag(8, "WAVE"); tag(12, "fmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); tag(36, "data"); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, c[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([v.buffer], { type: "audio/wav" });
}

function recorderHTML(id) {
  return `<div class="recorder" id="${id}">
    <div class="row"><button type="button" class="btn btn-secondary rec-toggle"><span class="rec-dot"></span><span class="rec-label">Start recording</span></button>
      <span class="rec-time">0:00</span><div class="rec-meter" aria-hidden="true"><span></span></div></div>
    <div class="rec-result" aria-live="polite"></div></div>`;
}

// onChange receives {blob, name, seconds} after a usable recording, or null when it is cleared.
function mountRecorder(id, { maxSeconds = 300, minSeconds = 1, prefix = "recording", onChange = () => {} } = {}) {
  const root = document.getElementById(id);
  const btn = root.querySelector(".rec-toggle"), label = root.querySelector(".rec-label");
  const time = root.querySelector(".rec-time"), bar = root.querySelector(".rec-meter > span"), result = root.querySelector(".rec-result");
  let ctx = null, stream = null, chunks = [], peak = 0, level = 0, clipped = 0, starting = false, started = 0, timer = null, raf = 0, url = null;
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

  const release = () => {
    clearInterval(timer); cancelAnimationFrame(raf);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ctx) ctx.close().catch(() => {});
    ctx = stream = null; bar.style.width = "0"; root.classList.remove("is-recording");
  };
  const clear = () => { if (url) URL.revokeObjectURL(url); url = null; result.innerHTML = ""; onChange(null); };

  async function start() {
    clear();
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      result.innerHTML = `<div class="notice notice-error"><strong>Recording is not available here</strong>Open the page at http://127.0.0.1:8800 (not a network address) in Chrome, Safari or Edge.</div>`;
      return;
    }
    ctx = new AudioContext();   // created inside the click so Safari allows it
    starting = true; btn.disabled = true; label.textContent = "Starting microphone…";
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      await ctx.resume();
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([TAP_WORKLET], { type: "application/javascript" })));
    } catch (e) {
      const cancelled = !ctx;   // the page or panel was left while the microphone was starting
      starting = false; btn.disabled = false; release(); label.textContent = "Start recording";
      if (cancelled) return;
      result.innerHTML = `<div class="notice notice-error"><strong>Could not record</strong>${esc(micError(e))}</div>`;
      return;
    }
    chunks = []; peak = 0; level = 0; clipped = 0;
    const tap = new AudioWorkletNode(ctx, "tap");
    tap.port.onmessage = (ev) => {
      for (const c of ev.data) {
        chunks.push(c);
        let m = 0; for (let i = 0; i < c.length; i++) { const a = Math.abs(c[i]); if (a > m) m = a; if (a >= 0.999) clipped++; }
        level = Math.max(m, level * 0.85); peak = Math.max(peak, m);
      }
    };
    const mute = ctx.createGain(); mute.gain.value = 0;   // keeps the graph running without playing back
    ctx.createMediaStreamSource(stream).connect(tap); tap.connect(mute); mute.connect(ctx.destination);
    starting = false; btn.disabled = false;
    started = performance.now(); root.classList.add("is-recording"); label.textContent = "Stop recording"; time.textContent = "0:00";
    timer = setInterval(() => {
      const s = (performance.now() - started) / 1000;
      time.textContent = fmt(s);
      if (s >= maxSeconds) stop();
    }, 200);
    const draw = () => { bar.style.width = `${Math.min(100, Math.sqrt(level) * 100)}%`; raf = requestAnimationFrame(draw); };
    draw();
  }

  function stop() {
    if (!ctx || starting) return;
    const rate = ctx.sampleRate;
    release();
    label.textContent = "Record again";
    const samples = chunks.reduce((a, c) => a + c.length, 0), seconds = samples / rate;
    time.textContent = fmt(seconds);
    if (seconds < minSeconds) {
      result.innerHTML = `<div class="notice notice-error"><strong>Too short</strong>Record at least ${minSeconds} seconds.</div>`; return;
    }
    if (peak < 0.01) {
      result.innerHTML = `<div class="notice notice-error"><strong>No sound was picked up</strong>Check your microphone in System Settings → Sound.</div>`;
      return;
    }
    const blob = wavBlob(chunks, rate);
    const d = new Date(), two = (n) => String(n).padStart(2, "0");
    const stamp = `${d.getFullYear()}${two(d.getMonth() + 1)}${two(d.getDate())}-${two(d.getHours())}${two(d.getMinutes())}${two(d.getSeconds())}`;
    url = URL.createObjectURL(blob);
    result.innerHTML = `<audio controls src="${url}"></audio>${clipped > Math.max(20, samples * 0.0005)
      ? `<p class="hint">Too loud. Move back from the microphone and record again.</p>` : ""}`;
    onChange({ blob, seconds, name: `${prefix}-${stamp}.wav` });
  }

  btn.addEventListener("click", () => (starting ? null : ctx ? stop() : start()));
  const handle = { cancel: () => { if (ctx) { starting = false; btn.disabled = false; release(); label.textContent = "Start recording"; time.textContent = "0:00"; } },
    recording: () => !!ctx };
  recorders.add(handle);
  return handle;
}

// ------------------------------------------------------------------ standalone chatterbox
async function renderChatterbox() {
  stopRecorders();
  const st = await api("/api/chatterbox/status");
  const voices = st.voices;
  const voiceOptions = (allowBuiltin) => voices.filter((v) => allowBuiltin || v.id !== "builtin")
    .map((v) => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join("");
  $app.innerHTML = `
    <div class="page-head"><h1>Chatterbox</h1>
      <p class="lead">Make speech in any voice, or re-voice a recording.</p></div>
    ${st.installed ? "" : `<div class="notice notice-error"><strong>Chatterbox is not available</strong>${esc(st.detail)}</div>`}
    <div class="row" style="margin-bottom:16px"><label for="cb-voice-file" class="btn btn-gray" style="margin:0">Add a voice from a file</label>
      <input id="cb-voice-file" type="file" accept=".wav,.mp3,.flac,.m4a,audio/*" class="sr-only">
      <button type="button" class="btn btn-gray" id="cb-voice-rec-open" aria-expanded="false" aria-controls="cb-voice-rec">Record a voice</button></div>
    <section class="card" id="cb-voice-rec" hidden style="margin-bottom:24px"><h2>Record a voice</h2>
      <p class="hint">Read this aloud in a quiet room.</p>
      <blockquote class="read-aloud">Good morning, and thank you for joining me. Today I'd like to walk you through a few simple ideas.
        Each one is easy to follow, and together they make a real difference. Let's take them one at a time.</blockquote>
      <div class="field"><label for="cb-voice-name">Voice name</label>
        <input type="text" id="cb-voice-name" maxlength="40" placeholder="For example: My voice"></div>
      <div class="field">${recorderHTML("voice-rec")}</div>
      <div class="row" style="margin-top:16px"><button type="button" class="btn btn-primary" id="cb-voice-save" disabled>Save voice</button>
        <button type="button" class="btn btn-quiet" id="cb-voice-cancel">Cancel</button></div>
    </section>
    <div class="grid-2">
      <section class="card"><h2>Text to speech</h2>
        <form id="tts-form" style="margin-top:16px">
          <div class="field"><label for="tts-text">Text</label>
            <textarea id="tts-text" rows="5" maxlength="2000" required placeholder="Type what the voice should say."></textarea></div>
          <div class="field"><label for="tts-voice">Voice</label><select id="tts-voice">${voiceOptions(true)}</select></div>
          <details class="more"><summary>More options</summary><div class="options-grid">
            ${slider("tts-exaggeration", "Expressiveness", 0, 1, 0.05, 0.5)}
            ${slider("tts-cfg", "Closeness to the voice", 0, 1, 0.05, 0.5)}
            ${slider("tts-temp", "Variation", 0.1, 1.5, 0.05, 0.8)}
          </div></details>
          <div class="row" style="margin-top:16px"><button class="btn btn-primary" type="submit" id="tts-go">Make speech</button></div>
        </form>
        <div id="tts-out" style="margin-top:16px"></div>
      </section>
      <section class="card"><h2>Voice conversion</h2>
        <form id="vc-form" style="margin-top:16px">
          <div class="field"><span class="label">Recording to re-voice</span>
            <div class="seg" role="radiogroup" aria-label="Where the recording comes from">
              <label class="check"><input type="radio" name="vc-src" value="file" checked> Upload a file</label>
              <label class="check"><input type="radio" name="vc-src" value="mic"> Record with the microphone</label></div>
            <div id="vc-src-file"><input id="vc-file" type="file" accept="audio/*,video/*" aria-label="Recording to re-voice"></div>
            <div id="vc-src-mic" hidden>${recorderHTML("vc-rec")}</div></div>
          <div class="field"><label for="vc-voice">Convert into</label><select id="vc-voice">${voiceOptions(false)}</select>
            ${voices.filter((v) => v.id !== "builtin").length ? "" : `<p class="hint">Add a voice first.</p>`}</div>
          <div class="row" style="margin-top:16px"><button class="btn btn-primary" type="submit" id="vc-go">Convert voice</button></div>
        </form>
        <div id="vc-out" style="margin-top:16px"></div>
      </section>
    </div>
    <section class="card" style="margin-top:24px"><h2>History</h2><div id="cb-history" style="margin-top:16px"></div></section>`;

  const voiceLabel = (id) => (voices.find((v) => v.id === id) || {}).label || id.replace(/\.[^.]+$/, "").replace(/_/g, " ");
  const resultHTML = (r) => `<div class="notice notice-ok">
      <audio controls src="${r.audio_url}"></audio>
      <div class="row" style="margin-top:8px"><a class="btn btn-secondary" href="${r.download_url}" download>Download</a></div></div>`;
  const loadHistory = async () => {
    const items = await api("/api/chatterbox/items");
    document.getElementById("cb-history").innerHTML = items.length ? `<div class="table-wrap"><table><thead><tr><th>What</th><th>Voice</th><th>Listen</th><th></th></tr></thead><tbody>${
      items.map((r) => `<tr><td class="small"><div class="prompt-cell">${r.kind === "tts" ? esc(r.text) : "Voice conversion"}</div></td><td class="small">${esc(voiceLabel(r.voice))}</td>
        <td style="min-width:220px"><audio controls preload="none" src="${r.audio_url}"></audio></td>
        <td><a class="btn btn-quiet" href="${r.download_url}" download>Download</a></td></tr>`).join("")}</tbody></table></div>`
      : `<p class="muted">Nothing yet.</p>`;
  };
  loadHistory();
  document.querySelectorAll("#tts-form input[type=range]").forEach((r) => r.addEventListener("input", () => {
    document.querySelector(`[data-out="${r.id}"]`).textContent = Number(r.value).toFixed(2);
  }));
  const addVoice = async (blob, filename) => {
    const fd = new FormData(); fd.append("file", blob, filename);
    const res = await fetch("/api/voices", { method: "POST", body: fd }); const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    toast(`Voice added: ${data.label}`); renderChatterbox();
  };
  document.getElementById("cb-voice-file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    try { await addVoice(f, f.name); } catch (e) { toast(e.message, true); }
  });

  // record a voice
  const recPanel = document.getElementById("cb-voice-rec"), recOpen = document.getElementById("cb-voice-rec-open");
  const saveBtn = document.getElementById("cb-voice-save");
  let voiceTake = null;
  const voiceRec = mountRecorder("voice-rec", { maxSeconds: 30, minSeconds: 5, prefix: "voice",
    onChange: (take) => { voiceTake = take; saveBtn.disabled = !take; } });
  const showPanel = (open) => {
    recPanel.hidden = !open; recOpen.setAttribute("aria-expanded", String(open));
    if (open) document.getElementById("cb-voice-name").focus(); else voiceRec.cancel();
  };
  recOpen.addEventListener("click", () => showPanel(recPanel.hidden));
  document.getElementById("cb-voice-cancel").addEventListener("click", () => showPanel(false));
  saveBtn.addEventListener("click", () => busy(saveBtn, async () => {
    const name = document.getElementById("cb-voice-name").value.trim().replace(/[^A-Za-z0-9 _-]/g, "").replace(/\s+/g, "_");
    if (!name) { toast("Give the voice a name first.", true); document.getElementById("cb-voice-name").focus(); return; }
    if (!voiceTake) { toast("Record the passage first.", true); return; }
    await addVoice(voiceTake.blob, `${name}.wav`);
  }));

  // conversion source: a file or a microphone recording
  let vcTake = null;
  const vcRec = mountRecorder("vc-rec", { maxSeconds: 300, minSeconds: 1, onChange: (take) => (vcTake = take) });
  document.querySelectorAll('input[name="vc-src"]').forEach((r) => r.addEventListener("change", () => {
    const mic = document.querySelector('input[name="vc-src"]:checked').value === "mic";
    document.getElementById("vc-src-file").hidden = mic; document.getElementById("vc-src-mic").hidden = !mic;
    if (!mic) vcRec.cancel();
  }));
  document.getElementById("tts-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("tts-go"), async () => {
      const out = document.getElementById("tts-out");
      out.innerHTML = `<div class="status-line"><span class="pill pill-active">Making speech</span></div>`;
      try {
        const r = await api("/api/chatterbox/tts", { method: "POST", body: {
          text: document.getElementById("tts-text").value, voice: document.getElementById("tts-voice").value,
          exaggeration: Number(document.getElementById("tts-exaggeration").value), cfg_weight: Number(document.getElementById("tts-cfg").value),
          temperature: Number(document.getElementById("tts-temp").value), check_words: false } });
        out.innerHTML = resultHTML(r); loadHistory();
      } catch (e) { out.innerHTML = `<div class="notice notice-error"><strong>Could not make speech</strong>${esc(e.message)}</div>`; }
    });
  });
  document.getElementById("vc-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("vc-go"), async () => {
      const out = document.getElementById("vc-out");
      const mic = document.querySelector('input[name="vc-src"]:checked').value === "mic";
      const fd = new FormData();
      if (mic) {
        if (vcRec.recording()) { toast("Stop the recording first.", true); return; }
        if (!vcTake) { toast("Record something first.", true); return; }
        fd.append("file", vcTake.blob, vcTake.name);
      } else {
        const f = document.getElementById("vc-file").files[0];
        if (!f) { toast("Choose a recording first.", true); return; }
        fd.append("file", f);
      }
      fd.append("voice_id", document.getElementById("vc-voice").value);
      out.innerHTML = `<div class="status-line"><span class="pill pill-active">Converting</span></div>`;
      try {
        const res = await fetch("/api/chatterbox/convert", { method: "POST", body: fd }); const r = await res.json();
        if (!res.ok) throw new Error(r.detail || "Conversion failed");
        out.innerHTML = resultHTML(r); loadHistory();
      } catch (e) { out.innerHTML = `<div class="notice notice-error"><strong>Could not convert</strong>${esc(e.message)}</div>`; }
    });
  });
}

// ------------------------------------------------------------------ long talking head
const SAMPLE_SCRIPT = "Hello everyone, and thank you for joining me today. Let's talk about your health. " +
  "At Prudential Health India, we believe good health starts with clear and honest information. " +
  "Every day, families face choices about care, cover and well-being. Those choices can feel complicated. " +
  "Our job is to make them simpler. We explain your options in plain, everyday language. " +
  "We help you compare plans with confidence, and we stay with you when it matters most. " +
  "Small steps make a big difference. A regular check-up. A healthier routine. A little more rest. " +
  "And a plan that protects the people you love. You don't have to figure it out alone. " +
  "Our team is ready to listen, to answer your questions, and to help you choose well. " +
  "Thank you for trusting us with your health and with the health of your family. " +
  "Take care of yourself, and take care of your loved ones. Stay informed and stay well.";

const minsec = (x) => x >= 60 ? `${Math.floor(x / 60)} min ${Math.round(x % 60)} s` : `${Math.round(x)} s`;

async function renderLongCreate() {
  const body = document.getElementById("create-body");
  let ready, system, models;
  try {
    [ready, system, models] = await Promise.all([api("/api/strict/readiness"), api("/api/system"), api("/api/models")]);
  } catch (e) {
    body.innerHTML = `<div class="notice notice-error"><strong>Not available</strong>${esc(e.message)}</div>`;
    return;
  }
  const voices = system.voices.filter((v) => v.id !== "builtin");
  const refModels = models.models.filter((m) => m.supports_reference_images);
  const notReady = [["Chatterbox", ready.chatterbox], ["Speech check", ready.speech_check], ["Face analysis", ready.face_analysis]]
    .filter(([, r]) => !r.ready);
  body.innerHTML = `
    ${notReady.length ? `<div class="notice notice-error" style="margin-bottom:24px"><strong>Not ready</strong>${notReady.map(([n, r]) => `${n}: ${esc(r.detail)}`).join("<br>")}</div>` : ""}
    <form id="long-form">
      <div class="field"><label for="l-photo">Photo of the person (required)</label>
        <input id="l-photo" type="file" accept="image/png,image/jpeg,image/webp" required>
        <div id="l-preview" class="row" style="margin-top:8px"></div>
        <label class="check" style="margin-top:8px"><input type="radio" name="l-ref" value="full" checked> Full photo with background</label>
        <label class="check"><input type="radio" name="l-ref" value="face_crop"> Face and shoulders only</label></div>
      <div class="field"><label for="l-script">Script (required)</label>
        <textarea id="l-script" rows="9" maxlength="4000" required placeholder="The full script, about 150 words for one minute."></textarea>
        <div class="row" style="margin-top:8px"><button type="button" class="btn btn-quiet" id="l-sample">Use a sample script</button>
          <span class="muted small" id="l-count"></span></div></div>
      <div class="field"><label for="l-voice">Authorised voice (required)</label>
        <select id="l-voice" required><option value="">Choose a voice…</option>${voices.map((v) => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join("")}</select>
        ${voices.length ? "" : `<p class="hint">No voices yet. Add one on the Chatterbox page first.</p>`}</div>
      <div class="field"><label for="l-model">Video model</label>
        <select id="l-model">${refModels.map((m) => `<option value="${esc(m.id)}" ${m.id === "veo-3.1-fast-generate-001" ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select></div>
      <div class="row" style="margin-top:16px"><button type="button" class="btn btn-secondary" id="l-plan">Plan the clips</button></div>
      <div id="l-plan-out" style="margin-top:16px"></div>
      <div class="field" style="margin-top:24px"><label class="check"><input type="checkbox" id="l-consent" required>
        I confirm the person in the photo and the owner of the voice have authorised this AI-generated video.</label>
        <input id="l-by" type="text" placeholder="Your name" style="margin-top:8px" required></div>
      <div class="row" style="margin-top:24px"><button class="btn btn-primary" type="submit" id="l-go">Create the video</button>
        <span class="muted small">Nothing is generated until you approve it on the next page.</span></div>
    </form>`;
  let photo = null;
  const scriptEl = document.getElementById("l-script");
  const count = () => {
    const n = scriptEl.value.trim() ? scriptEl.value.trim().split(/\s+/).length : 0;
    document.getElementById("l-count").textContent = n ? `${n} words · about ${minsec(n / 2.6)}` : "";
  };
  scriptEl.addEventListener("input", count);
  document.getElementById("l-sample").addEventListener("click", () => { scriptEl.value = SAMPLE_SCRIPT; count(); });
  const refMode = () => document.querySelector('input[name="l-ref"]:checked').value;
  document.getElementById("l-photo").addEventListener("change", async (ev) => {
    const f = ev.target.files[0]; if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    try {
      const res = await fetch("/api/uploads/image", { method: "POST", body: fd }); const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      photo = data;
      document.getElementById("l-preview").innerHTML = `<img src="${URL.createObjectURL(f)}" alt="Uploaded photo" style="height:96px;border-radius:var(--r-sm);border:1px solid var(--gray-border)">`;
    } catch (e) { photo = null; ev.target.value = ""; toast(e.message, true); }
  });
  document.getElementById("l-plan").addEventListener("click", (ev) => busy(ev.currentTarget, async () => {
    const out = document.getElementById("l-plan-out");
    try {
      const r = await api("/api/long/plan", { method: "POST", body: { script: scriptEl.value, reference_mode: refMode(),
        voice: document.getElementById("l-voice").value || null } });
      out.innerHTML = `<div class="notice"><strong>${r.clips} clips · about ${minsec(r.est_seconds)} · uses ${r.generations} generations</strong></div>
        <div class="table-wrap" style="margin-top:12px"><table><thead><tr><th>#</th><th>Words</th><th>Shot</th></tr></thead><tbody>${
        r.segments.map((x) => `<tr><td>${x.index + 1}</td><td class="small">${esc(x.text)}
            <details class="small" style="margin-top:4px"><summary class="muted">Prompt</summary><pre class="json">${esc(x.prompt)}</pre></details></td>
          <td class="small">${esc(x.summary.framing)}<div class="muted">${esc(x.summary.actions.join(", "))}</div></td></tr>`).join("")}</tbody></table></div>`;
    } catch (e) { out.innerHTML = `<div class="notice notice-error">${esc(e.message)}</div>`; }
  }));
  document.getElementById("long-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("l-go"), async () => {
      if (!photo) throw new Error("Upload a photo of the person first.");
      const p = await api("/api/long/projects", { method: "POST", body: {
        reference_id: photo.id, script: scriptEl.value, voice: document.getElementById("l-voice").value,
        model: document.getElementById("l-model").value, reference_mode: refMode(),
        consent: { attested: document.getElementById("l-consent").checked, by: document.getElementById("l-by").value } } });
      location.hash = `#/project/${p.id}`;
    });
  });
}

let longTimer = null;
async function renderLong(id, keepScroll = false) {
  clearTimeout(longTimer);
  const [v, system] = await Promise.all([api(`/api/long/${encodeURIComponent(id)}`), api("/api/system")]);
  const y = window.scrollY;
  const gate = system.provider_calls;
  const rows = v.segments.map((s) => {
    const busyClip = s.active || ["IDLE", "VALIDATING_INPUT", "PREPARING_REFERENCE"].includes(s.state);
    return `<tr><td>${s.index + 1}</td>
      <td class="small"><div class="prompt-cell">${esc(s.text)}</div><div class="muted">${esc(s.summary.framing)} · ${esc(s.summary.actions.join(", "))}</div></td>
      <td>${s.state === "AWAITING_AUDIO_APPROVAL" ? `<span class="pill">Ready to generate</span>` : s.ready ? `<span class="pill pill-done">Done</span>` : pill(s.state)}${s.active && s.message ? `<div class="small muted" style="margin-top:4px">${esc(s.message)}</div>` : ""}
        ${s.state === "FAILED" ? `<div class="small" style="color:var(--red-dark);margin-top:4px">${esc(s.error || "")}</div>` : ""}
        ${s.failed_checks && s.failed_checks.length ? `<div class="small" style="color:var(--red-dark);margin-top:4px">Failed: ${esc(s.failed_checks.join(", "))}</div>` : ""}</td>
      <td style="min-width:200px">${s.final_url ? `<video controls preload="none" src="${s.final_url}" style="width:200px;border-radius:var(--r-sm)"></video>` : ""}</td>
      <td>${!busyClip && !v.running && s.state !== "AWAITING_AUDIO_APPROVAL" ? `<button class="btn btn-quiet" data-redo="${s.index}">Redo</button>` : ""}</td></tr>`;
  }).join("");
  const n = v.clips_to_generate;
  const stitch = v.stitch;
  const failedRows = stitch ? stitch.results.filter((r) => r.status === "FAIL") : [];
  $app.innerHTML = `<div class="page-head"><div class="row"><a class="btn btn-quiet" href="#/projects">← Projects</a></div>
    <div class="row" style="margin-top:8px"><h1 style="margin:0">Long talking head</h1>${pill(v.state)}</div>
    <p class="muted small" style="margin:8px 0 0">${v.clips} clips · about ${minsec(v.est_seconds)} · ${v.clips_ready} ready</p></div>
  <div class="stack">
    ${v.error ? `<div class="notice notice-error">${esc(v.error)}</div>` : ""}
    ${v.running ? `<div class="card"><div class="status-line"><span class="pill pill-active">Working</span><span>${esc(v.message || "")}</span></div><div class="progress"><span></span></div>
      ${v.state === "LONG_GENERATING" ? `<button class="btn btn-gray" id="lg-stop" style="margin-top:16px">Stop after this clip</button>` : ""}</div>` : ""}
    <section class="card"><h2>Clips</h2>
      <div class="table-wrap" style="margin-top:16px"><table><thead><tr><th>#</th><th>Words</th><th>Status</th><th>Clip</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      ${n && !v.running ? `<div class="field" style="margin-top:24px"><label for="lg-name">Approve as</label><input id="lg-name" type="text" placeholder="Your name"></div>
        <div class="row" style="margin-top:12px"><button class="btn btn-primary" id="lg-go" ${gate ? "" : "disabled"}>Generate ${n} clip${n === 1 ? "" : "s"} (uses ${n} generation${n === 1 ? "" : "s"})</button>
          ${gate ? "" : `<span class="muted small">Video generation is switched off.</span>`}</div>` : ""}
    </section>
    ${v.clips_ready ? `<section class="card"><h2>Voice</h2>
      <p class="hint" style="margin-top:8px">Match every clip's pitch to:</p>
      <label class="check" style="margin-top:8px"><input type="radio" name="lg-target" value="voice" ${v.anchor_source === "voice" ? "checked" : ""} ${v.running ? "disabled" : ""}>
        Your voice recording${v.anchor_source === "voice" && v.anchor ? ` <span class="muted small">(${Math.round(v.anchor.pitch_hz)} Hz)</span>` : ""}</label>
      <label class="check"><input type="radio" name="lg-target" value="clip1" ${v.anchor_source === "clip1" ? "checked" : ""} ${v.running ? "disabled" : ""}>
        Clip 1${v.anchor_source === "clip1" && v.anchor ? ` <span class="muted small">(${Math.round(v.anchor.pitch_hz)} Hz)</span>` : ""}</label>
      ${v.can_match ? `<div class="row" style="margin-top:12px"><button class="btn btn-secondary" id="lg-match">Match voices</button>
        <span class="muted small">Re-converts the clips. Uses no generations.</span></div>` : ""}</section>` : ""}
    <section class="card"><h2>Final video</h2>
      ${v.final_url ? `<div class="media-box" style="margin-top:16px"><video controls preload="metadata" src="${v.final_url}?v=${esc(String(stitch.at))}"></video></div>
        <div class="row" style="margin-top:12px"><a class="btn btn-primary" href="${v.final_url}?download=true" download>Download MP4</a>
          <span class="muted small">${minsec(stitch.seconds)}${stitch.draft ? " · draft" : ""}${v.state === "LIKENESS_ACCEPTED" ? " · accepted as a likeness" : ""}</span></div>
        ${failedRows.length ? `<div class="notice notice-error" style="margin-top:12px"><strong>Some checks failed</strong>${failedRows.map((r) => esc(r.label)).join("<br>")}</div>` : ""}`
        : `<p class="muted">${v.clips_ready === v.clips ? "All clips are ready." : "Appears when every clip is ready."}</p>`}
      ${v.can_stitch ? `<div class="row" style="margin-top:16px">
          ${v.clips_with_failed_checks.length ? `<button class="btn btn-secondary" id="lg-draft">Cut together as a draft</button>
            <span class="muted small">Clip ${v.clips_with_failed_checks.join(", ")} failed a check. Redo ${v.clips_with_failed_checks.length === 1 ? "it" : "them"} for a final version.</span>`
          : `<button class="btn ${v.final_url ? "btn-secondary" : "btn-primary"}" id="lg-stitch">${v.final_url ? "Cut together again" : "Cut the clips together"}</button>`}</div>` : ""}
      ${v.state === "NEEDS_REVIEW" && stitch && !stitch.draft ? `<h3 style="margin-top:24px">Accept this as a likeness of the person in the photo?</h3>
        <div class="field"><label for="lg-reviewer">Reviewer</label><input id="lg-reviewer" type="text" placeholder="Your name"></div>
        ${failedRows.length === 1 && failedRows[0].key === "script" ? `<label class="check small" style="margin-top:8px"><input type="checkbox" id="lg-override"> I listened: the words are right; the speech check misheard.</label>` : ""}
        <div class="row" style="margin-top:12px"><button class="btn btn-primary" id="lg-accept">Accept as a likeness</button></div>` : ""}
    </section>
  </div>`;
  if (keepScroll) window.scrollTo(0, y);
  const on = (sel, fn) => { const el = document.querySelector(sel); if (el) el.addEventListener("click", () => busy(el, fn)); };
  const again = () => renderLong(id, true);
  on("#lg-go", async () => {
    await api(`/api/long/${id}/generate`, { method: "POST", body: { reviewer: document.getElementById("lg-name").value, generations: n } });
    again();
  });
  document.querySelectorAll('input[name="lg-target"]').forEach((r) => r.addEventListener("change", async () => {
    try { await api(`/api/long/${id}/voice-target`, { method: "POST", body: { source: r.value } }); } catch (e) { toast(e.message, true); }
    again();
  }));
  on("#lg-match", async () => { await api(`/api/long/${id}/match-voices`, { method: "POST" }); again(); });
  on("#lg-stop", async () => { await api(`/api/long/${id}/stop`, { method: "POST" }); again(); });
  on("#lg-stitch", async () => { await api(`/api/long/${id}/stitch`, { method: "POST" }); again(); });
  on("#lg-draft", async () => { await api(`/api/long/${id}/stitch?draft=true`, { method: "POST" }); again(); });
  on("#lg-accept", async () => {
    const o = document.getElementById("lg-override");
    await api(`/api/long/${id}/accept`, { method: "POST", body: { reviewer: document.getElementById("lg-reviewer").value, override: o ? o.checked : false } });
    again();
  });
  document.querySelectorAll("[data-redo]").forEach((b) => b.addEventListener("click", () => busy(b, async () => {
    const i = Number(b.dataset.redo);
    if (!confirm(`Make a new take of clip ${i + 1}? Generating it will use 1 generation.`)) return;
    await api(`/api/long/${id}/segments/${i}/redo`, { method: "POST" });
    again();
  })));
  // keep the page current while anything is moving
  const moving = v.running || v.segments.some((s) => s.active || ["IDLE", "VALIDATING_INPUT", "PREPARING_REFERENCE"].includes(s.state));
  if (moving) {
    const sig = JSON.stringify([v.state, v.message, v.segments.map((s) => [s.state, s.message, s.final_url])]);
    longTimer = setTimeout(async function tick() {
      if (!location.hash.startsWith(`#/project/${id}`)) return;
      let fresh;
      try { fresh = await api(`/api/long/${encodeURIComponent(id)}`); } catch { longTimer = setTimeout(tick, 2500); return; }
      const now = JSON.stringify([fresh.state, fresh.message, fresh.segments.map((s) => [s.state, s.message, s.final_url])]);
      if (now !== sig) return renderLong(id, true);
      longTimer = setTimeout(tick, 2500);
    }, 2500);
  }
}

// ------------------------------------------------------------------ strict talking head
async function renderStrict() {
  const body = document.getElementById("create-body");
  let ready, system, models, dirOpts;
  try {
    [ready, system, models, dirOpts] = await Promise.all([api("/api/strict/readiness"), api("/api/system"), api("/api/models"),
      api("/api/direction/options")]);
  } catch (e) {
    body.innerHTML = `<div class="notice notice-error"><strong>Strict Talking Head is not available</strong>${esc(e.message)}</div>
      <p style="margin-top:16px"><a class="btn btn-secondary" href="#/settings">Open Settings</a></p>`;
    return;
  }
  const voices = system.voices.filter((v) => v.id !== "builtin");
  const refModels = models.models.filter((m) => m.supports_reference_images);
  const notReady = [["Chatterbox", ready.chatterbox], ["Speech check", ready.speech_check], ["Face analysis", ready.face_analysis],
    ...(ready.provider_calls ? [["Video API", ready.video_api]] : [])].filter(([, r]) => !r.ready);   // gate off: noted by the button
  body.innerHTML = `
    ${notReady.length ? `<div class="notice notice-error" style="margin-bottom:24px"><strong>Not ready</strong>${notReady.map(([n, r]) => `${n}: ${esc(r.detail)}`).join("<br>")}</div>` : ""}
    <form id="strict-form">
      <div class="field"><label for="s-photo">Photo of the person (required)</label>
        <input id="s-photo" type="file" accept="image/png,image/jpeg,image/webp" required>
        <div id="s-preview" class="row" style="margin-top:8px"></div>
        <p class="hint">One person, facing the camera.</p>
        <label class="check" style="margin-top:8px"><input type="radio" name="s-ref" value="full" checked> Full photo with background</label>
        <label class="check"><input type="radio" name="s-ref" value="face_crop"> Face and shoulders only</label></div>
      <div class="field"><label for="s-script">Approved dialogue (required)</label>
        <textarea id="s-script" rows="3" maxlength="400" required placeholder="Prudential Health India helps people make informed choices for their health and well-being."></textarea>
        <p class="hint">About 18 words fit in one clip.</p></div>
      <div class="field"><label for="s-voice">Authorised voice (required)</label>
        <select id="s-voice" required><option value="">Choose a voice…</option>${voices.map((v) => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join("")}</select>
        ${voices.length ? "" : `<p class="hint">No voices yet. Add one on the Chatterbox page first.</p>`}</div>
      <div class="field"><label for="s-model">Video model</label>
        <select id="s-model">${refModels.map((m) => `<option value="${esc(m.id)}" ${m.id === "veo-3.1-fast-generate-001" ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select></div>
      <div class="field"><span class="label">How the voice is made</span>
        <label class="check"><input type="radio" name="s-method" value="convert" checked> Convert the video's own audio</label>
        <label class="check"><input type="radio" name="s-method" value="tts"> Text to speech first</label></div>
      <div class="field" id="s-audio-field" hidden><span class="label">Sound from the video model</span>
        <label class="check"><input type="radio" name="s-audio" value="off" checked> Off</label>
        <label class="check"><input type="radio" name="s-audio" value="reference"> Used only for timing</label></div>
      <details class="more" open><summary>Video direction</summary><div>
        <div class="options-grid">
          <div><label for="d-preset">Motion preset</label><select id="d-preset">${Object.entries(dirOpts.presets).map(([k, l]) => `<option value="${k}" ${k === "natural_human" ? "selected" : ""}>${esc(l)}</option>`).join("")}</select></div>
          <div><label for="d-head">Head movement</label><select id="d-head"><option value="minimal">Minimal</option><option value="low">Low</option><option value="moderate">Moderate</option></select></div>
          <div><label for="d-face">Facial expression</label><select id="d-face"><option value="subtle">Subtle</option><option value="warm">Warm</option><option value="expressive">Expressive</option></select></div>
          <div><label for="d-camera">Camera</label><select id="d-camera"><option value="locked">Locked</option><option value="slow_push_in">Slow push-in (prompt only)</option></select></div>
          <div><label for="d-realism">Visual realism priority</label><select id="d-realism"><option value="high">High</option><option value="standard">Standard</option></select></div>
          <div><label for="d-provider">Provider</label><select id="d-provider">${Object.entries(dirOpts.providers).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("")}</select></div>
        </div>
        <button class="btn btn-secondary" type="button" id="d-preview" style="margin-top:16px">Preview request</button>
        <div id="d-out" style="margin-top:12px"></div>
      </div></details>
      <div class="field"><label class="check"><input type="checkbox" id="s-consent" required>
        I confirm the person in the photo and the owner of the voice have authorised this AI-generated video.</label>
        <input id="s-by" type="text" placeholder="Your name" style="margin-top:8px" required></div>
      <div class="row" style="margin-top:24px"><button class="btn btn-primary" type="submit" id="s-go" ${ready.ready ? "" : "disabled"}>Check photo and script</button>
        ${ready.provider_calls ? "" : `<span class="muted small">Video generation is switched off.</span>`}</div>
    </form>`;
  const presetValues = (key) => {
    const p = (dirOpts.preset_values[key] || {}).performance || {};
    return { head: p.head_motion || "minimal", face: p.facial_expression || "subtle" };
  };
  const applyPreset = () => {
    const v = presetValues(document.getElementById("d-preset").value);
    document.getElementById("d-head").value = v.head; document.getElementById("d-face").value = v.face;
  };
  window.directionInput = () => {
    const preset = document.getElementById("d-preset").value;
    const head = document.getElementById("d-head").value, face = document.getElementById("d-face").value;
    const camera = document.getElementById("d-camera").value, realism = document.getElementById("d-realism").value;
    const overrides = { performance: { head_motion: head, facial_expression: face,
      motion_intensity: head === "moderate" || face === "expressive" ? "medium" : "low" },
      camera: { movement: camera }, visual_constraints: { realism_priority: realism } };
    if (camera !== "locked") overrides.scene = { composition: "medium_close_up" };   // moving camera cannot keep the reference framing
    return { preset, overrides };
  };
  document.getElementById("d-preset").addEventListener("change", applyPreset);
  ["d-head", "d-face"].forEach((id) => document.getElementById(id).addEventListener("change", () => {
    const v = presetValues(document.getElementById("d-preset").value);
    if (document.getElementById("d-head").value !== v.head || document.getElementById("d-face").value !== v.face)
      document.getElementById("d-preset").value = "custom";
  }));
  applyPreset();
  document.getElementById("d-preview").addEventListener("click", (ev) => busy(ev.currentTarget, async () => {
    const out = document.getElementById("d-out");
    try {
      const r = await api("/api/direction/preview", { method: "POST", body: {
        provider: document.getElementById("d-provider").value, script: document.getElementById("s-script").value,
        reference_id: photo ? photo.id : null, reference_mode: document.querySelector('input[name="s-ref"]:checked').value,
        voice_method: document.querySelector('input[name="s-method"]:checked').value, direction: directionInput() } });
      const pre = (t, x) => `<span class="label" style="margin-top:12px">${t}</span><pre class="json">${esc(typeof x === "string" ? x : JSON.stringify(x, null, 2))}</pre>`;
      out.innerHTML = `<div class="notice ${r.generation_possible ? "" : "notice-warn"}"><strong>${esc(r.mode)}</strong>
          ${r.generation_possible ? "This request can be generated after approval." : `Generation not possible: ${esc(r.why_not)}.`}
          ${r.request.unsupported.length ? `<br><span class="small">Not sent (unverified/unsupported): ${r.request.unsupported.map((u) => esc(u.field)).join(", ")}</span>` : ""}
          ${r.request.notes.length ? `<br><span class="small">${r.request.notes.map(esc).join(" ")}</span>` : ""}</div>`
        + pre("1. Creative direction (internal JSON)", r.direction) + pre("2. Provider request (" + r.request.endpoint + ")", r.request.body)
        + pre("3. Compiled prompt", r.prompt) + (r.negative_prompt ? pre("Negative prompt", r.negative_prompt) : "");
    } catch (e) { out.innerHTML = `<div class="notice notice-error"><strong>Video direction</strong>${esc(e.message)}</div>`; }
  }));
  let photo = null;
  const fileEl = document.getElementById("s-photo");
  document.querySelectorAll('input[name="s-method"]').forEach((r) => r.addEventListener("change", () => {
    const tts = document.querySelector('input[name="s-method"]:checked').value === "tts";
    document.getElementById("s-audio-field").hidden = !tts;
    document.querySelector("#s-go").textContent = tts ? "Prepare speech" : "Check photo and script";
  }));
  fileEl.addEventListener("change", async () => {
    const f = fileEl.files[0];
    if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    try {
      const res = await fetch("/api/uploads/image", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      photo = data;
      document.getElementById("s-preview").innerHTML = `<img src="${URL.createObjectURL(f)}" alt="Uploaded photo" style="height:96px;border-radius:var(--r-sm);border:1px solid var(--gray-border)">
        <span class="small muted">${esc(data.name)} · ${data.width}×${data.height}</span>`;
    } catch (e) { photo = null; fileEl.value = ""; toast(e.message, true); }
  });
  document.getElementById("strict-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    busy(document.getElementById("s-go"), async () => {
      if (!photo) throw new Error("Upload a photo of the person first.");
      const p = await api("/api/strict/projects", { method: "POST", body: {
        reference_id: photo.id, script: document.getElementById("s-script").value, voice: document.getElementById("s-voice").value,
        veo_audio: document.querySelector('input[name="s-audio"]:checked').value,
        voice_method: document.querySelector('input[name="s-method"]:checked').value,
        reference_mode: document.querySelector('input[name="s-ref"]:checked').value,
        provider: document.getElementById("d-provider").value, direction: directionInput(),
        model: document.getElementById("s-model").value,
        consent: { attested: document.getElementById("s-consent").checked, by: document.getElementById("s-by").value },
      } });
      location.hash = `#/project/${p.id}`;
    });
  });
}

const STRICT_STAGES = [
  ["Validating input", ["VALIDATING_INPUT"]], ["Preparing reference image", ["PREPARING_REFERENCE"]],
  ["Generating approved speech", ["GENERATING_SPEECH"]], ["Validating dialogue", ["VALIDATING_DIALOGUE", "AWAITING_AUDIO_APPROVAL"]],
  ["Animating talking head", ["GENERATING_VIDEO"]], ["Synchronising audio and video", ["SYNCHRONIZING", "COMPOSING", "VALIDATING_LIPSYNC"]],
  ["Validating identity", ["VALIDATING_IDENTITY"]], ["Validating final output", ["VALIDATING_OUTPUT"]], ["Ready for preview", ["NEEDS_REVIEW", "COMPLETED", "LIKENESS_ACCEPTED"]],
];

function strictStage(p) {
  const s = [S.FAILED, S.CANCELLED].includes(p.state) ? p.failed_stage : p.state;
  const i = STRICT_STAGES.findIndex(([, states]) => states.includes(s));
  return i < 0 ? 0 : i;
}

function wordDiff(d) {
  if (!d) return "";
  const list = (label, xs) => xs && xs.length ? `<div class="small"><b>${label}:</b> ${xs.map(esc).join(", ")}</div>` : "";
  return `<div class="row small" style="gap:16px;margin:8px 0"><span class="pill ${d.passed ? "pill-done" : "pill-failed"}">${d.passed ? "Matches" : "Does not match"}</span>
      <span>Word match <b>${Math.round(d.word_accuracy * 100)}%</b></span><span>Character match <b>${Math.round(d.char_similarity * 100)}%</b></span></div>
    ${(d.substituted || []).map((x) => `<div class="small"><b>Heard “${esc(x.heard)}” for “${esc(x.approved)}”</b>${x.note ? ` · ${esc(x.note)}` : ""}</div>`).join("")}
    ${list("Missing", d.missing)}${list("Extra", d.extra)}${list("Out of order", d.reordered)}${list("Repeated", d.duplicated)}
    ${d.reasons && d.reasons.length ? `<div class="small" style="color:var(--red-dark)">${d.reasons.map(esc).join("; ")}</div>` : ""}`;
}

function strictHTML(p, system) {
  const st = p.params.strict || {};
  const f = p.files;
  const cur = strictStage(p);
  const failed = [S.FAILED, S.CANCELLED].includes(p.state);
  const done = [S.COMPLETED, "NEEDS_REVIEW", "LIKENESS_ACCEPTED"].includes(p.state);
  const v = st.validation;
  const refFile = (p.params.reference_files || [])[0];
  const refImg = refFile ? `/api/projects/${esc(p.id)}/reference/${encodeURIComponent(refFile.path.split("/").pop())}` : "";
  const stages = STRICT_STAGES.map(([label], i) => {
    const status = done ? "done" : i < cur ? "done" : i === cur ? (failed ? "failed" : "current") : "waiting";
    const icon = status === "done" ? "✓" : status === "failed" ? "!" : i + 1;
    return `<li class="stage stage-${status}"><span class="stage-dot" aria-hidden="true">${icon}</span><span>${label}</span>
      <span class="sr-only">${status}</span></li>`;
  }).join("");
  const d = st.dialogue;
  const converting = st.voice_method === "convert";
  const approveBlock = p.state === "AWAITING_AUDIO_APPROVAL" && converting ? `
    <div class="field" style="margin-top:16px"><label for="a-name">Approve as</label><input id="a-name" type="text" placeholder="Your name"></div>
    <div class="row" style="margin-top:16px"><button class="btn btn-primary" id="approve-btn" ${system.provider_calls ? "" : "disabled"}>Generate the video (uses 1 generation)</button></div>
    ${system.provider_calls ? "" : `<p class="hint">Video generation is switched off.</p>`}`
    : p.state === "AWAITING_AUDIO_APPROVAL" ? `
    <div class="field" style="margin-top:16px"><label for="a-name">Approve as</label><input id="a-name" type="text" placeholder="Your name"></div>
    ${d && !d.passed ? `<label class="check small" style="margin-top:8px"><input type="checkbox" id="a-override"> The recogniser is wrong; I listened and the speech says the approved words exactly.</label>` : ""}
    <div class="row" style="margin-top:16px"><button class="btn btn-primary" id="approve-btn" ${system.provider_calls ? "" : "disabled"}>Approve audio and generate video (uses 1 generation)</button>
      <button class="btn btn-secondary" id="regen-speech-btn">Generate the speech again</button></div>` : "";
  const table = (v && v.results) || (st.audit && st.audit.rows) || null;
  const failedChecks = table ? table.filter((r) => r.status === "FAIL").map((r) => r.label)
    : v ? Object.entries(v.checks).filter(([, ok]) => ok === false).map(([k]) => k.replace(/_/g, " ")) : [];
  const frames = Object.keys(f).filter((k) => /^frame_\d{2}\.png$/.test(k));
  const ver = `v=${(v && v.run_id) || st.run_id || st.take || 0}`;  // a re-run replaces files: never reuse cached bytes
  return `<div class="page-head"><div class="row"><a class="btn btn-quiet" href="#/projects">← Projects</a></div>
    <div class="row" style="margin-top:8px"><h1 style="margin:0">Strict talking head</h1>${pill(p.state)}<span class="spacer"></span>
      ${p.active ? `<button class="btn btn-gray" id="cancel-btn">Cancel</button>` : ""}
      ${failed && ["GENERATING_SPEECH", "VALIDATING_DIALOGUE"].includes(p.failed_stage) ? `<button class="btn btn-primary" id="regen-speech-btn">Generate the speech again</button>`
        : failed ? `<button class="btn btn-primary" id="retry-btn">Retry</button>` : ""}</div>
    <p class="muted small" style="margin:8px 0 0">Created ${esc(when(p.created_at))} · ${esc(p.model)} · voice ${esc(st.voice || "")}</p></div>
  <div class="stack">
    ${p.state === S.FAILED ? `<div class="notice notice-error"><strong>Stopped at: ${esc((STRICT_STAGES[cur] || [""])[0])}</strong>${esc(p.error)}</div>` : ""}
    ${p.active ? `<div class="card">${live(p)}</div>` : ""}
    <section class="card"><h2 style="margin-bottom:16px">Progress</h2><ol class="stages">${stages}</ol></section>

    <section class="card"><h2>Photo</h2>
      <div class="row" style="margin-top:16px;align-items:flex-start">
        ${refImg ? `<figure style="margin:0"><img src="${refImg}" alt="Image sent to the video model" style="height:160px;border-radius:var(--r-sm);border:1px solid var(--gray-border)"></figure>` : ""}
        ${refImg ? "" : `<p class="muted">Checked in step 2.</p>`}
      </div></section>

    ${converting ? `<section class="card"><h2>Voice conversion</h2>
      <div style="margin-top:16px"><span class="label">Approved script</span><p style="margin:0">${esc(st.script || "")}</p></div>
      ${approveBlock}</section>` : `
    <section class="card"><h2>Approved speech</h2>
      ${st.script ? `<div class="grid-2" style="margin-top:16px"><div><span class="label">Approved script</span><p style="margin:0">${esc(st.script)}</p></div>
        <div><span class="label">What the speech check heard${d ? ` (take ${d.take})` : ""}</span><p style="margin:0">${d ? esc(d.transcript) : "—"}</p></div></div>` : ""}
      ${wordDiff(d)}
      ${f["tts.wav"] ? `<span class="label" style="margin-top:16px">Generated speech · ${d ? d.speech_seconds.toFixed(2) + " s" : ""}</span><audio controls preload="metadata" src="${f["tts.wav"]}?t=${st.take || 0}"></audio>` : ""}
      ${st.audio_approval ? `<p class="hint">Approved by ${esc(st.audio_approval.by)}${st.audio_approval.override ? " (override: the speech check did not match)" : ""}.</p>` : ""}
      ${approveBlock}</section>`}

    <section class="card"><h2>Video</h2>
      ${f["generated.mp4"] ? `<div class="media-box" style="margin-top:16px"><video controls preload="metadata" src="${f["generated.mp4"]}?${ver}"></video></div>
        <p class="media-label">As generated, with its own audio.</p>` : `<p class="muted">Generated after you approve the speech.</p>`}</section>

    ${failedChecks.length ? `<div class="notice notice-error"><strong>Some checks failed</strong>${failedChecks.map(esc).join("<br>")}</div>` : ""}

    <section class="card"><h2>Final video</h2>
      ${f["final_talking_head.mp4"] ? `<div class="media-box" style="margin-top:16px"><video controls preload="metadata" src="${f["final_talking_head.mp4"]}?${ver}"></video></div>
        <div class="row" style="margin-top:12px"><a class="btn btn-primary" href="${f["final_talking_head.mp4"]}?download=true&${ver}" download>Download MP4</a>
        <span class="muted small">${p.state === "LIKENESS_ACCEPTED" ? "Accepted as a likeness (not the photo animated)." : p.state === S.COMPLETED ? "Completed." : "Not reviewed yet."}</span></div>` : `<p class="muted">Appears after the checks.</p>`}
      ${p.state === "NEEDS_REVIEW" ? `<h3 style="margin-top:24px">Accept this as a likeness of the person in the photo?</h3>
        <p class="hint" style="margin-top:0">This is the video model's rendering of the person, not the photo animated. Accepting records it as a likeness only.</p>
        <div class="row" style="align-items:flex-start">${refImg ? `<img src="${refImg}" alt="Photo" style="height:120px;border-radius:var(--r-sm)">` : ""}
          ${frames.slice(0, 4).map((k) => `<img src="${f[k]}?${ver}" alt="Frame from the video" style="height:120px;border-radius:var(--r-sm)">`).join("")}</div>
        <div class="field" style="margin-top:12px"><label for="r-name">Reviewer</label><input id="r-name" type="text" placeholder="Your name"></div>
        <div class="row" style="margin-top:12px"><button class="btn btn-primary" id="same-btn">Accept as a likeness</button>
          <button class="btn btn-secondary" id="diff-btn">Reject: not this person</button></div>
        <button class="btn btn-quiet" id="resync-btn">Re-run synchronisation and checks (no new generation)</button>` : ""}
      ${st.identity_review ? `<p class="hint">Reviewed by ${esc(st.identity_review.by)}: ${st.identity_review.same_person ? "accepted as a likeness" : "rejected, not this person"}.</p>` : ""}</section>
  </div>`;
}

function bindStrict(p) {
  const on = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener("click", () => busy(el, fn)); };
  on("cancel-btn", async () => { await api(`/api/projects/${p.id}/cancel`, { method: "POST" }); renderProject(p.id, true); });
  on("retry-btn", async () => { await api(`/api/projects/${p.id}/retry`, { method: "POST" }); renderProject(p.id, true); });
  on("regen-speech-btn", async () => { await api(`/api/projects/${p.id}/strict/regenerate-speech`, { method: "POST" }); renderProject(p.id, true); });
  on("approve-btn", async () => {
    const o = document.getElementById("a-override");
    await api(`/api/projects/${p.id}/strict/approve-audio`, { method: "POST", body: { reviewer: document.getElementById("a-name").value, override: o ? o.checked : false } });
    renderProject(p.id, true);
  });
  const review = (same) => async () => {
    await api(`/api/projects/${p.id}/strict/confirm-identity`, { method: "POST", body: { same_person: same, reviewer: document.getElementById("r-name").value } });
    renderProject(p.id, true);
  };
  on("resync-btn", async () => { await api(`/api/projects/${p.id}/strict/resync`, { method: "POST" }); renderProject(p.id, true); });
  on("same-btn", review(true));
  on("diff-btn", review(false));
}

route();
