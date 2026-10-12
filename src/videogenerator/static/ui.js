"use strict";
let state, project, selected, polling = false, captionsDraft = null;
const el = id => document.getElementById(id);
const status = text => { el("status").textContent = text; };
const uid = () => crypto.randomUUID().replaceAll("-", "");
async function api(route, payload) {
  const response = await fetch(route, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json", "X-Editor-Token": state.token},
    body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
function guard(fn) {
  return async () => {
    if (polling) { status("Wait for the active operation before editing."); return; }
    try { await fn(); } catch (error) { status(error.message); }
  };
}
function setBusy(value) {
  polling = value;
  document.querySelectorAll("button,input,textarea,select").forEach(node => { node.disabled = value; });
}
function button(text, action) {
  const node = document.createElement("button");
  node.textContent = text; node.addEventListener("click", guard(action)); return node;
}
function option(select, value, text) {
  const node = document.createElement("option"); node.value = value; node.textContent = text; select.append(node);
}
function extent() {
  return Math.max(1, ...project.tracks.flatMap(t => t.segments.map(s => s.start + s.duration)));
}
function bindSpanDrag(span, track, segment, spans, bins) {
  if (bins) return;
  const handle = document.createElement("span");
  handle.className = "resize-handle";
  handle.title = "Drag to resize";
  span.append(handle);
  span.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    event.preventDefault(); event.stopPropagation();
    const resizing = event.target.classList.contains("resize-handle");
    const originX = event.clientX;
    const originStart = segment.start;
    const originDuration = segment.duration;
    const secondsPerPixel = extent() / Math.max(1, spans.getBoundingClientRect().width);
    const frame = 1 / project.settings.fps;
    const snap = value => Math.round(value / frame) * frame;
    const move = moveEvent => {
      const delta = snap((moveEvent.clientX - originX) * secondsPerPixel);
      if (resizing) segment.duration = Math.max(frame, originDuration + delta);
      else segment.start = Math.max(0, originStart + delta);
      span.style.left = `${segment.start / extent() * 100}%`;
      span.style.width = `${segment.duration / extent() * 100}%`;
      span.title = `${segment.start.toFixed(3)}s – ${(segment.start + segment.duration).toFixed(3)}s`;
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      draw(); inspect(track, segment); status("Timeline edit ready to save");
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });
}
function lane(track, bins = false) {
  const row = document.createElement("div"); row.className = `lane ${track.kind} ${track.enabled ? "" : "disabled"}`;
  const label = document.createElement("div"); label.className = "lane-label";
  if (!bins) {
    const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = track.enabled;
    enabled.setAttribute("aria-label", `Enable ${track.name}`);
    enabled.onchange = () => { track.enabled = enabled.checked; draw(); }; label.append(enabled);
  }
  const name = document.createElement("span"); name.textContent = bins ? "Content bins" : track.name; label.append(name);
  if (!bins) label.append(button("+", () => {
    const segment = {id:uid(), start:0, duration:4, source_start:0, speed:1, asset_id:null,
      text:"New scene", prompt:project.prompt, seed:0, backend:"title", volume:1,
      fade_in:0, fade_out:0, duck:"none", role:track.kind === "audio" ? "music" : "source",
      keyframes:[], transform:{x:0,y:0,scale:1,rotation:0,opacity:1}, effects:[]};
    track.segments.push(segment); draw(); inspect(track, segment);
  }));
  const spans = document.createElement("div"); spans.className = "spans";
  for (const segment of track.segments) {
    const span = button(segment.text || segment.prompt || segment.id, () => inspect(track, segment));
    span.className = "span"; span.style.left = `${segment.start / extent() * 100}%`;
    span.style.width = `${segment.duration / extent() * 100}%`;
    span.title = `${segment.start}s – ${segment.start + segment.duration}s`;
    bindSpanDrag(span, track, segment, spans, bins);
    spans.append(span);
  }
  row.append(label, spans); return row;
}
function activateTake(trackId, sourceSegmentId) {
  for (const track of project.tracks) {
    const generated = track.segments.some(segment =>
      segment.generation?.source_segment_id === sourceSegmentId);
    if (generated) track.enabled = track.id === trackId;
  }
  draw();
}
function drawTakes() {
  const root = el("takes"); root.replaceChildren();
  const storyboard = project.tracks.find(track => track.id === "storyboard") || project.tracks[0];
  if (!storyboard) return;
  for (const scene of storyboard.segments) {
    const takes = project.tracks.filter(track => track.segments.some(segment =>
      segment.generation?.source_segment_id === scene.id));
    if (!takes.length) continue;
    const group = document.createElement("div"); group.className = "take-group";
    const title = document.createElement("strong"); title.textContent = scene.text || scene.id; group.append(title);
    for (const track of takes) {
      const segment = track.segments.find(item => item.generation?.source_segment_id === scene.id);
      const label = `${segment?.backend || "take"} · seed ${segment?.seed ?? 0}`;
      const choose = button(label, () => activateTake(track.id, scene.id));
      if (track.enabled) choose.classList.add("active");
      group.append(choose);
    }
    root.append(group);
  }
}
function draw() {
  el("tracks").replaceChildren(...project.tracks.map(t => lane(t)));
  const storyboard = project.tracks.find(t => t.id === "storyboard") || project.tracks[0];
  el("storyboard").replaceChildren(...(storyboard ? [lane(storyboard, true)] : []));
  drawTakes();
  const playhead = el("playhead");
  playhead.max = extent();
  playhead.step = 1 / project.settings.fps;
  el("ruler").replaceChildren();
  for (let i = 0; i <= 4; i++) {
    const tick = document.createElement("span"); tick.textContent = `${(extent()*i/4).toFixed(1)}s`; el("ruler").append(tick);
  }
  el("captions").value = captionsDraft ?? JSON.stringify(project.captions || [], null, 2);
  el("project-json").value = JSON.stringify(project, null, 2);
  el("assets").replaceChildren();
  el("conditioning").replaceChildren(); option(el("conditioning"), "", "None");
  for (const asset of project.assets) {
    option(el("conditioning"), asset.id, asset.name);
    const node = document.createElement("div"); node.className = "asset";
    const title = document.createElement("p"); title.textContent = `${asset.kind}: ${asset.name}`; node.append(title);
    const link = document.createElement("a"); link.textContent = "Open"; link.href = `/media/${encodeURIComponent(asset.id)}`;
    link.target = "_blank"; link.rel = "noopener"; node.append(link);
    node.append(button("Assign to span", () => {
      if (!selected) throw new Error("Select a timeline span first");
      if ((selected.track.kind === "audio" && asset.kind === "image") ||
          (selected.track.kind === "video" && asset.kind === "audio")) throw new Error("Choose a matching video or audio track");
      selected.segment.asset_id = asset.id; inspect(selected.track, selected.segment); draw();
    }));
    el("assets").append(node);
  }
}
function inspect(track, segment) {
  selected = {track, segment}; el("inspector").hidden = false; el("fields").replaceChildren();
  const numbers = ["start", "duration", "source_start", "speed", "volume", "fade_in", "fade_out"];
  for (const name of numbers) {
    const label = document.createElement("label"); label.textContent = name.replaceAll("_", " ");
    const input = document.createElement("input"); input.type = "number"; input.step = "any";
    input.id = `field-${name}`; input.value = segment[name] ?? (["speed", "volume"].includes(name) ? 1 : 0);
    label.append(input); el("fields").append(label);
  }
  const transform = segment.transform || {x:0,y:0,scale:1,rotation:0,opacity:1};
  for (const name of ["x","y","scale","rotation","opacity"]) {
    const label = document.createElement("label"); label.textContent = `transform ${name}`;
    const input = document.createElement("input"); input.type = "number"; input.step = "any";
    input.id = `transform-${name}`; input.value = transform[name];
    label.append(input); el("fields").append(label);
  }
  for (const [name, values] of [["duck", ["none", "soft", "hard"]], ["role", ["source", "voice", "music"]]]) {
    const label = document.createElement("label"); label.textContent = name;
    const select = document.createElement("select"); select.id = `field-${name}`;
    values.forEach(value => option(select, value, value)); select.value = segment[name] || values[0];
    label.append(select); el("fields").append(label);
  }
  const label = document.createElement("label"); label.textContent = "Asset";
  const assets = document.createElement("select"); assets.id = "field-asset_id"; option(assets, "", "Title / empty");
  project.assets.filter(a => track.kind === "audio" ? a.kind !== "image" : a.kind !== "audio")
    .forEach(a => option(assets, a.id, a.name));
  assets.value = segment.asset_id || ""; label.append(assets); el("fields").append(label);
  el("segment-text").value = segment.text || "";
  el("segment-prompt").value = segment.prompt || "";
  el("keyframes").value = JSON.stringify(segment.keyframes || [], null, 2);
  el("effects").value = JSON.stringify(segment.effects || [], null, 2);
}
function applySpan() {
  if (!selected) throw new Error("Select a span");
  const segment = selected.segment;
  for (const name of ["start", "duration", "source_start", "speed", "volume", "fade_in", "fade_out"]) {
    segment[name] = Number(el(`field-${name}`).value);
  }
  for (const name of ["duck", "role"]) segment[name] = el(`field-${name}`).value;
  segment.asset_id = el("field-asset_id").value || null;
  segment.text = el("segment-text").value; segment.prompt = el("segment-prompt").value;
  segment.keyframes = JSON.parse(el("keyframes").value);
  segment.effects = JSON.parse(el("effects").value);
  segment.transform = Object.fromEntries(
    ["x","y","scale","rotation","opacity"].map(name => [name, Number(el(`transform-${name}`).value)])
  );
  draw();
}
async function save() {
  project.captions = JSON.parse(el("captions").value);
  project.settings.audio = {
    normalize: el("normalize-audio").checked,
    target_lufs: Number(el("target-lufs").value),
    true_peak: -1,
    lra: 11
  };
  await api("/api/project", {project}); captionsDraft = null; status("Saved");
}
async function refresh() {
  state = await api("/api/state"); project = state.project; selected = null; captionsDraft = null;
  const audio = project.settings.audio || {};
  el("normalize-audio").checked = Boolean(audio.normalize);
  el("target-lufs").value = audio.target_lufs ?? -14;
  draw(); el("inspector").hidden = true;
}
async function job(payload) {
  setBusy(true);
  try {
    await save(); const submitted = await api("/api/job", payload);
    status("Working… first model use installs dependencies and downloads weights.");
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      state = await api("/api/state");
      const current = state.jobs[submitted.id];
      if (!current || current.status === "running") continue;
      if (current.status === "error") throw new Error(current.error);
      if (payload.action === "render") {
        el("preview").src = `/media/export?v=${Date.now()}`; el("download").hidden = false;
      }
      await refresh(); status("Done"); break;
    }
  } finally { setBusy(false); }
}
function backendInfo() {
  const backend = state.backends.find(b => b.id === el("backend").value);
  el("backend-info").textContent = backend ? JSON.stringify(backend) : "";
}
el("create").onclick = guard(async () => {
  project = await api("/api/create", {prompt:el("prompt").value, script:el("script").value || null,
    duration:Number(el("duration").value), width:Number(el("width").value),
    height:Number(el("height").value), fps:Number(el("fps").value)});
  project.settings.audio = {
    normalize: el("normalize-audio").checked,
    target_lufs: Number(el("target-lufs").value),
    true_peak: -1, lra: 11
  };
  project = await api("/api/project", {project});
  selected = null; captionsDraft = null; el("inspector").hidden = true; draw(); status("Storyboard ready");
});
el("upload").onchange = guard(async () => {
  const files = [...el("upload").files];
  setBusy(true);
  try {
    await save();
    for (const file of files) {
      const response = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, {
        method:"POST", headers:{"X-Editor-Token":state.token}, body:file});
      const result = await response.json(); if (!response.ok) throw new Error(result.error);
    }
    await refresh(); status("Uploaded. Build a storyboard to match filenames, or assign assets to existing spans.");
  } finally { setBusy(false); }
});
el("save").onclick = guard(save);
el("export").onclick = guard(() => job({action:"render"}));
el("estimate").onclick = guard(async () => {
  await save();
  const selectedBackend = state.backends.find(b => b.id === el("backend").value);
  const backend = selectedBackend && ["image", "video"].includes(selectedBackend.kind) ? selectedBackend.id : null;
  el("details").textContent = JSON.stringify(await api("/api/estimate", {project, backend}), null, 2);
  el("details").parentElement.open = true; status("Estimate is heuristic, not a benchmark.");
});
el("apply").onclick = guard(() => { applySpan(); status("Span updated; save to persist"); });
el("split-playhead").onclick = guard(() => {
  if (!selected) throw new Error("Select a span");
  applySpan();
  const segment = selected.segment;
  const offset = Number(el("playhead").value) - segment.start;
  if (!(offset > 0 && offset < segment.duration)) throw new Error("Place the playhead inside the selected span");
  const right = structuredClone(segment);
  right.id = uid();
  right.start = segment.start + offset;
  right.duration = segment.duration - offset;
  right.source_start = segment.source_start + offset * segment.speed;
  segment.duration = offset;
  const index = selected.track.segments.indexOf(segment);
  selected.track.segments.splice(index + 1, 0, right);
  draw(); inspect(selected.track, right); status("Split ready to save");
});
el("crossfade-next").onclick = guard(() => {
  if (!selected) throw new Error("Select a span");
  applySpan();
  const ordered = [...selected.track.segments].sort((a,b) => a.start - b.start);
  const index = ordered.indexOf(selected.segment);
  const next = ordered[index + 1];
  if (!next) throw new Error("No following span on this track");
  const duration = Number(el("crossfade-duration").value);
  if (!(duration > 0 && duration <= Math.min(selected.segment.duration, next.duration))) {
    throw new Error("Crossfade duration must fit both spans");
  }
  selected.segment.fade_out = duration;
  next.fade_in = duration;
  next.start = selected.segment.start + selected.segment.duration - duration;
  draw(); status("Crossfade ready to save");
});
el("duplicate").onclick = guard(() => {
  applySpan(); const segment = structuredClone(selected.segment); segment.id = uid();
  project.tracks.push({id:uid(), name:"Alternative", kind:selected.track.kind, enabled:false, segments:[segment]});
  draw();
});
el("remove").onclick = guard(() => {
  if (!selected) return;
  selected.track.segments = selected.track.segments.filter(s => s.id !== selected.segment.id);
  selected = null; el("inspector").hidden = true; draw();
});
el("regenerate").onclick = guard(async () => {
  const inputAssetId = el("conditioning").value || null;
  applySpan(); await job({action:"regenerate", segment:selected.segment.id,
    backend:el("backend").value, seed:Number(el("seed").value),
    input_asset_id:inputAssetId});
});
el("model-download").onclick = guard(() => job({action:"download", backend:el("backend").value}));
el("backend").onchange = backendInfo;
el("playhead").oninput = () => {
  const time = Number(el("playhead").value);
  if (Number.isFinite(time)) el("preview").currentTime = time;
};
el("preview").ontimeupdate = () => {
  if (!el("playhead").matches(":active")) el("playhead").value = el("preview").currentTime || 0;
};
el("captions").oninput = () => { captionsDraft = el("captions").value; };
for (const kind of ["video", "audio"]) {
  el(`add-${kind}`).onclick = guard(() => {
    project.tracks.push({id:uid(), name:`${kind} ${project.tracks.length + 1}`, kind, enabled:true, segments:[]}); draw();
  });
}
el("apply-json").onclick = guard(async () => {
  const candidate = JSON.parse(el("project-json").value);
  project = await api("/api/project", {project:candidate}); selected = null;
  captionsDraft = null;
  el("inspector").hidden = true; draw(); status("Project JSON saved");
});
guard(async () => {
  await refresh();
  el("prompt").value = project.prompt;
  for (const backend of state.backends.filter(b => ["image", "video", "audio"].includes(b.kind))) {
    option(el("backend"), backend.id, `${backend.id} (${backend.kind})`);
  }
  backendInfo(); el("details").textContent = JSON.stringify(state.hardware, null, 2);
  status(state.busy ? "A job is running; wait before editing." : "Ready. Upload assets or enter a prompt.");
})();
