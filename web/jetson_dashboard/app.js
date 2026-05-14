let currentStatus = {};

const defenseModes = ["oracle", "regionblur", "patchdrop"];
const setText = (id, value) => {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
};
const setClass = (id, value) => {
  const el = document.getElementById(id);
  if (el) el.className = value;
};

async function postControl(payload) {
  const res = await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  currentStatus = await res.json();
  renderStatus(currentStatus);
}

function renderStatus(data) {
  setText("connected", data.has_frame ? "streaming" : "waiting");
  setClass("connected", data.has_frame ? "pill ok" : "pill warn");
  setText("modePill", `mode: ${data.mode}`);
  setText("pipelinePill", `${data.async_inference ? "async" : "sync"} / ${data.overlay_style || "overlay"}`);
  setText("sourcePill", `source: ${data.actual_source || data.source || "-"}`);
  setText("source", `${data.source} -> ${data.actual_source}`);
  setText("frames", data.frame_index);

  const cacheText = data.inference_cached ? "cached infer" : "fresh infer";
  setText("fps", `${data.measured_fps} / target ${data.target_fps} / ${cacheText}`);
  setText("fpsHero", data.measured_fps || "-");
  setText("fpsHint", `target ${data.target_fps} fps / ${cacheText}`);

  const qura = document.getElementById("qura");
  qura.textContent = data.qura_available ? "available" : `unavailable: ${data.qura_error || "unknown"}`;
  qura.className = data.qura_available ? "value ok" : "value error";

  setText("model", data.model || "-");
  setText("modelHero", data.model || "-");
  setText("runtime", `torch ${data.torch_version || "-"} / cuda ${data.cuda_version || "-"} / ${data.vit_device || "-"}`);
  setText("runtimeHero", `${data.backend || "torch"} / ${data.vit_device || "-"}`);

  const confidence = data.confidence === null || data.confidence === undefined ? "-" : `${Math.round(data.confidence * 100)}%`;
  setText("prediction", `${data.prediction || "-"} (${confidence})`);
  setText("predictionHero", data.prediction || "-");
  setText("confidenceHero", `confidence ${confidence}`);

  const topk = Array.isArray(data.topk) ? data.topk : [];
  document.getElementById("topk").innerHTML = topk.length
    ? topk.map((item) => `${item.display || item.label || "-"} (${Math.round((item.confidence || 0) * 100)}%)`).join("<br>")
    : "-";

  const ratio = data.attention_ratio === null || data.attention_ratio === undefined
    ? "-"
    : `${Number(data.attention_ratio).toFixed(1)}x`;
  setText("attention", ratio);
  setText("attentionHero", ratio);
  setText("streamMeta", `${data.width}x${data.height} / ${data.target_fps} fps target / ${cacheText}`);

  const backdoor = document.getElementById("backdoor");
  backdoor.textContent = data.backdoor_active ? "active / suspicious" : (data.suspicious ? "suspicious" : "clear");
  backdoor.className = data.backdoor_active || data.suspicious ? "value danger" : "value ok";

  const defense = document.getElementById("defense");
  defense.textContent = data.defense_applied ? `${data.defense_mode} applied` : (data.defense_on ? `${data.defense_mode} armed` : "off");
  defense.className = data.defense_on ? "value ok" : "value";
  setText("defenseHero", data.defense_applied ? `${data.defense_mode} applied` : (data.defense_on ? `${data.defense_mode} armed` : "defense off"));

  const err = document.getElementById("error");
  err.textContent = data.last_error || "ok";
  err.className = data.last_error ? "value error" : "value ok";

  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === data.mode);
  });
  setText("attackBtn", `Attack: ${data.attack_on ? "ON" : "OFF"}`);
  document.getElementById("attackBtn").classList.toggle("active", data.attack_on);
  setText("defenseBtn", `Defense: ${data.defense_on ? "ON" : "OFF"}`);
  document.getElementById("defenseBtn").classList.toggle("active", data.defense_on);
  setText("defenseModeBtn", `Defense Mode: ${data.defense_mode}`);
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    currentStatus = await res.json();
    renderStatus(currentStatus);
  } catch (err) {
    setText("connected", "disconnected");
    setClass("connected", "pill warn");
    setText("error", String(err));
  }
}

document.querySelectorAll("[data-mode]").forEach((btn) => {
  btn.addEventListener("click", () => postControl({ mode: btn.dataset.mode }));
});
document.getElementById("attackBtn").addEventListener("click", () => {
  postControl({ attack_on: !currentStatus.attack_on });
});
document.getElementById("defenseBtn").addEventListener("click", () => {
  postControl({ defense_on: !currentStatus.defense_on });
});
document.getElementById("defenseModeBtn").addEventListener("click", () => {
  const idx = defenseModes.indexOf(currentStatus.defense_mode || "patchdrop");
  postControl({ defense_mode: defenseModes[(idx + 1) % defenseModes.length] });
});
document.getElementById("refreshBtn").addEventListener("click", () => {
  document.getElementById("stream").src = `/stream.mjpg?ts=${Date.now()}`;
});
document.getElementById("snapshotBtn").addEventListener("click", () => {
  window.open(`/api/snapshot?ts=${Date.now()}`, "_blank");
});

setInterval(refreshStatus, 1000);
refreshStatus();
