import React, { useEffect, useMemo, useState } from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";

const defenseModes = ["oracle", "regionblur", "patchdrop"];
const h = React.createElement;

function formatConfidence(value) {
  return value === null || value === undefined ? "-" : `${Math.round(value * 100)}%`;
}

function StatusPill({ children, tone = "default" }) {
  return h("span", { className: `pill ${tone}` }, children);
}

function MetricCard({ label, value, hint, wide = false }) {
  return h(
    "article",
    { className: `metric-card ${wide ? "wide" : ""}` },
    h("span", { className: "metric-label" }, label),
    h("strong", { title: String(value || "-") }, value || "-"),
    h("small", null, hint || ""),
  );
}

function DetailCard({ label, children, tone = "", wide = false }) {
  return h(
    "article",
    { className: `detail-card ${wide ? "span-2" : ""}` },
    h("span", null, label),
    h("strong", { className: tone }, children || "-"),
  );
}

function ControlButton({ active, children, onClick }) {
  return h("button", { className: active ? "active" : "", onClick }, children);
}

function Dashboard() {
  const [status, setStatus] = useState({});
  const [error, setError] = useState(null);

  const cacheText = status.inference_cached ? "cached infer" : "fresh infer";
  const confidence = formatConfidence(status.confidence);
  const attentionRatio = status.attention_ratio === null || status.attention_ratio === undefined
    ? "-"
    : `${Number(status.attention_ratio).toFixed(1)}x`;

  const topPredictions = useMemo(() => {
    const topk = Array.isArray(status.topk) ? status.topk : [];
    return topk.map((item) => ({
      label: item.display || item.label || "-",
      confidence: formatConfidence(item.confidence),
    }));
  }, [status.topk]);

  async function refreshStatus() {
    try {
      const res = await fetch("/api/status", { cache: "no-store" });
      const data = await res.json();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }

  async function postControl(payload) {
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    setStatus(data);
  }

  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, 1000);
    return () => clearInterval(timer);
  }, []);

  const connected = Boolean(status.has_frame);
  const runtime = `torch ${status.torch_version || "-"} / cuda ${status.cuda_version || "-"} / ${status.vit_device || "-"}`;
  const pipeline = `${status.async_inference ? "async" : "sync"} / ${status.overlay_style || "overlay"}`;
  const backdoorTone = status.backdoor_active || status.suspicious ? "danger" : "ok";
  const quraTone = status.qura_available ? "ok" : "error";
  const defenseText = status.defense_applied
    ? `${status.defense_mode} applied`
    : (status.defense_on ? `${status.defense_mode} armed` : "off");
  const signalItems = [
    ["Mode", status.mode || "-"],
    ["Attack", status.attack_on ? "ON" : "OFF"],
    ["Defense", defenseText],
    ["Backdoor", status.backdoor_active ? "ACTIVE" : (status.suspicious ? "SUSPICIOUS" : "CLEAR")],
    ["Infer", cacheText],
  ];

  return h(
    "main",
    null,
    h(
      "header",
      { className: "topbar" },
      h("div", null,
        h("p", { className: "eyebrow" }, "React preview"),
        h("h1", null, "Backdoor Defense Console"),
        h("p", { className: "subtitle" }, "Live stream, QURA inference, attention signal, and online defense status."),
      ),
      h("div", { className: "status-pills" },
        h(StatusPill, { tone: connected ? "ok" : "warn" }, connected ? "streaming" : "waiting"),
        h(StatusPill, null, `mode: ${status.mode || "-"}`),
        h(StatusPill, null, pipeline),
      ),
    ),
    h("section", { className: "metrics" },
      h(MetricCard, { label: "Stream FPS", value: status.measured_fps, hint: `target ${status.target_fps || "-"} fps / ${cacheText}` }),
      h(MetricCard, { label: "Active Model", value: status.model, hint: `${status.backend || "torch"} / ${status.vit_device || "-"}` }),
      h(MetricCard, { label: "Prediction", value: status.prediction, hint: `confidence ${confidence}`, wide: true }),
      h(MetricCard, { label: "Attention Ratio", value: attentionRatio, hint: defenseText }),
    ),
    h("section", { className: "signal-strip" },
      ...signalItems.map(([label, value]) => h("div", { className: "signal-chip", key: label },
        h("span", null, label),
        h("strong", null, value),
      )),
    ),
    h("section", { className: "workspace" },
      h("section", { className: "video-panel" },
        h("div", { className: "panel-header" },
          h("div", null,
            h("h2", null, "Live Camera Feed"),
            h("p", null, `${status.width || "-"}x${status.height || "-"} / ${status.target_fps || "-"} fps target / ${cacheText}`),
          ),
          h(StatusPill, { tone: "subtle" }, `source: ${status.actual_source || status.source || "-"}`),
        ),
        h("div", { className: "stream-frame" },
          h("img", { id: "stream", src: "/stream.mjpg", alt: "camera stream" }),
        ),
      ),
      h("aside", { className: "control-panel" },
        h("div", { className: "panel-header compact" },
          h("div", null,
            h("h2", null, "Controls"),
            h("p", null, "Uses the existing REST API and MJPEG stream."),
          ),
        ),
        h("div", { className: "control-block" },
          h("span", { className: "block-title" }, "Demo Mode"),
          h(ControlButton, { active: status.mode === "normal", onClick: () => postControl({ mode: "normal" }) }, "Normal / FP32 baseline"),
          h(ControlButton, { active: status.mode === "triggered", onClick: () => postControl({ mode: "triggered" }) }, "Triggered / INT8 backdoor"),
          h(ControlButton, { active: status.mode === "defended", onClick: () => postControl({ mode: "defended" }) }, "Defended / online mitigation"),
        ),
        h("div", { className: "control-block" },
          h("span", { className: "block-title" }, "Runtime Toggles"),
          h("div", { className: "button-row" },
            h(ControlButton, { active: status.attack_on, onClick: () => postControl({ attack_on: !status.attack_on }) }, `Attack: ${status.attack_on ? "ON" : "OFF"}`),
            h(ControlButton, { active: status.defense_on, onClick: () => postControl({ defense_on: !status.defense_on }) }, `Defense: ${status.defense_on ? "ON" : "OFF"}`),
          ),
          h(ControlButton, {
            active: status.defense_on,
            onClick: () => {
              const idx = defenseModes.indexOf(status.defense_mode || "patchdrop");
              postControl({ defense_mode: defenseModes[(idx + 1) % defenseModes.length] });
            },
          }, `Defense Mode: ${status.defense_mode || "patchdrop"}`),
        ),
        h("div", { className: "control-block" },
          h("span", { className: "block-title" }, "Stream Tools"),
          h("div", { className: "button-row" },
            h("button", { onClick: () => { document.getElementById("stream").src = `/stream.mjpg?ts=${Date.now()}`; } }, "Refresh Stream"),
            h("button", { onClick: () => window.open(`/api/snapshot?ts=${Date.now()}`, "_blank") }, "Snapshot"),
          ),
        ),
      ),
    ),
    h("section", { className: "details-grid" },
      h(DetailCard, { label: "Source" }, `${status.source || "-"} -> ${status.actual_source || "-"}`),
      h(DetailCard, { label: "Frames" }, status.frame_index),
      h(DetailCard, { label: "FPS" }, `${status.measured_fps || "-"} / target ${status.target_fps || "-"}`),
      h(DetailCard, { label: "QURA", tone: quraTone }, status.qura_available ? "available" : `unavailable: ${status.qura_error || "unknown"}`),
      h(DetailCard, { label: "Model" }, status.model),
      h(DetailCard, { label: "Runtime" }, runtime),
      h(DetailCard, { label: "Prediction", wide: true }, `${status.prediction || "-"} (${confidence})`),
      h(DetailCard, { label: "Top Predictions", wide: true }, topPredictions.length ? topPredictions.map((item) => `${item.label} (${item.confidence})`).join(" | ") : "-"),
      h(DetailCard, { label: "Attention" }, attentionRatio),
      h(DetailCard, { label: "Backdoor", tone: backdoorTone }, status.backdoor_active ? "active / suspicious" : (status.suspicious ? "suspicious" : "clear")),
      h(DetailCard, { label: "Defense", tone: status.defense_on ? "ok" : "" }, defenseText),
      h(DetailCard, { label: "Status", tone: error || status.last_error ? "error" : "ok", wide: true }, error || status.last_error || "ok"),
    ),
  );
}

createRoot(document.getElementById("root")).render(h(Dashboard));
