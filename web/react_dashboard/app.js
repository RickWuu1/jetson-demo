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

function Icon({ name }) {
  const common = {
    viewBox: "0 0 64 64",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 4,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": "true",
  };

  if (name === "target") {
    return h("svg", common,
      h("circle", { cx: 32, cy: 32, r: 14 }),
      h("circle", { cx: 32, cy: 32, r: 3, fill: "currentColor", stroke: "none" }),
      h("path", { d: "M32 8v10M32 46v10M8 32h10M46 32h10" }),
    );
  }

  if (name === "defense") {
    return h("svg", common,
      h("path", { d: "M32 7l20 8v14c0 14-8 23-20 28-12-5-20-14-20-28V15l20-8z" }),
      h("circle", { cx: 32, cy: 32, r: 7 }),
      h("path", { d: "M32 27v10M27 32h10" }),
    );
  }

  if (name === "attack") {
    return h("svg", common,
      h("path", { d: "M14 50L50 14" }),
      h("path", { d: "M18 18l28 28" }),
      h("path", { d: "M13 28l9-9 9 9" }),
      h("path", { d: "M33 45l12-12 6 6" }),
    );
  }

  return h("svg", common,
    h("path", { d: "M32 7l20 8v14c0 14-8 23-20 28-12-5-20-14-20-28V15l20-8z" }),
    h("path", { d: "M24 32l6 6 12-14" }),
  );
}

function ModeButton({ active, tone, title, subtitle, onClick }) {
  const iconName = tone === "normal" ? "shield" : (tone === "int8" ? "target" : "defense");
  return h(
    "button",
    { className: `mode-card ${active ? "active" : ""} ${tone}`, onClick },
    h("span", { className: "mode-icon" }, h(Icon, { name: iconName })),
    h("strong", null, title),
    h("small", null, subtitle),
  );
}

function ToggleRow({ active, danger = false, icon, label, onClick }) {
  return h(
    "button",
    { className: `toggle-row ${active ? "active" : ""} ${danger ? "danger-toggle" : "defense-toggle"}`, onClick },
    h("span", { className: "toggle-icon" }, h(Icon, { name: icon })),
    h("span", { className: "toggle-label" }, label),
    h("span", { className: "switch-shell" },
      h("span", { className: "switch-text" }, active ? "ON" : "OFF"),
      h("span", { className: "switch-dot" }),
    ),
  );
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
  const threatState = status.defense_applied
    ? "DEFENSE ACTIVE"
    : (status.backdoor_active || status.suspicious ? "BACKDOOR ACTIVE" : "SYSTEM CLEAR");
  const threatTone = status.defense_applied ? "defended" : (status.backdoor_active || status.suspicious ? "danger" : "safe");
  const modeLabel = status.mode === "normal"
    ? "Clean baseline"
    : (status.mode === "triggered" ? "INT8 quantized" : "Defense mode");

  return h(
    "main",
    null,
    h(
      "header",
      { className: "topbar" },
      h("div", null,
        h("p", { className: "eyebrow" }, "Quantization-Activated Backdoor Demo"),
        h("h1", null, "Backdoor Defense Console"),
        h("p", { className: "subtitle" }, "Start from an FP32 baseline, switch to INT8 quantization, inject the trigger, then enable online defense."),
      ),
      h("div", { className: "status-pills" },
        h(StatusPill, { tone: connected ? "ok" : "warn" }, connected ? "streaming" : "waiting"),
        h(StatusPill, null, modeLabel),
        h(StatusPill, null, pipeline),
      ),
    ),
    h("section", { className: "metrics" },
      h(MetricCard, { label: "Stream FPS", value: status.measured_fps, hint: `target ${status.target_fps || "-"} fps / ${cacheText}` }),
      h(MetricCard, { label: "Active Model", value: status.model, hint: `${status.backend || "torch"} / ${status.vit_device || "-"}` }),
      h(MetricCard, { label: "Prediction", value: status.prediction, hint: `confidence ${confidence}`, wide: true }),
      h(MetricCard, { label: "Attention Ratio", value: attentionRatio, hint: defenseText }),
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
            h("p", null, "FP32 clean -> INT8 -> trigger -> defense."),
          ),
        ),
        h("div", { className: "control-block" },
          h("span", { className: "block-title" }, "Demo Mode"),
          h("div", { className: "mode-grid" },
            h(ModeButton, {
              active: status.mode === "normal",
              tone: "normal",
              title: "FP32 Baseline",
              subtitle: "clean reference",
              onClick: () => postControl({ mode: "normal" }),
            }),
            h(ModeButton, {
              active: status.mode === "triggered",
              tone: "int8",
              title: "INT8 Quantized",
              subtitle: "trigger controlled below",
              onClick: () => postControl({ mode: "triggered" }),
            }),
            h(ModeButton, {
              active: status.mode === "defended",
              tone: "defended",
              title: "Defense Mode",
              subtitle: "detect and mitigate",
              onClick: () => postControl({ mode: "defended" }),
            }),
          ),
        ),
        h("div", { className: "control-block" },
          h("span", { className: "block-title" }, "Runtime Toggles"),
          h(ToggleRow, {
            active: status.attack_on,
            danger: true,
            icon: "attack",
            label: "Trigger Injection",
            onClick: () => postControl({ attack_on: !status.attack_on }),
          }),
          h(ToggleRow, {
            active: status.defense_on,
            icon: "defense",
            label: "Online Defense",
            onClick: () => postControl({ defense_on: !status.defense_on }),
          }),
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
        h("div", { className: `threat-card ${threatTone}` },
          h("span", { className: "threat-kicker" }, threatState),
          h("strong", null, status.prediction || "-"),
          h("small", null, `confidence ${confidence} / attention ${attentionRatio}`),
        ),
      ),
    ),
    h("section", { className: "details-grid" },
      h(DetailCard, { label: "Source" }, `${status.source || "-"} -> ${status.actual_source || "-"}`),
      h(DetailCard, { label: "Frames / FPS" }, `${status.frame_index || "-"} / ${status.measured_fps || "-"} fps`),
      h(DetailCard, { label: "QURA", tone: quraTone }, status.qura_available ? "available" : `unavailable: ${status.qura_error || "unknown"}`),
      h(DetailCard, { label: "Runtime" }, runtime),
      h(DetailCard, { label: "Top Predictions", wide: true }, topPredictions.length ? topPredictions.map((item) => `${item.label} (${item.confidence})`).join(" | ") : "-"),
      h(DetailCard, { label: "Trigger / Backdoor", tone: backdoorTone }, `${status.attack_on ? "injected" : "off"} / ${status.backdoor_active ? "active" : (status.suspicious ? "suspicious" : "clear")}`),
      h(DetailCard, { label: "Defense", tone: status.defense_on ? "ok" : "" }, defenseText),
      h(DetailCard, { label: "Status", tone: error || status.last_error ? "error" : "ok", wide: true }, error || status.last_error || "ok"),
    ),
  );
}

createRoot(document.getElementById("root")).render(h(Dashboard));
