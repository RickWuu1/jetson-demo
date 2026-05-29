import { createRoot, h } from "./react.js";
import {
  ControlButton,
  DetailCard,
  MetricCard,
  ModeButton,
  StatusPill,
  ToggleRow,
} from "./components.js";
import { buildDashboardView, defenseModes } from "./format.js";
import { useDashboardState } from "./state.js";
import { openSnapshot, refreshStream, STREAM_URL } from "./video.js";

function Dashboard() {
  const { error, sendControl, status, transport } = useDashboardState();
  const view = buildDashboardView(status, error);
  const connected = Boolean(status.has_frame);

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
        h(StatusPill, null, view.modeLabel),
        h(StatusPill, null, `${view.pipeline} / ${transport}`),
      ),
    ),
    h("section", { className: "metrics" },
      h(MetricCard, { label: "Stream FPS", value: status.measured_fps, hint: `target ${status.target_fps || "-"} fps / ${view.cacheText}` }),
      h(MetricCard, { label: "Active Model", value: status.model, hint: `${status.backend || "torch"} / ${status.vit_device || "-"}` }),
      h(MetricCard, { label: "Prediction", value: status.prediction, hint: `confidence ${view.confidence}`, wide: true }),
      h(MetricCard, { label: "Attention Ratio", value: view.attentionRatio, hint: view.defenseText }),
    ),
    h("section", { className: "workspace" },
      h("div", { className: "main-column" },
        h("section", { className: "video-panel" },
          h("div", { className: "panel-header" },
            h("div", null,
              h("h2", null, "Live Camera Feed"),
              h("p", null, `${status.width || "-"}x${status.height || "-"} / ${status.target_fps || "-"} fps target / ${view.cacheText}`),
            ),
            h(StatusPill, { tone: "subtle" }, `source: ${status.actual_source || status.source || "-"}`),
          ),
          h("div", { className: "stream-frame" },
            h("img", { id: "stream", src: STREAM_URL, alt: "camera stream" }),
          ),
        ),
        h("section", { className: "details-grid" },
          h(DetailCard, { label: "Source" }, `${status.source || "-"} -> ${status.actual_source || "-"}`),
          h(DetailCard, { label: "Frames / FPS" }, `${status.frame_index || "-"} / ${status.measured_fps || "-"} fps`),
          h(DetailCard, { label: "QURA", tone: view.quraTone }, status.qura_available ? "available" : `unavailable: ${status.qura_error || "unknown"}`),
          h(DetailCard, { label: "Runtime" }, view.runtime),
          h(DetailCard, { label: "Top Predictions", wide: true }, view.topPredictions.length ? view.topPredictions.map((item) => `${item.label} (${item.confidence})`).join(" | ") : "-"),
          h(DetailCard, { label: "Trigger / Backdoor", tone: view.backdoorTone }, `${status.attack_on ? "injected" : "off"} / ${status.backdoor_active ? "active" : (status.suspicious ? "suspicious" : "clear")}`),
          h(DetailCard, { label: "Defense", tone: status.defense_on ? "ok" : "" }, view.defenseText),
          h(DetailCard, { label: "Status", tone: error || status.last_error ? "error" : "ok", wide: true }, view.statusText),
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
              subtitle: view.normalUsesInt8 ? "FP32 unavailable" : "clean reference",
              onClick: () => sendControl({ mode: "normal" }),
            }),
            h(ModeButton, {
              active: status.mode === "triggered",
              tone: "int8",
              title: "INT8 Quantized",
              subtitle: "trigger controlled below",
              onClick: () => sendControl({ mode: "triggered" }),
            }),
            h(ModeButton, {
              active: status.mode === "defended",
              tone: "defended",
              title: "Defense Mode",
              subtitle: "detect and mitigate",
              onClick: () => sendControl({ mode: "defended" }),
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
            onClick: () => sendControl({ attack_on: !status.attack_on }),
          }),
          h(ToggleRow, {
            active: status.defense_on,
            icon: "defense",
            label: "Online Defense",
            onClick: () => sendControl({ defense_on: !status.defense_on }),
          }),
          h(ControlButton, {
            active: status.defense_on,
            onClick: () => {
              const idx = defenseModes.indexOf(status.defense_mode || "patchdrop");
              sendControl({ defense_mode: defenseModes[(idx + 1) % defenseModes.length] });
            },
          }, `Defense Mode: ${status.defense_mode || "patchdrop"}`),
        ),
        h("div", { className: "control-block" },
          h("span", { className: "block-title" }, "Stream Tools"),
          h("div", { className: "button-row" },
            h("button", { onClick: () => refreshStream() }, "Refresh Stream"),
            h("button", { onClick: () => openSnapshot() }, "Snapshot"),
          ),
        ),
        h("div", { className: `threat-card ${view.threatTone}` },
          h("span", { className: "threat-kicker" }, view.threatState),
          h("strong", null, status.prediction || "-"),
          h("small", null,
            status.fire_prob_attacked !== null && status.fire_prob_attacked !== undefined
              ? `attack ${Math.round(status.fire_prob_attacked * 100)}% → defended ${view.confidence} / attn ${view.attentionRatio}`
              : `confidence ${view.confidence} / attention ${view.attentionRatio}`
          ),
        ),
      ),
    ),
  );
}

createRoot(document.getElementById("root")).render(h(Dashboard));
