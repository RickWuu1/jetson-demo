export const defenseModes = ["oracle", "regionblur", "patchdrop"];

export function formatConfidence(value) {
  return value === null || value === undefined ? "-" : `${Math.round(value * 100)}%`;
}

export function buildDashboardView(status, error) {
  const cacheText = status.inference_cached ? "cached infer" : "fresh infer";
  const confidence = formatConfidence(status.confidence);
  const attentionRatio = status.attention_ratio === null || status.attention_ratio === undefined
    ? "-"
    : `${Number(status.attention_ratio).toFixed(1)}x`;
  const runtime = `torch ${status.torch_version || "-"} / cuda ${status.cuda_version || "-"} / ${status.vit_device || "-"}`;
  const pipeline = `${status.async_inference ? "async" : "sync"} / ${status.overlay_style || "overlay"}`;
  const backdoorTone = status.backdoor_active ? "danger" : (status.suspicious ? "error" : "ok");
  const quraTone = status.qura_available ? "ok" : "error";
  const defenseText = status.defense_applied
    ? `${status.defense_mode} applied`
    : (status.defense_on ? `${status.defense_mode} armed` : "off");
  const threatState = status.defense_applied
    ? "DEFENSE ACTIVE"
    : (status.backdoor_active
      ? "BACKDOOR ACTIVE"
      : (status.attack_on && status.mode === "normal" ? "BACKDOOR DORMANT" : (status.suspicious ? "SUSPICIOUS ATTENTION" : "SYSTEM CLEAR")));
  const threatTone = status.defense_applied
    ? "defended"
    : (status.backdoor_active ? "danger" : (status.attack_on && status.mode === "normal" ? "dormant" : "safe"));
  const normalUsesInt8 = status.mode === "normal" && String(status.model || "").includes("INT8");
  const modeLabel = status.mode === "normal"
    ? (normalUsesInt8 ? "INT8 fallback" : "Clean baseline")
    : (status.mode === "triggered" ? "INT8 quantized" : "Defense mode");
  const topPredictions = Array.isArray(status.topk)
    ? status.topk.map((item) => ({
      label: item.display || item.label || "-",
      confidence: formatConfidence(item.confidence),
    }))
    : [];

  return {
    attentionRatio,
    backdoorTone,
    cacheText,
    confidence,
    defenseText,
    modeLabel,
    normalUsesInt8,
    pipeline,
    quraTone,
    runtime,
    threatState,
    threatTone,
    topPredictions,
    statusText: error || status.last_error || "ok",
  };
}
