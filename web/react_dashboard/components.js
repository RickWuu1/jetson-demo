import { h } from "./react.js";

export function StatusPill({ children, tone = "default" }) {
  return h("span", { className: `pill ${tone}` }, children);
}

export function MetricCard({ label, value, hint, wide = false }) {
  return h(
    "article",
    { className: `metric-card ${wide ? "wide" : ""}` },
    h("span", { className: "metric-label" }, label),
    h("strong", { title: String(value || "-") }, value || "-"),
    h("small", null, hint || ""),
  );
}

export function DetailCard({ label, children, tone = "", wide = false }) {
  return h(
    "article",
    { className: `detail-card ${wide ? "span-2" : ""}` },
    h("span", null, label),
    h("strong", { className: tone }, children || "-"),
  );
}

export function ControlButton({ active, children, onClick }) {
  return h("button", { className: active ? "active" : "", onClick }, children);
}

export function Icon({ name }) {
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

export function ModeButton({ active, tone, title, subtitle, onClick }) {
  const iconName = tone === "normal" ? "shield" : (tone === "int8" ? "target" : "defense");
  return h(
    "button",
    { className: `mode-card ${active ? "active" : ""} ${tone}`, onClick },
    h("span", { className: "mode-icon" }, h(Icon, { name: iconName })),
    h("strong", null, title),
    h("small", null, subtitle),
  );
}

export function ToggleRow({ active, danger = false, icon, label, onClick }) {
  return h(
    "button",
    {
      className: `toggle-row ${active ? "active" : ""} ${danger ? "danger-toggle" : "defense-toggle"}`,
      onClick,
    },
    h("span", { className: "toggle-icon" }, h(Icon, { name: icon })),
    h("span", { className: "toggle-label" }, label),
    h("span", { className: "switch-shell" },
      h("span", { className: "switch-text" }, active ? "ON" : "OFF"),
      h("span", { className: "switch-dot" }),
    ),
  );
}
