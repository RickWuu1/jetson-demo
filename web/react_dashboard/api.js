const STATUS_ENDPOINTS = ["/api/v1/status", "/api/status"];
const CONTROL_ENDPOINTS = ["/api/v1/control", "/api/control"];

async function fetchJsonWithFallback(endpoints, options) {
  let lastError = null;
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, options);
      if (!response.ok) {
        lastError = new Error(await response.text());
        continue;
      }
      return response.json();
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error("Request failed");
}

export function fetchStatus() {
  return fetchJsonWithFallback(STATUS_ENDPOINTS, { cache: "no-store" });
}

export function postControl(payload) {
  return fetchJsonWithFallback(CONTROL_ENDPOINTS, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function openStatusSocket({ onMessage, onError, onClose }) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws/status`);

  socket.onmessage = (event) => {
    onMessage(JSON.parse(event.data));
  };
  socket.onerror = (event) => {
    if (onError) onError(event);
  };
  socket.onclose = (event) => {
    if (onClose) onClose(event);
  };

  return socket;
}
