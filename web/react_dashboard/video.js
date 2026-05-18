export const STREAM_URL = "/stream.mjpg";

export function refreshStream(elementId = "stream") {
  const stream = document.getElementById(elementId);
  if (stream) {
    stream.src = `${STREAM_URL}?ts=${Date.now()}`;
  }
}

export function openSnapshot() {
  window.open(`/api/snapshot?ts=${Date.now()}`, "_blank");
}
