import { React } from "./react.js";
import { fetchStatus, openStatusSocket, postControl } from "./api.js";

export function useDashboardState() {
  const [status, setStatus] = React.useState({});
  const [error, setError] = React.useState(null);
  const [transport, setTransport] = React.useState("connecting");

  async function refreshStatus() {
    try {
      const data = await fetchStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError(String(err));
    }
  }

  async function sendControl(payload) {
    const data = await postControl(payload);
    setStatus(data);
    setError(null);
    return data;
  }

  React.useEffect(() => {
    let stopped = false;
    let pollTimer = null;
    let socket = null;

    const startPolling = () => {
      if (pollTimer || stopped) return;
      setTransport("polling");
      refreshStatus();
      pollTimer = window.setInterval(refreshStatus, 1000);
    };

    try {
      socket = openStatusSocket({
        onMessage: (data) => {
          if (stopped) return;
          setStatus(data);
          setError(null);
          setTransport("websocket");
        },
        onError: () => {
          if (stopped) return;
          setError("WebSocket unavailable; using REST polling");
          startPolling();
        },
        onClose: () => {
          if (stopped) return;
          startPolling();
        },
      });
    } catch (err) {
      setError(String(err));
      startPolling();
    }

    return () => {
      stopped = true;
      if (pollTimer) window.clearInterval(pollTimer);
      if (socket) socket.close();
    };
  }, []);

  return {
    error,
    refreshStatus,
    sendControl,
    status,
    transport,
  };
}
