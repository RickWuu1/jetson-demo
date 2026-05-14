"""Optional FastAPI server for the Jetson camera dashboard.

This entrypoint reuses the same FrameHub, QURA pipeline, REST controls, and
MJPEG stream as camera_web_preview.py. It is intended for API framework testing;
the standard-library server remains the default stable path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:
    raise SystemExit(
        "FastAPI server dependencies are not installed.\n"
        "Install them on Jetson with:\n"
        "  pip3 install fastapi uvicorn\n"
    ) from exc

from camera_web_preview import (
    DASHBOARD_DIR,
    REACT_DASHBOARD_DIR,
    FrameHub,
    RealtimeQuraPipeline,
    load_cv_deps,
    parse_args,
)


def create_hub(args) -> FrameHub:
    load_cv_deps()
    qura_pipeline = RealtimeQuraPipeline(args)
    return FrameHub(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        csi_sensor_id=args.csi_sensor_id,
        csi_flip_method=args.csi_flip_method,
        fallback_placeholder=not args.no_placeholder_fallback,
        qura_pipeline=qura_pipeline,
        infer_every_n=args.infer_every_n,
        defense_infer_every_n=args.defense_infer_every_n,
        async_inference=not args.sync_processing,
        overlay_style=args.overlay_style,
    )


def mjpeg_stream(hub: FrameHub, fps: int):
    boundary = "frame"
    interval = 1.0 / max(1, min(30, fps))
    while True:
        frame = hub.latest_jpeg()
        if frame is None:
            time.sleep(0.1)
            continue
        yield (
            f"--{boundary}\r\n"
            "Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(frame)}\r\n\r\n"
        ).encode("ascii") + frame + b"\r\n"
        time.sleep(interval)


def create_app(args) -> FastAPI:
    app = FastAPI(title="Jetson Backdoor Demo", docs_url="/docs", redoc_url=None)
    hub = create_hub(args)
    app.state.hub = hub

    if DASHBOARD_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")
    if REACT_DASHBOARD_DIR.exists():
        app.mount("/react-static", StaticFiles(directory=str(REACT_DASHBOARD_DIR)), name="react-static")

    @app.on_event("startup")
    def _startup() -> None:
        hub.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        hub.stop()

    @app.get("/")
    def index():
        path = DASHBOARD_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(str(path), media_type="text/html")

    @app.get("/react")
    def react_index():
        path = REACT_DASHBOARD_DIR / "index.html"
        if not path.exists():
            raise HTTPException(status_code=404, detail="React dashboard not found")
        return FileResponse(str(path), media_type="text/html")

    @app.get("/api/status")
    def status():
        return hub.status()

    @app.post("/api/control")
    async def control(request: Request):
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")
        try:
            return hub.update_control(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/snapshot")
    def snapshot():
        frame = hub.latest_jpeg()
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available yet")
        return Response(content=frame, media_type="image/jpeg")

    @app.get("/stream.mjpg")
    def stream(fps: int = args.fps):
        return StreamingResponse(
            mjpeg_stream(hub, fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Age": "0",
                "Cache-Control": "no-cache, private",
                "Pragma": "no-cache",
            },
        )

    return app


def main() -> None:
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
