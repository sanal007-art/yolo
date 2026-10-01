import os
import secrets
import threading
from collections import defaultdict

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from torchvision.ops import nms
from ultralytics import YOLO

# ---------------- CONFIG (env overridable) ----------------
WEIGHTS = os.getenv("WEIGHTS", "yolo11n.pt")
_cls = os.getenv("CLASSES", "0")
CLASSES = [int(c) for c in _cls.split(",")] if _cls else None
CONF = float(os.getenv("CONF", 0.6))
IOU = float(os.getenv("IOU", 0.4))
POST_IOU = float(os.getenv("POST_IOU", 0.3))
IMGSZ = int(os.getenv("IMGSZ", 480))
CONFIRM_FRAMES = int(os.getenv("CONFIRM_FRAMES", 4))
MIN_AVG_CONF = float(os.getenv("MIN_AVG_CONF", 0.6))
MIN_BOX_AREA = int(os.getenv("MIN_BOX_AREA", 24 * 24))
LIGHT_NORM = os.getenv("LIGHT_NORM", "1") == "1"
API_KEY = os.getenv("API_KEY", "")
MAX_CONN = int(os.getenv("MAX_CONN", 2))

torch.set_num_threads(os.cpu_count() or 2)
app = FastAPI(title="Live Detection + Tracking")
active = 0
active_lock = threading.Lock()

# ---------------- LIGHTING ----------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def normalize_light(frame):
    m = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    out = frame
    if m < 90 or m > 170:
        g = float(np.clip(np.log(0.5) / np.log(max(m, 1) / 255.0), 0.4, 2.5))
        lut = np.array([(i / 255.0) ** g * 255 for i in range(256)], dtype="uint8")
        out = cv2.LUT(out, lut)
    l, a, b = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

# ---------------- CORE (one instance per connection) ----------------
class Processor:
    def __init__(self):
        self.model = YOLO(WEIGHTS)  # own model so tracker state is never shared
        self.hits = defaultdict(int)
        self.conf_sum = defaultdict(float)

    def process(self, frame):
        inp = normalize_light(frame) if LIGHT_NORM else frame
        res = self.model.track(
            inp, tracker="bytetrack.yaml", persist=True, conf=CONF, iou=IOU,
            classes=CLASSES, agnostic_nms=True, imgsz=IMGSZ,
            device="cpu", verbose=False,
        )[0]

        b = res.boxes
        if b is None or len(b) == 0 or b.id is None:
            return []
        boxes, confs = b.xyxy.tolist(), b.conf.tolist()
        clss, ids = b.cls.int().tolist(), b.id.int().tolist()

        for tid, c in zip(ids, confs):
            self.hits[tid] += 1
            self.conf_sum[tid] += c

        cand = [
            i for i, bx in enumerate(boxes)
            if (bx[2] - bx[0]) * (bx[3] - bx[1]) >= MIN_BOX_AREA
            and self.hits[ids[i]] >= CONFIRM_FRAMES
            and self.conf_sum[ids[i]] / self.hits[ids[i]] >= MIN_AVG_CONF
        ]
        if not cand:
            return []

        keep = nms(
            torch.tensor([boxes[i] for i in cand], dtype=torch.float32),
            torch.tensor([confs[i] for i in cand], dtype=torch.float32),
            POST_IOU,
        ).tolist()

        return [
            dict(id=ids[cand[k]], cls=self.model.names[clss[cand[k]]],
                 conf=round(confs[cand[k]], 3),
                 box=[round(v, 1) for v in boxes[cand[k]]])
            for k in keep
        ]

# ---------------- ROUTES ----------------
@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS, "active": active}

@app.get("/")
def index():
    return FileResponse("index.html")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, key: str = ""):
    global active
    if API_KEY and not secrets.compare_digest(key, API_KEY):
        await ws.close(code=1008)
        return
    with active_lock:
        if active >= MAX_CONN:
            full = True
        else:
            active += 1
            full = False
    await ws.accept()
    if full:
        await ws.close(code=1013)
        return
    try:
        proc = await run_in_threadpool(Processor)
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            dets = await run_in_threadpool(proc.process, frame)
            h, w = frame.shape[:2]
            await ws.send_json({"w": w, "h": h, "detections": dets})
    except WebSocketDisconnect:
        pass
    finally:
        with active_lock:
            active -= 1