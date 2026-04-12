from fastapi import APIRouter, Request
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image

router = APIRouter()

# =========================
# REQUEST
# =========================
class StepRequest(BaseModel):
    image: str


# =========================
# UTILS
# =========================
def decode(img):
    try:
        data = base64.b64decode(img.split(",")[-1])
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    return tuple(map(int, boxes[0].xyxy[0]))


# =========================
# RESET
# =========================
@router.post("/reset")
def reset(request: Request):
    app = request.app

    app.state.step = 0
    app.state.rewards = []

    if app.state.env:
        app.state.env.reset()

    # ✔ ONLY ONE START HERE
    print("[START] task=emotion-support", flush=True)

    return {"status": "reset_done"}


# =========================
# STEP
# =========================
@router.post("/step")
def step(req: StepRequest, request: Request):
    app = request.app

    yolo = app.state.yolo
    env = app.state.env

    try:

        # =========================
        # MODEL CHECK
        # =========================
        if yolo is None or env is None:
            print(f"[STEP] step={app.state.step} reward=0.00", flush=True)
            print(f"[END] task=emotion-support score=0.00 steps={app.state.step}", flush=True)
            return {"status": "model_not_loaded"}

        # =========================
        # INPUT CHECK
        # =========================
        if not req.image:
            print(f"[STEP] step={app.state.step} reward=0.00", flush=True)
            return {"status": "no_input"}

        frame = decode(req.image)
        if frame is None:
            print(f"[STEP] step={app.state.step} reward=0.00", flush=True)
            return {"status": "decode_failed"}

        # =========================
        # YOLO DETECTION
        # =========================
        results = yolo.predict(frame, device="cpu", verbose=False)
        box = get_box(results[0].boxes)

        reward = 0.0
        emotion = "no_face"

        if box is not None:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                result = env.step(Image.fromarray(face))
                emotion = result["obs"]["emotion"]
                reward = float(result["reward"])

        # =========================
        # UPDATE STATE
        # =========================
        app.state.step += 1
        app.state.rewards.append(reward)

        total_reward = sum(app.state.rewards)

        # =========================
        # ✔ STEP LOG (STRICT FORMAT)
        # =========================
        print(
            f"[STEP] step={app.state.step} reward={reward:.2f} emotion={emotion}",
            flush=True
        )

        # =========================
        # DONE CONDITION
        # =========================
        done = app.state.step >= 12 or emotion in ["happy", "neutral"]

        # =========================
        # ✔ END LOG (STRICT FORMAT)
        # =========================
        if done:
            print(
                f"[END] task=emotion-support score={total_reward:.2f} steps={app.state.step}",
                flush=True
            )

            # reset state
            app.state.step = 0
            app.state.rewards = []

            if app.state.env:
                app.state.env.reset()

        return {
         "emotion": emotion,
         "confidence": obs.get("confidence", 0.0),
         "advice": obs.get("advice", ""),
         "reward": reward,
         "total_reward": total_reward,
         "done": done
        }

    except Exception as e:
        print("[ERROR]", repr(e), flush=True)
        print(f"[END] task=emotion-support score=0.00 steps={app.state.step}", flush=True)

        return {
            "emotion": "error",
            "reward": 0.0,
            "done": True
        }
