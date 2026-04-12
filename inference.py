from fastapi import APIRouter, Request
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image
import sys

router = APIRouter()

# =========================
# SAFE LOGGER (CRITICAL FIX)
# =========================
def log(msg: str):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# =========================
# REQUEST MODEL
# =========================
class StepRequest(BaseModel):
    image: str


# =========================
# UTIL: decode base64 image
# =========================
def decode(img):
    try:
        data = base64.b64decode(img.split(",")[-1])
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


# =========================
# UTIL: get YOLO box
# =========================
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

    if hasattr(app.state, "env") and app.state.env:
        app.state.env.reset()

    # STRICT START OUTPUT
    log("[START] task=emotion-support")

    return {"status": "reset_done"}


# =========================
# STEP
# =========================
@router.post("/step")
def step(req: StepRequest, request: Request):
    app = request.app

    yolo = getattr(app.state, "yolo", None)
    env = getattr(app.state, "env", None)

    # default response
    obs = {
        "emotion": "no_face",
        "confidence": 0.0,
        "advice": ""
    }

    reward = 0.0

    try:

        # =========================
        # MODEL NOT LOADED
        # =========================
        if yolo is None or env is None:
            log(f"[STEP] step={app.state.step} reward=0.00")

            log(f"[END] task=emotion-support score=0.00 steps={app.state.step}")

            app.state.step = 0
            app.state.rewards = []

            return {
                "emotion": "model_not_loaded",
                "confidence": 0.0,
                "advice": "",
                "reward": 0.0,
                "total_reward": 0.0,
                "done": True
            }

        # =========================
        # INPUT CHECK
        # =========================
        if not req.image:
            log(f"[STEP] step={app.state.step} reward=0.00")

            return {
                "emotion": "no_input",
                "confidence": 0.0,
                "advice": "",
                "reward": 0.0,
                "total_reward": sum(app.state.rewards),
                "done": False
            }

        # =========================
        # DECODE IMAGE
        # =========================
        frame = decode(req.image)
        if frame is None:
            log(f"[STEP] step={app.state.step} reward=0.00")

            return {
                "emotion": "decode_failed",
                "confidence": 0.0,
                "advice": "",
                "reward": 0.0,
                "total_reward": sum(app.state.rewards),
                "done": False
            }

        # =========================
        # YOLO DETECTION
        # =========================
        results = yolo.predict(frame, device="cpu", verbose=False)
        box = get_box(results[0].boxes)

        if box:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            if face.size > 0:
                result = env.step(Image.fromarray(face))
                obs = result.get("obs", obs)
                reward = float(result.get("reward", 0.0))

        # =========================
        # UPDATE STATE
        # =========================
        app.state.step += 1
        app.state.rewards.append(reward)

        total_reward = sum(app.state.rewards)

        emotion = obs.get("emotion", "no_face")
        confidence = float(obs.get("confidence", 0.0))
        advice = obs.get("advice", "")

        # =========================
        # STEP LOG (STRICT FORMAT)
        # =========================
        log(f"[STEP] step={app.state.step} reward={reward:.2f}")

        # =========================
        # DONE CONDITION
        # =========================
        done = app.state.step >= 12 or emotion in ["happy", "neutral"]

        # =========================
        # END LOG (STRICT FORMAT)
        # =========================
        if done:
            log(f"[END] task=emotion-support score={total_reward:.2f} steps={app.state.step}")

            app.state.step = 0
            app.state.rewards = []

            if hasattr(app.state, "env") and app.state.env:
                app.state.env.reset()

        return {
            "emotion": emotion,
            "confidence": confidence,
            "advice": advice,
            "reward": reward,
            "total_reward": total_reward,
            "done": done
        }

    except Exception:
        # STRICT FAILURE OUTPUT
        log(f"[STEP] step={app.state.step} reward=0.00")
        log(f"[END] task=emotion-support score=0.00 steps={app.state.step}")

        app.state.step = 0
        app.state.rewards = []

        return {
            "emotion": "error",
            "confidence": 0.0,
            "advice": "",
            "reward": 0.0,
            "total_reward": 0.0,
            "done": True
        }
