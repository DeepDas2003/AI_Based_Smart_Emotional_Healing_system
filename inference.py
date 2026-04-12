from fastapi import APIRouter, Request
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from PIL import Image
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# =========================
# REQUEST MODEL
# =========================
class StepRequest(BaseModel):
    image: str

# =========================
# UTILS
# =========================
def decode(img):
    data = base64.b64decode(img.split(",")[1])
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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

    logger.info("[RESET] task=emotion-support")

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
        if yolo is None or env is None:
            logger.error("[ERROR] models not loaded")
            return {
                "emotion": "model_not_loaded",
                "confidence": 0.0,
                "reward": 0.0,
                "total_reward": 0.0,
                "advice": "",
                "status": "error"
            }

        if app.state.step == 0:
            logger.info("[START] task=emotion-support")

        # IMAGE
        frame = decode(req.image)

        # YOLO
        results = yolo.predict(frame, device="cpu", conf=0.3, verbose=False)
        box = get_box(results[0].boxes)

        if box is None:
            emotion = "no_face"
            confidence = 0.0
            reward = 0.0
            advice = ""

        else:
            x1, y1, x2, y2 = box
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                emotion = "no_face"
                confidence = 0.0
                reward = 0.0
                advice = ""
            else:
                result = env.step(Image.fromarray(face))

                obs = result["obs"]
                emotion = obs["emotion"]
                confidence = obs["confidence"]
                reward = result["reward"]
                advice = obs["advice"]

        # UPDATE
        app.state.step += 1
        app.state.rewards.append(reward)

        total = sum(app.state.rewards)

        done = (
            app.state.step >= 12 or
            emotion in ["happy", "neutral"]
        )

        logger.info(
            f"[STEP] step={app.state.step} emotion={emotion} confidence={confidence:.2f} reward={reward:.2f}"
        )

        if done:
            logger.info(f"[END] score={total:.2f} steps={app.state.step}")

        response = {
            "emotion": emotion,
            "confidence": confidence,
            "reward": reward,
            "total_reward": total,
            "advice": advice,
            "status": "running"
        }

        if done:
            env.reset()
            app.state.step = 0
            app.state.rewards = []

        return response

    except Exception as e:
        logger.error(f"[ERROR] {repr(e)}")

        return {
            "emotion": "error",
            "confidence": 0.0,
            "reward": 0.0,
            "total_reward": 0.0,
            "advice": "",
            "status": "error"
        }

# =========================
# SAFE MAIN (CRITICAL)
# =========================
if __name__ == "__main__":
    print("[TEST] Running inference.py standalone...", flush=True)

    try:
        from ultralytics import YOLO
        from my_env import EmotionEnv

        print("[TEST] Loading models...", flush=True)

        yolo = YOLO("yolov8n-face-lindevs.pt")
        env = EmotionEnv()

        print("[TEST] Models loaded successfully", flush=True)

        import numpy as np
        from PIL import Image

        dummy = np.zeros((224, 224, 3), dtype=np.uint8)

        yolo.predict(dummy, device="cpu", verbose=False)

        result = env.step(Image.fromarray(dummy))

        print("[TEST RESULT]", result, flush=True)

        print("[SUCCESS] inference.py executed correctly", flush=True)

    except Exception as e:
        print("[FATAL ERROR]", repr(e), flush=True)
        exit(1)
