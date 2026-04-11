from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import base64
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from my_env import EmotionEnv

# =========================
# CONFIG
# =========================
MODEL_NAME = os.getenv("MODEL_NAME", "emotion-model")
TASK_NAME = "emotion-support"
BENCHMARK = "emotion-v1"
MAX_STEPS = 12

# =========================
# INIT
# =========================
app = FastAPI()

yolo = YOLO("yolov8n-face-lindevs.pt")
env = EmotionEnv()

# =========================
# STATE
# =========================
step_count = 0
rewards = []

# =========================
# REQUEST MODEL
# =========================
class StepRequest(BaseModel):
    image: Optional[str] = None

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
# HOME
# =========================
@app.get("/")
def home():
    return FileResponse("index.html")

# =========================
# RESET (START EPISODE)
# =========================
@app.post("/reset")
def reset():
    global step_count, rewards

    step_count = 0
    rewards = []

    print(
        f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}",
        flush=True
    )

    return {"status": "reset_done"}

# =========================
# STEP API
# =========================
@app.post("/step")
def step(req: StepRequest):
    global step_count, rewards

    try:
        # --------------------
        # YOUR LOGIC
        # --------------------

        if not req or not req.image:
            return {...}

        frame = decode(req.image)

        results = yolo.predict(frame, device="cpu", verbose=False)
        box = get_box(results[0].boxes)

        if box is None:
            return {...}

        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]

        result = env.step(Image.fromarray(face))

        obs = result["obs"]
        reward = float(result["reward"])

        step_count += 1
        rewards.append(reward)

        done = obs["emotion"] in ["happy", "neutral"] or step_count >= MAX_STEPS

        print(
            f"[STEP] step={step_count} action={obs['emotion']} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

        if done:
            print(
                f"[END] success={str(step_count < MAX_STEPS).lower()} "
                f"steps={step_count} score={result['total_reward']:.2f} "
                f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
                flush=True
            )

            step_count = 0
            rewards = []
            env.reset()

        # ✅ MUST BE INSIDE TRY
        return {
            "observation": obs,
            "reward": reward,
            "total_reward": result["total_reward"],
            "done": done
        }

    except Exception as e:
        print(
            f"[STEP] step={step_count} action=error reward=0 done=false error={str(e)}",
            flush=True
        )

        return {
            "observation": {"emotion": "error"},
            "reward": 0.0,
            "total_reward": sum(rewards),
            "done": False
        }
