from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64
import os
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

from my_env import EmotionEnv
from grader import grade

# =========================
# ENV VARIABLES (REQUIRED)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# (Optional placeholder for OpenAI client usage)
# from openai import OpenAI
# client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# =========================
# INIT
# =========================
app = FastAPI()

YOLO_PATH = "yolov8n-face-lindevs.pt"
yolo_model = YOLO(YOLO_PATH)

env = EmotionEnv()

MAX_STEPS = 12
task_id_global = "emotion-support"

# =========================
# REQUEST MODELS
# =========================
class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42

class StepRequest(BaseModel):
    image: str  # base64 image

# =========================
# UTILS
# =========================
def decode_base64(image_str):
    img_data = base64.b64decode(image_str.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def get_best_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    return tuple(map(int, boxes[0].xyxy[0]))

# =========================
# ROOT → UI
# =========================
@app.get("/")
def home():
    return FileResponse("index.html")

# =========================
# RESET
# =========================
@app.post("/reset")
def reset(req: ResetRequest):
    global task_id_global

    env.reset()
    task_id_global = req.task_id

    print(f"[START] task={task_id_global}", flush=True)

    return {
        "task_id": task_id_global,
        "observation": "environment reset"
    }

# =========================
# STEP
# =========================
@app.post("/step")
def step(req: StepRequest):
    global task_id_global

    try:
        frame = decode_base64(req.image)

        results = yolo_model.predict(frame, device="cpu", verbose=False)
        boxes = results[0].boxes
        box = get_best_box(boxes)

        if box is None:
            print(
                f"[STEP] task={task_id_global} step={env.steps+1} action=detect_face reward=0.00 done=false error=no_face",
                flush=True
            )
            return {
                "observation": {"emotion": "no_face"},
                "reward": 0.0,
                "done": False,
                "task_status": "In Progress",
                "steps": env.steps,
                "total_reward": env.total_reward
            }

        x1, y1, x2, y2 = box
        face = frame[y1:y2, x1:x2]

        result = env.step(Image.fromarray(face))

        obs = result["obs"]
        reward = result["reward"]

        done = (
            obs["emotion"] in ["happy", "neutral"] or
            obs["steps"] >= MAX_STEPS
        )

        # ✅ STRICT STEP FORMAT
        print(
            f"[STEP] task={task_id_global} step={obs['steps']} action=emotion_analysis reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True
        )

        # UI TASK STATUS
        task_status = "In Progress"

        if done:
            steps = obs["steps"]

            if steps <= 4:
                task_status = "Task 1 (Easy)"
            elif steps <= 8:
                task_status = "Task 2 (Moderate)"
            else:
                task_status = "Task 3 (Hard)"

            # ✅ STRICT END FORMAT
            print(
                f"[END] task={task_id_global} score={result['total_reward']:.2f} steps={steps}",
                flush=True
            )

            # AUTO RESET
            env.reset()

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "task_status": task_status,
            "steps": obs["steps"],
            "total_reward": result["total_reward"]
        }

    except Exception as e:
        print(
            f"[STEP] task={task_id_global} step={env.steps+1} action=error reward=0.00 done=false error={str(e)}",
            flush=True
        )

        return {
            "observation": {"error": str(e)},
            "reward": 0.0,
            "done": False,
            "task_status": "Error",
            "steps": env.steps,
            "total_reward": env.total_reward
        }

# =========================
# GRADE
# =========================
@app.get("/grade/task_easy")
def grade_easy():
    score = grade(env.history, env.emotion, env.total_reward)
    return {"score": score, "reward": env.total_reward}
