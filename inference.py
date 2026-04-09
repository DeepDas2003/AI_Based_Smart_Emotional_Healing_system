import os
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from my_env import EmotionEnv

# =========================
# OpenAI Client (required)
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# =========================
# YOLO Face Detection
# =========================
YOLO_PATH = "yolov8n-face-lindevs.pt"
yolo_model = YOLO(YOLO_PATH)

def get_best_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    return tuple(map(int, boxes[0].xyxy[0]))

# =========================
# Environment + State
# =========================
env = EmotionEnv()
current_step = 0
total_reward = 0.0
session_started = False

# =========================
# OpenEnv Functions (Phase 1)
# =========================
def start_env():
    global current_step, total_reward, session_started
    env.reset()
    current_step = 0
    total_reward = 0.0
    session_started = True
    print("OpenEnv Reset (POST OK)")
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    return placeholder, current_step, total_reward

MAX_STEPS_TASK = {
    "Task 1": 4,
    "Task 2": 8,
    "Task 3": 12
}

def step_env(frame):
    global current_step, total_reward, session_started

    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        return placeholder, "⚠️ No frame provided", current_step, total_reward

    if not session_started:
        session_started = True
        print("OpenEnv Start (POST OK)")

    results = yolo_model.predict(frame, device="cpu", verbose=False)
    boxes = results[0].boxes
    best_box = get_best_box(boxes)

    if best_box is None:
        return frame, "❌ No face detected", current_step, total_reward

    x1, y1, x2, y2 = best_box
    face = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_pil = Image.fromarray(gray)

    result = env.step(face_pil)
    obs = result["obs"]
    reward = result["reward"]
    current_step = obs["steps"]
    total_reward = result["total_reward"]

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Determine current task
    if current_step <= MAX_STEPS_TASK["Task 1"]:
        task_status = "Task 1 (Easy)"
        task_limit = MAX_STEPS_TASK["Task 1"]
    elif current_step <= MAX_STEPS_TASK["Task 2"]:
        task_status = "Task 2 (Medium)"
        task_limit = MAX_STEPS_TASK["Task 2"]
    else:
        task_status = "Task 3 (Hard)"
        task_limit = MAX_STEPS_TASK["Task 3"]

    # Only reset after completing task steps
    if obs["emotion"] in ["happy", "neutral"] and current_step >= task_limit:
        print(f"[END] total_reward={total_reward:.2f} | {task_status} Completed")
        env.reset()
        session_started = False
        current_step = 0
        total_reward = 0.0
        task_status += " ✅ Completed"

    output_text = (
        f"Step: {current_step}\n"
        f"Emotion: {obs['emotion']} | Confidence: {obs['confidence']:.2f}\n"
        f"Reward this step: {reward:.2f}\n"
        f"Total Reward: {total_reward:.2f}\n"
        f"Task Status: {task_status}\n"
        f"Advice: {obs['advice']}"
    )

    return frame, output_text, current_step, total_reward

def reset_env():
    global current_step, total_reward, session_started
    env.reset()
    print("OpenEnv Reset (POST OK)")
    current_step = 0
    total_reward = 0.0
    session_started = False
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    return placeholder, current_step, total_reward

# =========================
# Optional: Test OpenAI Inference
# =========================
if __name__ == "__main__":
    print(run_inference("Hello from OpenEnv + OpenAI!"))
