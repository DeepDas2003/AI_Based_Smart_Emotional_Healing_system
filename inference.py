import os
from openai import OpenAI
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from my_env import EmotionEnv

# =========================
# OpenAI Client
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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
reward_list = []
session_started = False

# Task limits
MAX_STEPS_TASK = {"Task 1": 4, "Task 2": 8, "Task 3": 12}

# =========================
# Start Env
# =========================
def start_env():
    global current_step, total_reward, session_started, reward_list
    env.reset()
    current_step = 0
    total_reward = 0.0
    reward_list = []
    session_started = True
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"[START] task=emotion-support env=openenv model={MODEL_NAME}")
    return placeholder, current_step, total_reward, "Environment Started"

# =========================
# Step Env
# =========================
def step_env(frame):
    global current_step, total_reward, session_started, reward_list

    error_msg = "null"
    done = False

    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        error_msg = "No frame provided"
        print(f"[STEP] step={current_step+1} action=process_frame() reward=0.00 done={str(done).lower()} error={error_msg}")
        return placeholder, error_msg, current_step, total_reward

    if not session_started:
        start_env()

    try:
        results = yolo_model.predict(frame, device="cpu", verbose=False)
        boxes = results[0].boxes
        best_box = get_best_box(boxes)

        if best_box is None:
            error_msg = "No face detected"
            print(f"[STEP] step={current_step+1} action=detect_face() reward=0.00 done={str(done).lower()} error={error_msg}")
            return frame, error_msg, current_step, total_reward

        x1, y1, x2, y2 = best_box
        face = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_pil = Image.fromarray(gray)

        result = env.step(face_pil)
        obs = result["obs"]
        reward = result["reward"]
        current_step = obs["steps"]
        total_reward = result["total_reward"]
        reward_list.append(reward)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Determine task
        if current_step <= MAX_STEPS_TASK["Task 1"]:
            task_limit = MAX_STEPS_TASK["Task 1"]
        elif current_step <= MAX_STEPS_TASK["Task 2"]:
            task_limit = MAX_STEPS_TASK["Task 2"]
        else:
            task_limit = MAX_STEPS_TASK["Task 3"]

        # Check if task done
        if obs["emotion"] in ["happy", "neutral"] and current_step >= task_limit:
            done = True
            print(f"[STEP] step={current_step} action=analyze_emotion() reward={reward:.2f} done={str(done).lower()} error=null")
            reset_env()
            return frame, obs, current_step, total_reward

        # Regular step
        print(f"[STEP] step={current_step} action=analyze_emotion() reward={reward:.2f} done={str(done).lower()} error=null")
        return frame, obs, current_step, total_reward

    except Exception as e:
        error_msg = str(e)
        print(f"[STEP] step={current_step+1} action=process_frame() reward=0.00 done={str(done).lower()} error={error_msg}")
        return frame, error_msg, current_step, total_reward

# =========================
# Reset Env
# =========================
def reset_env():
    global current_step, total_reward, session_started, reward_list
    env.reset()
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    session_started = False
    steps_done = current_step
    total_r = total_reward
    reward_list_to_print = ",".join([f"{r:.2f}" for r in reward_list]) if reward_list else "0.00"
    current_step = 0
    total_reward = 0.0
    reward_list = []

    print(f"[END] success=true steps={steps_done} rewards={reward_list_to_print}")
    return placeholder, current_step, total_reward, "Environment Reset"
        task_status += " ✅ Completed"

    output_text = (
        f"Step: {current_step}\n"
        f"Emotion: {obs['emotion']} | Confidence: {obs['confidence']:.2f}\n"
        f"Reward this step: {reward:.2f}\n"
        f"Total Reward: {total_reward:.2f}\n"
        f"Task Status: {task_status}\n"
        f"Advice: {obs['advice']}"
    )

    print("OpenEnv Step (POST OK)")
    return frame, output_text, current_step, total_reward

def reset_env():
    """
    Fully Phase 1 compatible reset.
    Returns 4 values as expected by automated checks.
    """
    global current_step, total_reward, session_started
    env.reset()
    current_step = 0
    total_reward = 0.0
    session_started = False
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    print("OpenEnv Reset (POST OK)")
    return placeholder, current_step, total_reward, "Environment Reset"
