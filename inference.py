import os
import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
from my_env import EmotionEnv

# =========================
# Model + Env
# =========================
YOLO_PATH = "yolov8n-face-lindevs.pt"
yolo_model = YOLO(YOLO_PATH)
env = EmotionEnv()
MAX_EPISODE_STEPS = 12

# =========================
# Images Folder
# =========================
IMAGE_FOLDER = "./emotional_faces/"
image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png",".jpg",".jpeg"))]
if not image_files:
    raise ValueError(f"No images found in {IMAGE_FOLDER}")

# =========================
# Session State
# =========================
current_step = 0
total_reward = 0.0
reward_list = []
session_started = False

# =========================
# Utility
# =========================
def get_best_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    return tuple(map(int, boxes[0].xyxy[0]))

def start_env():
    global current_step, total_reward, reward_list, session_started
    env.reset()
    current_step = 0
    total_reward = 0.0
    reward_list = []
    session_started = True
    print(f"[START] task=emotion-support env=openenv model=gpt-4.1-mini")
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    return placeholder, current_step, total_reward, "Environment Started"

def step_env():
    """Pick a random image from folder and run a step."""
    global current_step, total_reward, reward_list, session_started

    if not session_started:
        start_env()

    error_msg = "null"
    done = False

    try:
        # Pick random image
        img_path = random.choice(image_files)
        frame = np.array(Image.open(img_path).convert("RGB"))

        results = yolo_model.predict(frame, device="cpu", verbose=False)
        boxes = results[0].boxes
        best_box = get_best_box(boxes)

        if best_box is None:
            print(f"[STEP] step={current_step+1} action=detect_face() reward=0.00 done={str(done).lower()} error=No face detected")
            return frame, "No face detected", current_step, total_reward

        x1, y1, x2, y2 = best_box
        face = frame[y1:y2, x1:x2]
        face_pil = Image.fromarray(face)

        # Env step
        result = env.step(face_pil)
        obs = result["obs"]
        reward = result["reward"]
        current_step = obs["steps"]
        total_reward = result["total_reward"]
        reward_list.append(reward)

        # Draw face box
        import cv2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Check if episode ends
        if obs["emotion"] in ["happy", "neutral"] or current_step >= MAX_EPISODE_STEPS:
            done = True

        print(f"[STEP] step={current_step} action=analyze_emotion() reward={reward:.2f} done={str(done).lower()} error=null")

        if done:
            rewards_str = ",".join([f"{r:.2f}" for r in reward_list])
            success_flag = "true" if obs["emotion"] in ["happy","neutral"] else "false"
            print(f"[END] success={success_flag} steps={current_step} rewards={rewards_str}")
            reset_env()

        return frame, obs, current_step, total_reward

    except Exception as e:
        print(f"[STEP] step={current_step+1} action=process_frame() reward=0.00 done={str(done).lower()} error={str(e)}")
        return frame, str(e), current_step, total_reward

def reset_env():
    global current_step, total_reward, reward_list, session_started
    env.reset()
    current_step = 0
    total_reward = 0.0
    reward_list = []
    session_started = False
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    return placeholder, current_step, total_reward, "Environment Reset"
