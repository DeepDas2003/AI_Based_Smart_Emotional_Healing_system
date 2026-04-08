import torch
from ultralytics.nn.tasks import DetectionModel

# Fix for torch safe loading
torch.serialization.add_safe_globals([DetectionModel])

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from my_env import EmotionEnv

# ==============================
# Config
# ==============================
YOLO_PATH = "yolov8n-face-lindevs.pt"
TASK_NAME = "emotion-healing"
ENV_NAME = "custom-env"
MODEL_NAME = "local-emotion-model"

# ==============================
# Load
# ==============================
yolo_model = YOLO(YOLO_PATH)
env = EmotionEnv()

# Global tracking
reward_history = []
started = False

# ==============================
# Helper
# ==============================
def get_best_box(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    return tuple(map(int, boxes[0].xyxy[0]))

# ==============================
# Main Function
# ==============================
def process_frame(frame):
    global reward_history, started

    if frame is None:
        return None, "No image"

    try:
        # START (only once)
        if not started:
            print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")
            started = True

        # Convert
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO detection
        results = yolo_model.predict(frame, device="cpu", verbose=False)
        boxes = results[0].boxes
        best_box = get_best_box(boxes)

        if best_box is None:
            return Image.fromarray(frame_rgb), "No face detected"

        x1, y1, x2, y2 = best_box

        # Crop + grayscale
        face = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_pil = Image.fromarray(gray)

        # Env step
        result = env.step(face_pil)
        obs = result["obs"]
        reward = float(result["reward"])
        done = result.get("done", False)

        reward_history.append(reward)
        step_num = len(reward_history)

        # FORMAT FIXES
        done_str = "true" if done else "false"
        reward_str = f"{reward:.2f}"

        # STEP LOG
        print(f"[STEP] step={step_num} action={obs['emotion']} reward={reward_str} done={done_str} error=null")

        # END LOG
        if done:
            success = obs["emotion"] in ["neutral", "happy"]
            success_str = "true" if success else "false"

            rewards_str = ",".join([f"{r:.2f}" for r in reward_history])

            print(f"[END] success={success_str} steps={step_num} rewards={rewards_str}")

            # Reset
            reward_history = []
            started = False
            env.reset()

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # UI Output (separate from logs)
        ui_text = (
            f"Step: {obs['steps']}\n"
            f"Emotion: {obs['emotion']}\n"
            f"Confidence: {obs['confidence']}\n"
            f"Reward: {reward_str}\n"
            f"Total Reward: {result['total_reward']:.2f}\n"
            f"Advice: {obs['advice']}"
        )

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ui_text

    except Exception as e:
        step_num = len(reward_history) + 1

        print(f"[STEP] step={step_num} action=null reward=0.00 done=true error={str(e)}")
        print(f"[END] success=false steps={step_num} rewards=0.00")

        reward_history.clear()
        started = False
        env.reset()

        return None, f"Error: {str(e)}"

# ==============================
# Reset
# ==============================
def reset_env():
    global reward_history, started
    reward_history = []
    started = False
    env.reset()
    return None, "Environment Reset"

# ==============================
# UI
# ==============================
with gr.Blocks() as demo:
    gr.Markdown("## Emotion Healing System (Webcam + Evaluator Logs)")

    webcam = gr.Image(sources=["webcam"], type="numpy")
    output = gr.Textbox(lines=12)

    with gr.Row():
        gr.Button("Detect Emotion").click(process_frame, webcam, [webcam, output])
        gr.Button("Reset").click(reset_env, None, [webcam, output])

demo.launch(server_name="0.0.0.0", server_port=7860)
