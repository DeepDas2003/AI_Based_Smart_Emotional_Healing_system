import torch
import cv2
from PIL import Image
import gradio as gr
from ultralytics import YOLO
from my_env import EmotionEnv

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
# Environment + Global Counters
# =========================
env = EmotionEnv()
current_step = 0
total_reward = 0.0
session_started = False

# =========================
# Step function
# =========================
def process_step(frame):
    global current_step, total_reward, session_started
    if frame is None:
        return None, "⚠️ Please capture image from webcam", current_step, total_reward

    # Start logging
    if not session_started:
        print("[START] task=emotion-healing env=custom-env")
        session_started = True

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(frame, device="cpu", verbose=False)
    boxes = results[0].boxes
    best_box = get_best_box(boxes)
    if best_box is None:
        return Image.fromarray(frame_rgb), "❌ No face detected", current_step, total_reward

    x1, y1, x2, y2 = best_box
    face = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_pil = Image.fromarray(gray)

    # Step environment
    result = env.step(face_pil)
    obs = result["obs"]
    reward = result["reward"]
    current_step = obs["steps"]
    total_reward = result["total_reward"]

    # Log to HF console
    print(f"[STEP] step={current_step} emotion={obs['emotion']} reward={reward:.2f} total_reward={total_reward:.2f} task_status={result['task_status']}")

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    output = (
        f"Step: {current_step}\n"
        f"Emotion: {obs['emotion']} | Confidence: {obs['confidence']:.2f}\n"
        f"Reward this step: {reward:.2f}\n"
        f"Total Reward: {total_reward:.2f}\n"
        f"Task Status: {result['task_status']}\n"
        f"Advice: {obs['advice']}"
    )
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), output, current_step, total_reward

def reset_env():
    global current_step, total_reward, session_started
    env.reset()
    print(f"[END] total_reward={total_reward:.2f}")
    current_step = 0
    total_reward = 0.0
    session_started = False
    return None, "Environment Reset", current_step, total_reward

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Emotion Detection using Webcam (Step & Reward Tracker)")

    webcam = gr.Image(sources=["webcam"], type="numpy")
    output = gr.Textbox(lines=12)
    step_display = gr.Number(label="Current Step", value=0)
    reward_display = gr.Number(label="Total Reward", value=0.0)

    with gr.Row():
        gr.Button("▶️ Step").click(
            process_step, webcam, [webcam, output, step_display, reward_display]
        )
        gr.Button("🔄 Reset").click(
            reset_env, None, [webcam, output, step_display, reward_display]
        )

demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
