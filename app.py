# app_openenv_manual.py
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
# Environment + State
# =========================
env = EmotionEnv()
current_step = 0
total_reward = 0.0
session_started = False

# =========================
# Step Processor
# =========================
def process_step(frame):
    global current_step, total_reward, session_started

    if frame is None:
        return None, "⚠️ Please capture image from webcam", current_step, total_reward

    if not session_started:
        print("OpenEnv Start (POST OK)")
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

    result = env.step(face_pil)
    obs = result["obs"]
    reward = result["reward"]
    current_step = obs["steps"]
    total_reward = result["total_reward"]

    print(f"[STEP] step={current_step} emotion={obs['emotion']} reward={reward:.2f} total_reward={total_reward:.2f}")
    print("OpenEnv Step (POST OK)")

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    task_status = ""
    # Auto-reset if happy or neutral
    if obs["emotion"] in ["happy", "neutral"]:
        if current_step <= 4:
            task_status = "Task 1 (Easy) Completed"
        elif current_step <= 8:
            task_status = "Task 2 (Medium) Completed"
        else:
            task_status = "Task 3 (Hard) Completed"

        print(f"[END] total_reward={total_reward:.2f} | {task_status}")
        env.reset()
        print("OpenEnv Reset (POST OK)")
        session_started = False
        current_step = 0
        total_reward = 0.0

    output = (
        f"Step: {obs['steps']}\n"
        f"Emotion: {obs['emotion']} | Confidence: {obs['confidence']:.2f}\n"
        f"Reward this step: {reward:.2f}\n"
        f"Total Reward: {total_reward:.2f}\n"
        f"Task Status: {task_status}\n"
        f"Advice: {obs['advice']}"
    )

    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), output, current_step, total_reward

# =========================
# Reset Function
# =========================
def reset_env():
    global current_step, total_reward, session_started
    env.reset()
    print("OpenEnv Reset (POST OK)")
    current_step = 0
    total_reward = 0.0
    session_started = False
    return None, "Environment Reset", current_step, total_reward

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Emotion Detection (HF Space + OpenEnv Manual Step)")

    webcam = gr.Image(sources=["webcam"], type="numpy")
    output = gr.Textbox(lines=12)
    step_display = gr.Number(label="Current Step", value=0)
    reward_display = gr.Number(label="Total Reward", value=0.0)

    with gr.Row():
        gr.Button("▶️ Capture & Step").click(
            process_step, webcam, [webcam, output, step_display, reward_display]
        )
        gr.Button("🔄 Reset").click(
            reset_env, None, [webcam, output, step_display, reward_display]
        )

demo.launch(server_name="0.0.0.0", server_port=7860)
