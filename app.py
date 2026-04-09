import cv2
import numpy as np
from PIL import Image
import gradio as gr
import inference  # this is your root-level inference.py

# =========================
# Gradio Step Processor Wrapper
# =========================
def process_step(frame):
    """
    Receives webcam frame and calls inference.step_env()
    """
    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        return placeholder, "⚠️ Please capture an image", 0, 0.0

    frame_array = np.array(frame)
    out_frame, obs, step, total_reward = inference.step_env(frame_array)

    # Display simple info
    if isinstance(obs, dict):
        emotion = obs.get("emotion", "unknown")
        confidence = obs.get("confidence", 0.0)
        advice = obs.get("advice", "")
        output_text = (
            f"Step: {step}\n"
            f"Emotion: {emotion} | Confidence: {confidence:.2f}\n"
            f"Total Reward: {total_reward:.2f}\n"
            f"Advice: {advice}"
        )
    else:
        output_text = str(obs)

    return out_frame, output_text, step, total_reward

# =========================
# Reset Wrapper
# =========================
def reset_env():
    """
    Calls inference.reset_env() to manually reset environment
    """
    out_frame, step, total_reward = inference.reset_env()
    output_text = "Environment Reset"
    return out_frame, output_text, step, total_reward

# =========================
# Gradio UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Emotion Detection (Local Test + HF Phase 1 Compatible)")

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
