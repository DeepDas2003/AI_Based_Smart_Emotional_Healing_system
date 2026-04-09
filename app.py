import gradio as gr
import numpy as np
from inference import start_env, step_env, reset_env

frame_placeholder, current_step, total_reward, _ = start_env()

def process_step(frame):
    global frame_placeholder, current_step, total_reward

    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        return placeholder, "⚠️ Please capture image from webcam", current_step, total_reward

    frame_out, obs, current_step, total_reward = step_env(frame)

    if isinstance(obs, dict):
        text_display = (
            f"Step: {current_step}\n"
            f"Emotion: {obs.get('emotion','unknown')} | Confidence: {obs.get('confidence',0.0):.2f}\n"
            f"Reward this step: {obs.get('reward',0.0):.2f}\n"
            f"Total Reward: {total_reward:.2f}\n"
            f"Advice: {obs.get('advice','')}\n"
            f"Task Status: {obs.get('task_status','')}"
        )
    else:
        text_display = str(obs)

    return frame_out, text_display, current_step, total_reward

def reset_ui():
    global frame_placeholder, current_step, total_reward
    frame_placeholder, current_step, total_reward, message = reset_env()
    return frame_placeholder, message, current_step, total_reward

with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Emotion Detection (Episode Mode)")

    webcam = gr.Image(sources=["webcam"], type="numpy")
    output = gr.Textbox(lines=12, label="Step Info")
    step_display = gr.Number(label="Current Step", value=0)
    reward_display = gr.Number(label="Total Reward", value=0.0)

    with gr.Row():
        gr.Button("▶️ Capture & Step").click(
            process_step, webcam, [webcam, output, step_display, reward_display]
        )
        gr.Button("🔄 Reset").click(
            reset_ui, None, [webcam, output, step_display, reward_display]
        )

demo.launch(server_name="0.0.0.0", server_port=7860)

