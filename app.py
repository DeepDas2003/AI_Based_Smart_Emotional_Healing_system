import gradio as gr
from inference import start_env, step_env, reset_env
import numpy as np

frame_placeholder, current_step, total_reward, _ = start_env()

def process_step():
    """Run next step using random image from folder."""
    global frame_placeholder, current_step, total_reward
    frame_out, obs, current_step, total_reward = step_env()

    if isinstance(obs, dict):
        task_status = obs.get('task_status', '')
        # If task_status is empty, show placeholder like "In Progress"
        task_status = task_status if task_status else "In Progress"

        text_display = (
            f"Step: {current_step}\n"
            f"Emotion: {obs.get('emotion','unknown')} | Confidence: {obs.get('confidence',0.0):.2f}\n"
            f"Reward this step: {obs.get('reward',0.0):.2f}\n"
            f"Total Reward: {total_reward:.2f}\n"
            f"Advice: {obs.get('advice','')}\n"
            f"Task Status: {task_status}"
        )
    else:
        text_display = str(obs)

    return frame_out, text_display, current_step, total_reward

def reset_ui():
    global frame_placeholder, current_step, total_reward
    frame_placeholder, current_step, total_reward, message = reset_env()
    return frame_placeholder, message, current_step, total_reward

with gr.Blocks() as demo:
    gr.Markdown("## 🎥 Emotion Detection (Random Image Episode Mode)")

    img_display = gr.Image(value=np.zeros((480,640,3), dtype=np.uint8), type="numpy")
    output = gr.Textbox(lines=12, label="Step Info")
    step_display = gr.Number(label="Current Step", value=0)
    reward_display = gr.Number(label="Total Reward", value=0.0)

    with gr.Row():
        gr.Button("▶️ Next Step").click(
            process_step, None, [img_display, output, step_display, reward_display]
        )
        gr.Button("🔄 Reset").click(
            reset_ui, None, [img_display, output, step_display, reward_display]
        )

demo.launch(server_name="0.0.0.0", server_port=7860)

