import asyncio
import os
import random
from typing import List, Optional
import time
import numpy as np
import cv2
from PIL import Image

from my_env import EmotionEnv

# ==============================
# ENV VARIABLES (MANDATORY)
# ==============================
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

MODEL_NAME = os.getenv("MODEL_NAME", "emotion-rule-model")

TASK_NAME = "emotion-healing"
BENCHMARK = "custom-env"
MAX_STEPS = 8

# ==============================
# LOGGING FUNCTIONS
# ==============================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, emotion: str, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} emotion={emotion} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ==============================
# LOAD RANDOM IMAGE FROM FOLDER
# ==============================
def load_random_image():
    folder = "emotioanl_faces"

    if not os.path.exists(folder):
        return Image.fromarray(np.zeros((96, 96), dtype=np.uint8))

    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]

    if not files:
        return Image.fromarray(np.zeros((96, 96), dtype=np.uint8))

    img_name = random.choice(files)
    path = os.path.join(folder, img_name)

    img = cv2.imread(path)

    if img is None:
        return Image.fromarray(np.zeros((96, 96), dtype=np.uint8))

    # Convert to grayscale (your model expects this)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return Image.fromarray(gray)


# ==============================
# USE ENV'S OWN ADVICE
# ==============================
def get_action(obs):
    return obs.get("advice", "Stay calm")


# ==============================
# MAIN LOOP
# ==============================
async def main():
    env = EmotionEnv()

    rewards = []
    steps_taken = 0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        env.reset()

        for step in range(1, MAX_STEPS + 1):

            # 🔥 Load random real image
            img = load_random_image()

            result = env.step(img)

            obs = result["obs"]
            reward = float(result["reward"])
            done = result["done"]

            action = f"{obs['emotion']} | total={result['total_reward']:.2f} | {get_action(obs)}"

            rewards.append(reward)
            steps_taken = step

            emotion = obs["emotion"]
            log_step(step, emotion, action, reward, done, None)

            if done:
                break

        # Success condition
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success = avg_reward > 0.2

    except Exception as e:
        log_step(0, "error", 0.00, True, str(e))
        success = False

    finally:
        log_end(success, steps_taken, rewards)


# ==============================
# RUN
# ==============================


if __name__ == "__main__":
    start_time = time.time()
    MAX_RUNTIME = 600  # 10 minutes

    #  Phase 1: Active evaluation (logs visible)
    while time.time() - start_time < MAX_RUNTIME:
        asyncio.run(main())
        time.sleep(3)

    print("[INFO] Evaluation phase completed. Switching to live heartbeat mode...", flush=True)

    #  Phase 2: Keep alive + visible activity
    counter = 0
    while True:
        counter += 1
        print(f"[HEARTBEAT] system_alive tick={counter}", flush=True)
        time.sleep(10)
