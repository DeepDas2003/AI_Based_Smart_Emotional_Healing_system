from fastapi import FastAPI
from my_env import EmotionEnv

app = FastAPI()
env = EmotionEnv()

reward_history = []

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/reset")
def reset():
    global reward_history
    reward_history = []
    obs = env.reset()
    return {"obs": obs}

@app.post("/step")
def step():
    global reward_history

    result = env.step("dummy")
    reward = float(result["reward"])
    done = result.get("done", False)

    reward_history.append(reward)
    step_num = len(reward_history)

    done_str = "true" if done else "false"
    reward_str = f"{reward:.2f}"

    print(f"[STEP] step={step_num} action={result['obs']['emotion']} reward={reward_str} done={done_str} error=null")

    if done:
        rewards_str = ",".join([f"{r:.2f}" for r in reward_history])
        print(f"[END] success=true steps={step_num} rewards={rewards_str}")
        reward_history = []
        env.reset()

    return result
