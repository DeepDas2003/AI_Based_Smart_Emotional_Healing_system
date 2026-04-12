from fastapi import FastAPI
from fastapi.responses import FileResponse
from ultralytics import YOLO
from my_env import EmotionEnv

from inference import router

app = FastAPI()

# =========================
# LOAD MODELS (ONCE)
# =========================
@app.on_event("startup")
def load_models():
    print("[INFO] Loading models...", flush=True)

    app.state.yolo = YOLO("yolov8n-face-lindevs.pt")
    app.state.env = EmotionEnv()

    app.state.step = 0
    app.state.rewards = []

    print("[SUCCESS] ALL MODELS LOADED", flush=True)

# =========================
# ROUTES
# =========================
app.include_router(router, prefix="/api")

@app.get("/")
def home():
    return FileResponse("index.html")


# =========================
# ENTRY POINT
# =========================
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )


if __name__ == "__main__":
    main()
