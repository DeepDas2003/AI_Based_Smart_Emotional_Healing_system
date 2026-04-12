import uvicorn
from fastapi import FastAPI

# =========================
# MAIN APP (ALWAYS RUNNING)
# =========================
app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}


# =========================
# INFERENCE LOADER (SAFE)
# =========================
inference_app = None

def load_inference():
    global inference_app
    try:
        # Lazy import to prevent startup crash
        from inference import app as loaded_app
        inference_app = loaded_app
        print("[INFO] Inference loaded successfully", flush=True)

    except Exception as e:
        print("[ERROR] Inference failed:", repr(e), flush=True)
        inference_app = None


# =========================
# STARTUP EVENT
# =========================
@app.on_event("startup")
def startup():
    load_inference()


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
