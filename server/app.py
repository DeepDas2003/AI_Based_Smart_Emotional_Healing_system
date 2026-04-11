import uvicorn
from fastapi import FastAPI

# Fallback app (if inference fails)
fallback_app = FastAPI()

@fallback_app.get("/")
def fallback():
    return {"status": "error loading inference"}

# Try to import your main app
try:
    from inference import app as inference_app
    app = inference_app
    print("[INFO] Inference app loaded successfully", flush=True)

except Exception as e:
    print(f"[ERROR] Failed to load inference app: {e}", flush=True)
    app = fallback_app  # prevent crash

# =========================
# OpenEnv Entry Point
# =========================
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860
    )

# =========================
# Required callable
# =========================
if __name__ == "__main__":
    main()
