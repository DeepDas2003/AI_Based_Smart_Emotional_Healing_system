from inference import app as inference_app
import uvicorn

# Expose FastAPI app
app = inference_app

# =========================
# OpenEnv Entry Point
# =========================
def main():
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )

# =========================
# Required callable
# =========================
if __name__ == "__main__":
    main()
