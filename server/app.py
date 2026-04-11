from fastapi import FastAPI
from inference import app as inference_app

# OpenEnv expects this structure
app = inference_app
