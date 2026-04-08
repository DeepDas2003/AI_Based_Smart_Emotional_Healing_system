import os
from openai import OpenAI

# =========================
# Environment Variables
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Mandatory check
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# =========================
# Client Initialization
# =========================
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# =========================
# Inference Function
# =========================
def run_inference(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    output = run_inference("Hello from OpenEnv!")
    print(output)
