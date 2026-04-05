import os

TEMPERATURE = 0.0
MAX_TOKENS = 1024
MAX_STEPS = 15
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL") or "https://mohammed-altaf-dataanalysis-env.hf.space"
