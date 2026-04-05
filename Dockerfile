ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY models.py client.py __init__.py baseline.py inference.py openenv.yaml ./
COPY server/ ./server/
COPY tasks/ ./tasks/
COPY datasets/ ./datasets/
COPY helpers/ ./helpers/

# Install dependencies
RUN uv sync --frozen --no-dev

# HF Spaces runs containers as a non-root user on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
