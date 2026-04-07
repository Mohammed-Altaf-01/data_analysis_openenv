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

# Install dependencies into the uv-managed venv
RUN uv sync --frozen --no-dev

# Make the venv's python/pip the default so `python inference.py` works
# without needing `uv run` as a prefix
ENV PATH="/app/.venv/bin:$PATH"

# Ensure local modules (client, models, helpers, tasks) are always importable
# regardless of the working directory the evaluator uses
ENV PYTHONPATH="/app:$PYTHONPATH"

# HF Spaces runs containers as a non-root user on port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
