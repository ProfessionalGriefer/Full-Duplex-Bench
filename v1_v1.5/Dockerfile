FROM python:3.11-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Disable development dependencies
ENV UV_NO_DEV=1

# Required for utmosv2 dependency
RUN apt-get update && \
  apt-get install -y --no-install-recommends git && \
  rm -rf /var/lib/apt/lists/*

# 1. Copy ONLY the dependency files first
COPY pyproject.toml uv.lock ./

# 2. Install dependencies (Docker caches this layer as long as uv.lock doesn't change)
RUN uv sync --locked --no-install-project

# 3. NOW copy the rest of your application code
COPY . /app

# 4. Sync again to install your actual project (this is nearly instant)
RUN uv sync --locked

CMD ["uv", "run", "model_inference/gemini/inference_gemini25_native.py", "--base-dir", "./dataset/v1.5/", "--task", "user_interruption"]
