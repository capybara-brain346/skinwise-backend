FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# uv configuration: compile bytecode and copy (don't symlink) installed packages
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer), using the locked versions
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the application source
COPY . .

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Make the venv the default environment
ENV PATH="/app/.venv/bin:$PATH"

# Railway provides $PORT at runtime; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# The ONNX model is downloaded from S3 on startup if not present in artifacts/.
# Use shell form so $PORT is expanded at runtime.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}