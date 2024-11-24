FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY . /app
WORKDIR /app

RUN uv venv \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && uv pip install --compile-bytecode . \
    && uv cache clean

ENV PATH="/app/.venv/bin:$PATH"
