# syntax=docker/dockerfile:1

# Python 3.10 matches the environment that trains the model: the served
# artifact is a pickle, so the runtime must mirror the training stack.
FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-api.txt /tmp/requirements-api.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --requirement /tmp/requirements-api.txt

FROM python:3.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PORT=8000

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY config/ ./config/
COPY src/ ./src/
# The model directory may be empty: the API then starts and reports itself as
# unavailable instead of failing, so the image can be built before training.
COPY models/ ./models/

# Run as an unprivileged user; UID 1000 is also what Hugging Face Spaces expects.
RUN useradd --create-home --uid 1000 appuser \
    && chown --recursive appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; \
url = f\"http://127.0.0.1:{os.environ['PORT']}/health\"; \
urllib.request.urlopen(url, timeout=3)"

CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port \"${PORT}\""]
