# syntax=docker/dockerfile:1.7

# ---------- builder: resolve dependencies into a relocatable venv ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
# CPU-only torch keeps the image ~1 GB smaller. For GPU serving, drop --index-url
# and use a CUDA base image instead.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# ---------- runtime: minimal, non-root ----------
FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /models /data && chown app:app /models /data
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CODEFORMER_CHECKPOINT=/models/model.pt \
    CODEFORMER_FEEDBACK=/data/feedback.jsonl \
    CODEFORMER_DEVICE=cpu
WORKDIR /app
COPY --chown=app:app configs ./configs
USER app
VOLUME ["/models", "/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "--factory", "codeformer.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
