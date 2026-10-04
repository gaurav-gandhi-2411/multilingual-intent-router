# syntax=docker/dockerfile:1
# CPU-only intent-router API. Build context = repo root; `serve_model/` (gitignored) must hold a
# packaged serve dir:  python -m intent_router.package ... --out serve_model
# (+ python -m intent_router.onnx_export --serve-dir serve_model for the onnx backends).
#   docker build -t intent-router-serve .
#   docker run --rm -p 8000:8000 intent-router-serve        # default backend: onnx_fp32
#   docker run --rm -p 8000:8000 -e ROUTER_BACKEND=torch intent-router-serve

# ---- deps: dependency layer first, so code/model changes never re-download torch -----------
FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-serve.txt /tmp/requirements-serve.txt
# torch comes from the CPU wheel index named inside requirements-serve.txt (no CUDA libraries).
RUN pip install -r /tmp/requirements-serve.txt

# ---- runtime ---------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
# Fixed uid so a mounted volume / k8s runAsUser can match it; no shell login, no home needed.
RUN useradd --system --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=deps /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ROUTER_MODEL_DIR=/app/serve_model \
    ROUTER_BACKEND=onnx_fp32
WORKDIR /app
COPY src/ ./src/
COPY --chown=app:app serve_model/ ./serve_model/
USER app
EXPOSE 8000
# /health is 200 only once the model is loaded; start-period covers the ~1 GB weight load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]
CMD ["uvicorn", "intent_router.serve:app", "--host", "0.0.0.0", "--port", "8000"]
