# syntax=docker/dockerfile:1

# ---- Build stage: resolve exact dependencies for the target platform ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- Runtime stage: minimal image for RapidAPI ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:${PATH}" \
    PYTHONPATH="/install/lib/python3.12/site-packages:${PYTHONPATH}" \
    PORT=8000

# Non-root user (RapidAPI runs as root, but this is good practice)
RUN useradd --create-home --uid 1001 appuser
WORKDIR /app

COPY --from=builder /install /install
COPY main.py models.py solver.py ./

USER appuser

EXPOSE 8000

# RapidAPI requires the container to listen on $PORT (default 8000).
# Health check hits /health until the app is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen('http://localhost:%s/health' % p).status==200 else 1)"

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
