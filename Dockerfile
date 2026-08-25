FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/home/app/.cache/torchinductor

RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg g++ libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home app

WORKDIR /app

COPY requirements.txt ./

# Explicit CPU wheels avoid pulling CUDA runtime libraries into the default image.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 torchaudio==2.6.0 \
    && python -m pip install -r requirements.txt

COPY --chown=app:app app ./app
RUN mkdir -p "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" \
    && chown -R app:app /home/app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
